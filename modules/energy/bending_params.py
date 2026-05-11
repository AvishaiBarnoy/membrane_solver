"""Parameter and configuration helpers for Helfrich/Willmore bending energy."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from geometry.entities import Mesh

logger = logging.getLogger("membrane_solver")

BendingEnergyModel = Literal["willmore", "helfrich"]
BendingGradientMode = Literal["approx", "finite_difference", "analytic"]


def _energy_model(global_params) -> BendingEnergyModel:
    model = str(global_params.get("bending_energy_model", "helfrich") or "helfrich")
    model = model.lower().strip()
    return "helfrich" if model == "helfrich" else "willmore"


def _gradient_mode(global_params) -> BendingGradientMode:
    mode = str(global_params.get("bending_gradient_mode", "analytic") or "analytic")
    mode = mode.lower().strip()
    if mode in {"fd", "finite_difference"}:
        return "finite_difference"
    if mode == "analytic":
        return "analytic"
    return "approx"


def _spontaneous_curvature(global_params) -> float:
    val = global_params.get("spontaneous_curvature")
    if val is None:
        val = global_params.get("intrinsic_curvature", 0.0)
    return float(val or 0.0)


def _per_vertex_params(
    mesh: Mesh,
    global_params,
    *,
    model: BendingEnergyModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-vertex (kappa, c0) arrays in vertex-row order.

    Allows local overrides via `vertex.options`:
    - `bending_modulus`
    - `spontaneous_curvature` (alias: `intrinsic_curvature`)
    """
    mesh.build_position_cache()

    n = len(mesh.vertex_ids)
    kappa_default = float(global_params.get("bending_modulus", 0.0) or 0.0)
    if model == "helfrich":
        c0_default = _spontaneous_curvature(global_params)
    else:
        c0_default = 0.0

    cache_key = (
        mesh._vertex_ids_version,
        model,
        float(kappa_default),
        float(c0_default),
    )
    cached = getattr(mesh, "_bending_vertex_param_cache", None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["kappa"], cached["c0"]

    kappa = np.full(n, kappa_default, dtype=float)
    c0 = np.full(n, c0_default, dtype=float)

    override_rows_k: list[int] = []
    override_vals_k: list[float] = []
    override_rows_c0: list[int] = []
    override_vals_c0: list[float] = []

    for vid, vertex in mesh.vertices.items():
        row = mesh.vertex_index_to_row.get(int(vid))
        if row is None:
            continue
        opts = getattr(vertex, "options", None) or {}
        if "bending_modulus" in opts:
            try:
                override_rows_k.append(row)
                override_vals_k.append(float(opts["bending_modulus"]))
            except (TypeError, ValueError):
                pass
        if model == "helfrich":
            if "spontaneous_curvature" in opts:
                try:
                    override_rows_c0.append(row)
                    override_vals_c0.append(float(opts["spontaneous_curvature"]))
                except (TypeError, ValueError):
                    pass
            elif "intrinsic_curvature" in opts:
                try:
                    override_rows_c0.append(row)
                    override_vals_c0.append(float(opts["intrinsic_curvature"]))
                except (TypeError, ValueError):
                    pass

    if override_rows_k:
        kappa[np.asarray(override_rows_k, dtype=int)] = np.asarray(
            override_vals_k, dtype=float
        )
    if model == "helfrich" and override_rows_c0:
        c0[np.asarray(override_rows_c0, dtype=int)] = np.asarray(
            override_vals_c0, dtype=float
        )

    mesh._bending_vertex_param_cache = {"key": cache_key, "kappa": kappa, "c0": c0}
    return kappa, c0
