"""Unified tilt smoothness energy module for inner and outer leaflets."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh
from modules.energy import tilt_smoothness as _base
from modules.energy.tilt_smoothness_utils import (
    _masked_weights_and_tris,
    _resolve_smoothness_rigidity,
)


def compute_energy_and_gradient_array_leaflet(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray | None,
    ctx=None,
    tilts: np.ndarray | None,
    tilt_grad_arr: np.ndarray | None,
    leaflet: str,
) -> float:
    """Compute tilt smoothness energy and accumulate gradients for a leaflet."""
    _ = grad_arr  # Smoothness usually doesn't provide shape gradients
    k_smooth = _resolve_smoothness_rigidity(param_resolver, leaflet)
    if k_smooth == 0.0:
        return 0.0

    weights, tri_rows = _masked_weights_and_tris(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
        leaflet=leaflet,
    )
    if tri_rows is None or tri_rows.size == 0:
        return 0.0

    if tilts is None:
        if leaflet == "in":
            tilts = mesh.tilts_in_view()
        else:
            tilts = mesh.tilts_out_view()
    else:
        tilts = np.asarray(tilts, dtype=float)
        if tilts.shape != (len(mesh.vertex_ids), 3):
            raise ValueError(
                f"tilts for leaflet '{leaflet}' must have shape (N_vertices, 3)"
            )

    if tilt_grad_arr is not None:
        tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
        if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError(
                f"tilt_grad_arr for leaflet '{leaflet}' must have shape (N_vertices, 3)"
            )

    return _base._compute_smoothness_energy_and_gradient(
        mesh,
        k_smooth=k_smooth,
        positions=positions,
        tri_rows=tri_rows,
        weights=weights,
        tilts=tilts,
        tilt_grad_arr=tilt_grad_arr,
        transport_model=_base._resolve_transport_model(global_params),
        ctx=ctx,
        scratch_tag=f"tilt_smoothness_{leaflet}",
    )


def compute_energy_array_leaflet(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts: np.ndarray | None = None,
    leaflet: str,
    ctx=None,
) -> float:
    """Compute tilt smoothness energy for a leaflet (energy only)."""
    return compute_energy_and_gradient_array_leaflet(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=None,
        ctx=ctx,
        tilts=tilts,
        tilt_grad_arr=None,
        leaflet=leaflet,
    )


def compute_energy_and_gradient_leaflet(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    leaflet: str,
    compute_gradient: bool = True,
) -> (
    Tuple[float, Dict[int, np.ndarray]]
    | Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Return tilt smoothness energy and gradients for a leaflet."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions) if compute_gradient else None

    energy = compute_energy_and_gradient_array_leaflet(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts=None,
        tilt_grad_arr=tilt_grad_arr,
        leaflet=leaflet,
    )

    if not compute_gradient:
        return float(energy), {}

    shape_grad = {
        int(vid): grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }
    tg_dict = {
        int(vid): tilt_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if tilt_grad_arr is not None and np.any(tilt_grad_arr[row])
    }
    return float(energy), shape_grad, tg_dict
