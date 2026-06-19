"""Parameter and config helpers for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh
from modules.energy.bending_params import _spontaneous_curvature

_ASSUME_J0_PRESETS_KEY = "bending_tilt_assume_J0_presets"


def _use_inner_recovered_divergence(global_params, *, cache_tag: str) -> bool:
    """Return whether recovered inner divergence is enabled for this run."""
    if str(cache_tag) != "in" or global_params is None:
        return False
    return bool(str(global_params.get("theory_parity_lane") or "").strip())


def _assume_J0_presets(global_params, *, cache_tag: str) -> tuple[str, ...]:
    """Optional config: presets for which the Helfrich base term is set to zero."""
    if global_params is None:
        return ()
    raw = global_params.get(f"{_ASSUME_J0_PRESETS_KEY}_{cache_tag}")
    if raw is None:
        raw = global_params.get(_ASSUME_J0_PRESETS_KEY)
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [raw]
    else:
        try:
            items = list(raw)
        except TypeError:
            items = [raw]

    presets: list[str] = []
    for item in items:
        name = str(item).strip()
        if name:
            presets.append(name)
    return tuple(presets)


def _assume_J0_radius_max(global_params, *, cache_tag: str) -> float | None:
    """Optional config: radial cap for theory-mode J0 suppression rows."""
    if global_params is None:
        return None
    raw = global_params.get(f"{_ASSUME_J0_PRESETS_KEY}_radius_max_{cache_tag}")
    if raw is None:
        raw = global_params.get(f"{_ASSUME_J0_PRESETS_KEY}_radius_max")
    if raw is None:
        return None
    radius_max = float(raw)
    if radius_max < 0.0:
        raise ValueError("bending_tilt_assume_J0_presets_radius_max must be >= 0.")
    return radius_max


def _assume_J0_center_xy(global_params) -> np.ndarray:
    """Return the xy center used for radial J0-suppression clipping."""
    if global_params is None:
        return np.zeros(2, dtype=float)
    raw = global_params.get("tilt_thetaB_center")
    if raw is None:
        raw = global_params.get("pin_to_circle_point")
    if raw is None:
        return np.zeros(2, dtype=float)
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size < 2:
        return np.zeros(2, dtype=float)
    return arr[:2].astype(float, copy=False)


def _base_term_region_mode(global_params) -> str:
    """Return optional benchmark-scoped base-term region mode."""
    if global_params is None:
        return "off"
    raw = global_params.get("bending_tilt_base_term_region_mode")
    mode = str(raw or "off").strip().lower()
    if mode not in {"off", "physical_disk_split_v1", "disk_only_base_term_v1"}:
        raise ValueError(
            "bending_tilt_base_term_region_mode must be 'off' or "
            "'physical_disk_split_v1' or 'disk_only_base_term_v1'."
        )
    return mode


def _base_term_reference_mode(global_params, *, cache_tag: str | None = None) -> str:
    """Return the reference mode used for the Helfrich base curvature term."""
    if global_params is None:
        return "current_geometry"
    raw = None
    if cache_tag is not None:
        raw = global_params.get(f"bending_tilt_base_term_reference_mode_{cache_tag}")
    if raw is None:
        raw = global_params.get("bending_tilt_base_term_reference_mode")
    mode = str(raw or "current_geometry").strip().lower()
    if mode not in {"current_geometry", "flat_reference_zero_j0"}:
        raise ValueError(
            "bending_tilt_base_term_reference_mode must be "
            "'current_geometry' or 'flat_reference_zero_J0'."
        )
    return mode


def _bending_tilt_interface_divergence_mode(
    global_params, *, cache_tag: str | None = None
) -> str:
    """Return optional scaffold-interface divergence reconstruction mode."""
    if global_params is None:
        return "p1_triangle"
    raw = None
    if cache_tag is not None:
        raw = global_params.get(f"bending_tilt_interface_divergence_mode_{cache_tag}")
    if raw is None and cache_tag == "out":
        raw = global_params.get("bending_tilt_out_interface_divergence_mode")
    if raw is None:
        raw = global_params.get("bending_tilt_interface_divergence_mode")
    mode = str(raw or "p1_triangle").strip().lower()
    if mode not in {"p1_triangle", "trace_reconstructed_v1"}:
        raise ValueError(
            "bending_tilt_out_interface_divergence_mode must be "
            "'p1_triangle' or 'trace_reconstructed_v1'."
        )
    return mode


def _bending_tilt_in_scaffold_shape_stencil_mode(global_params) -> str:
    """Return opt-in scaffold trace treatment for inner shape gradients."""
    if global_params is None:
        return "off"
    raw = global_params.get("bending_tilt_in_scaffold_shape_stencil_mode")
    mode = str(raw or "off").strip().lower()
    if mode not in {"off", "trace_boundary_v1"}:
        raise ValueError(
            "bending_tilt_in_scaffold_shape_stencil_mode must be "
            "'off' or 'trace_boundary_v1'."
        )
    return mode


def _base_term_region_radius(global_params) -> float | None:
    """Return physical disk radius used by base-term region modes."""
    if global_params is None:
        return None
    raw = global_params.get("bending_tilt_base_term_region_radius")
    if raw is None:
        return None
    radius = float(raw)
    if radius < 0.0:
        raise ValueError("bending_tilt_base_term_region_radius must be >= 0.")
    return radius


def _bending_tilt_in_update_mode(global_params) -> str:
    """Return optional benchmark-scoped inner bending-tilt update mode."""
    if global_params is None:
        return "off"
    raw = global_params.get("bending_tilt_in_update_mode")
    mode = str(raw or "off").strip().lower()
    if mode not in {
        "off",
        "outer_near_divergence_cap_v1",
        "radial_cross_term_off_v1",
    }:
        raise ValueError(
            "bending_tilt_in_update_mode must be 'off' or "
            "'outer_near_divergence_cap_v1' or 'radial_cross_term_off_v1'."
        )
    return mode


def _use_stage_a_inner_shape_cross_suppression(
    mesh: Mesh, global_params, *, cache_tag: str
) -> bool:
    """Return whether to suppress the inner shape-gradient cross term for Stage A."""
    if str(cache_tag) != "in" or global_params is None:
        return False
    lane = str(global_params.get("theory_parity_lane") or "").strip().lower()
    return lane == "stage_a_emergent"


def _use_stage_a_outer_grad_linear_transition_operator(
    global_params, *, cache_tag: str
) -> bool:
    """Return whether the Stage A outer grad_linear transition operator is enabled."""
    if str(cache_tag) != "out" or global_params is None:
        return False
    lane = str(global_params.get("theory_parity_lane") or "").strip().lower()
    return lane == "stage_a_emergent"


def _base_term_boundary_group(global_params, *, cache_tag: str) -> str | None:
    """Optional config: treat a tagged interface ring as a base-term boundary."""
    if global_params is None:
        return None
    raw = global_params.get(f"bending_tilt_base_term_boundary_group_{cache_tag}")
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _resolve_bending_modulus(global_params, kappa_key: str) -> float:
    """Return the leaflet-specific bending modulus or the global default."""
    val = global_params.get(kappa_key)
    if val is None:
        val = global_params.get("bending_modulus", 0.0)
    return float(val or 0.0)


def _per_vertex_params_leaflet(
    mesh: Mesh,
    global_params,
    *,
    model: str,
    kappa_key: str,
    cache_tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-vertex (kappa, c0) arrays using a leaflet default modulus."""
    mesh.build_position_cache()

    n = len(mesh.vertex_ids)
    kappa_default = _resolve_bending_modulus(global_params, kappa_key)

    # Resolve leaflet-specific spontaneous curvature default
    c0_key = f"spontaneous_curvature_{cache_tag}"
    c0_default = global_params.get(c0_key)
    if c0_default is None:
        c0_default = (
            _spontaneous_curvature(global_params) if model == "helfrich" else 0.0
        )
    c0_default = float(c0_default or 0.0)

    cache_key = (
        mesh._vertex_ids_version,
        model,
        float(kappa_default),
        float(c0_default),
    )
    cache_attr = f"_bending_leaflet_param_cache_{cache_tag}"
    cached = getattr(mesh, cache_attr, None)
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
        if kappa_key in opts:
            try:
                override_rows_k.append(row)
                override_vals_k.append(float(opts[kappa_key]))
            except (TypeError, ValueError):
                pass
        elif "bending_modulus" in opts:
            try:
                override_rows_k.append(row)
                override_vals_k.append(float(opts["bending_modulus"]))
            except (TypeError, ValueError):
                pass

        if model == "helfrich":
            # Per-vertex c0 resolution: leaflet-specific -> generic
            v_c0 = opts.get(c0_key)
            if v_c0 is None:
                v_c0 = opts.get("spontaneous_curvature")
            if v_c0 is None:
                v_c0 = opts.get("intrinsic_curvature")

            if v_c0 is not None:
                try:
                    override_rows_c0.append(row)
                    override_vals_c0.append(float(v_c0))
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

    setattr(mesh, cache_attr, {"key": cache_key, "kappa": kappa, "c0": c0})
    return kappa, c0
