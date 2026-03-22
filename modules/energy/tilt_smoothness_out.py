"""Outer-leaflet tilt smoothness energy module.

This module applies a Dirichlet energy to the outer-leaflet tilt field:

    E = 1/2 * k_s * ∫ |∇t_out|^2 dA

The stiffness defaults to ``bending_modulus`` unless an explicit
``bending_modulus_out`` is provided.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh
from modules.energy import tilt_smoothness as _base
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)

USES_TILT_LEAFLETS = True


def _resolve_bending_modulus(param_resolver) -> float:
    k = param_resolver.get(None, "bending_modulus_out")
    if k is None:
        k = param_resolver.get(None, "bending_modulus")
    return float(k or 0.0)


def _masked_weights_and_tris(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return cached outer-leaflet masked smoothness payload."""
    weights, tri_rows = _base._get_weights_and_tris(
        mesh, positions=positions, index_map=index_map
    )
    if tri_rows is None:
        return None, None

    absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet="out")
    absent_token = id(absent_mask)
    use_cache = mesh._geometry_cache_active(positions)
    cache_key = (
        int(mesh._version),
        int(mesh._facet_loops_version),
        int(mesh._vertex_ids_version),
        id(positions),
        id(weights),
        id(tri_rows),
        absent_token,
    )
    cache_attr = "_tilt_smoothness_out_mask_cache"
    if use_cache:
        cached = getattr(mesh, cache_attr, None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["weights"], cached["tri_rows"]

    tri_keep = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_mask
    )
    if tri_keep.size and not np.any(tri_keep):
        weights_use = np.zeros((0, 3), dtype=float)
        tri_rows_use = np.zeros((0, 3), dtype=np.int32)
    elif tri_keep.size:
        weights_use = weights[tri_keep]
        tri_rows_use = tri_rows[tri_keep]
    else:
        weights_use = weights
        tri_rows_use = tri_rows

    if use_cache:
        setattr(
            mesh,
            cache_attr,
            {"key": cache_key, "weights": weights_use, "tri_rows": tri_rows_use},
        )
    return weights_use, tri_rows_use


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver, *, compute_gradient: bool = True
) -> (
    Tuple[float, Dict[int, np.ndarray]]
    | Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Dict-based wrapper returning (E, shape_grad[, tilt_grad])."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions) if compute_gradient else None
    energy = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts_out=None,
        tilt_out_grad_arr=tilt_grad_arr,
    )
    if not compute_gradient:
        return float(energy), {}

    shape_grad = {
        int(vid): grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }
    tilt_grad = {
        int(vid): tilt_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if tilt_grad_arr is not None and np.any(tilt_grad_arr[row])
    }
    return float(energy), shape_grad, tilt_grad


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray,
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
    tilt_in_grad_arr: np.ndarray | None = None,
    tilt_out_grad_arr: np.ndarray | None = None,
    ctx=None,
) -> float:
    """Dense-array outer-leaflet smoothness energy accumulation."""
    _ = index_map, tilts_in, tilt_in_grad_arr, grad_arr
    k_smooth = _resolve_bending_modulus(param_resolver)
    if k_smooth == 0.0:
        return 0.0

    weights, tri_rows = _masked_weights_and_tris(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
    )
    if tri_rows is None:
        return 0.0
    if tri_rows.size == 0:
        return 0.0

    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()
    else:
        tilts_out = np.asarray(tilts_out, dtype=float)
        if tilts_out.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")

    if tilt_out_grad_arr is not None:
        tilt_out_grad_arr = np.asarray(tilt_out_grad_arr, dtype=float)
        if tilt_out_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_out_grad_arr must have shape (N_vertices, 3)")

    return _base._compute_smoothness_energy_and_gradient(
        mesh,
        k_smooth=k_smooth,
        positions=positions,
        tri_rows=tri_rows,
        weights=weights,
        tilts=tilts_out,
        tilt_grad_arr=tilt_out_grad_arr,
        transport_model=_base._resolve_transport_model(global_params),
        ctx=ctx,
        scratch_tag="tilt_smoothness_out",
    )


def compute_energy_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
    ctx=None,
) -> float:
    """Dense-array outer-leaflet smoothness energy (energy only)."""
    _ = index_map, tilts_in
    k_smooth = _resolve_bending_modulus(param_resolver)
    if k_smooth == 0.0:
        return 0.0

    weights, tri_rows = _masked_weights_and_tris(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
    )
    if tri_rows is None:
        return 0.0
    if tri_rows.size == 0:
        return 0.0

    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()
    else:
        tilts_out = np.asarray(tilts_out, dtype=float)
        if tilts_out.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")

    return _base._compute_smoothness_energy_and_gradient(
        mesh,
        k_smooth=k_smooth,
        positions=positions,
        tri_rows=tri_rows,
        weights=weights,
        tilts=tilts_out,
        tilt_grad_arr=None,
        transport_model=_base._resolve_transport_model(global_params),
        ctx=ctx,
        scratch_tag="tilt_smoothness_out",
    )


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]
