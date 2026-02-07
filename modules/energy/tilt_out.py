"""Outer-leaflet tilt magnitude energy module.

This module models a per-vertex tilt penalty for the outer leaflet:

    E = 1/2 * k_t * sum_v (|t_out,v|^2 * A_v)

where ``t_out,v`` is a 3D tangent tilt vector stored on each vertex and ``A_v``
is a barycentric area weight.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)

USES_TILT_LEAFLETS = True


def _resolve_tilt_modulus(param_resolver) -> float:
    k = param_resolver.get(None, "tilt_modulus_out")
    if k is None:
        k = param_resolver.get(None, "tilt_modolus_out")
    return float(k or 0.0)


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver
) -> Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return tilt energy and gradients for the outer leaflet."""
    k_tilt = _resolve_tilt_modulus(param_resolver)
    shape_grad = {i: np.zeros(3, dtype=float) for i in mesh.vertices}
    tilt_grad = {i: np.zeros(3, dtype=float) for i in mesh.vertices}
    if k_tilt == 0.0:
        return 0.0, shape_grad, tilt_grad

    mesh.build_position_cache()
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0, shape_grad, tilt_grad

    tilts_out = mesh.tilts_out_view()
    tilt_sq = np.einsum("ij,ij->i", tilts_out, tilts_out)

    tri_pos = positions[tri_rows]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]

    n = _fast_cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm >= 1e-12
    if not np.any(mask):
        return 0.0, shape_grad, tilt_grad

    areas = 0.5 * n_norm[mask]
    tri_tilt_sq_sum = tilt_sq[tri_rows[mask]].sum(axis=1)
    coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)

    energy = float(np.dot(coeff, areas))

    n_hat = n[mask] / n_norm[mask][:, None]
    g0 = 0.5 * _fast_cross(n_hat, (v2[mask] - v1[mask]))
    g1 = 0.5 * _fast_cross(n_hat, (v0[mask] - v2[mask]))
    g2 = 0.5 * _fast_cross(n_hat, (v1[mask] - v0[mask]))

    grad_arr = np.zeros_like(positions)
    c = coeff[:, None]
    np.add.at(grad_arr, tri_rows[mask, 0], c * g0)
    np.add.at(grad_arr, tri_rows[mask, 1], c * g1)
    np.add.at(grad_arr, tri_rows[mask, 2], c * g2)

    vertex_areas = np.zeros(len(mesh.vertex_ids), dtype=float)
    area_thirds = areas / 3.0
    np.add.at(vertex_areas, tri_rows[mask, 0], area_thirds)
    np.add.at(vertex_areas, tri_rows[mask, 1], area_thirds)
    np.add.at(vertex_areas, tri_rows[mask, 2], area_thirds)

    tilt_grad_arr = k_tilt * tilts_out * vertex_areas[:, None]
    for row, vid in enumerate(mesh.vertex_ids):
        vidx = int(vid)
        shape_grad[vidx] = grad_arr[row]
        tilt_grad[vidx] = tilt_grad_arr[row]

    return energy, shape_grad, tilt_grad


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray | None,
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
    tilt_in_grad_arr: np.ndarray | None = None,
    tilt_out_grad_arr: np.ndarray | None = None,
) -> float:
    """Dense-array outer-leaflet tilt energy accumulation."""
    _ = index_map, tilts_in, tilt_in_grad_arr
    k_tilt = _resolve_tilt_modulus(param_resolver)
    if k_tilt == 0.0:
        return 0.0

    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0

    mesh.build_position_cache()
    absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet="out")
    tri_keep = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_mask
    )
    if tri_keep.size and not np.any(tri_keep):
        return 0.0

    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()
    else:
        tilts_out = np.asarray(tilts_out, dtype=float)
        if tilts_out.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")

    tilt_sq = np.einsum("ij,ij->i", tilts_out, tilts_out)

    tri_rows_eff = tri_rows[tri_keep] if tri_keep.size else tri_rows
    tri_pos = positions[tri_rows_eff]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]

    n = _fast_cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm >= 1e-12
    if not np.any(mask):
        return 0.0

    areas = 0.5 * n_norm[mask]
    tri_tilt_sq_sum = tilt_sq[tri_rows_eff[mask]].sum(axis=1)
    coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)

    energy = float(np.dot(coeff, areas))

    if grad_arr is not None:
        n_hat = n[mask] / n_norm[mask][:, None]
        g0 = 0.5 * _fast_cross(n_hat, (v2[mask] - v1[mask]))
        g1 = 0.5 * _fast_cross(n_hat, (v0[mask] - v2[mask]))
        g2 = 0.5 * _fast_cross(n_hat, (v1[mask] - v0[mask]))

        c = coeff[:, None]
        np.add.at(grad_arr, tri_rows_eff[mask, 0], c * g0)
        np.add.at(grad_arr, tri_rows_eff[mask, 1], c * g1)
        np.add.at(grad_arr, tri_rows_eff[mask, 2], c * g2)

    if tilt_out_grad_arr is not None:
        tilt_out_grad_arr = np.asarray(tilt_out_grad_arr, dtype=float)
        if tilt_out_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_out_grad_arr must have shape (N_vertices, 3)")

        vertex_areas = np.zeros(len(mesh.vertex_ids), dtype=float)
        area_thirds = areas / 3.0
        np.add.at(vertex_areas, tri_rows_eff[mask, 0], area_thirds)
        np.add.at(vertex_areas, tri_rows_eff[mask, 1], area_thirds)
        np.add.at(vertex_areas, tri_rows_eff[mask, 2], area_thirds)

        tilt_out_grad_arr += k_tilt * tilts_out * vertex_areas[:, None]

    return energy


def compute_energy_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
) -> float:
    """Dense-array outer-leaflet tilt energy (energy only)."""
    _ = index_map, tilts_in
    k_tilt = _resolve_tilt_modulus(param_resolver)
    if k_tilt == 0.0:
        return 0.0

    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0

    mesh.build_position_cache()
    absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet="out")
    tri_keep = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_mask
    )
    if tri_keep.size and not np.any(tri_keep):
        return 0.0

    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()
    else:
        tilts_out = np.asarray(tilts_out, dtype=float)
        if tilts_out.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")

    tilt_sq = np.einsum("ij,ij->i", tilts_out, tilts_out)

    tri_rows_eff = tri_rows[tri_keep] if tri_keep.size else tri_rows
    tri_pos = positions[tri_rows_eff]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]

    n = _fast_cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm >= 1e-12
    if not np.any(mask):
        return 0.0

    areas = 0.5 * n_norm[mask]
    tri_tilt_sq_sum = tilt_sq[tri_rows_eff[mask]].sum(axis=1)
    coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)

    return float(np.dot(coeff, areas))


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]
