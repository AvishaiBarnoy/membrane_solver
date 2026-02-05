"""Inner-leaflet tilt magnitude energy module.

This module models a per-vertex tilt penalty for the inner leaflet:

    E = 1/2 * k_t * sum_v (|t_in,v|^2 * A_v)

where ``t_in,v`` is a 3D tangent tilt vector stored on each vertex and ``A_v``
is a barycentric area weight.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross

USES_TILT_LEAFLETS = True


def _resolve_tilt_modulus(param_resolver) -> float:
    k = param_resolver.get(None, "tilt_modulus_in")
    if k is None:
        k = param_resolver.get(None, "tilt_modolus_in")
    return float(k or 0.0)


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver
) -> Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return tilt energy and gradients for the inner leaflet."""
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

    tilts_in = mesh.tilts_in_view()
    tilt_sq = np.einsum("ij,ij->i", tilts_in, tilts_in)

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

    tilt_grad_arr = k_tilt * tilts_in * vertex_areas[:, None]
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
    """Dense-array inner-leaflet tilt energy accumulation."""
    _ = global_params, index_map, tilts_out, tilt_out_grad_arr
    k_tilt = _resolve_tilt_modulus(param_resolver)
    if k_tilt == 0.0:
        return 0.0

    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0

    mesh.build_position_cache()
    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

    tilt_sq = np.einsum("ij,ij->i", tilts_in, tilts_in)

    tri_pos = positions[tri_rows]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]

    n = _fast_cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm >= 1e-12
    if not np.any(mask):
        return 0.0

    areas = 0.5 * n_norm[mask]
    tri_tilt_sq_sum = tilt_sq[tri_rows[mask]].sum(axis=1)
    coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)

    energy = float(np.dot(coeff, areas))

    if grad_arr is not None:
        n_hat = n[mask] / n_norm[mask][:, None]
        g0 = 0.5 * _fast_cross(n_hat, (v2[mask] - v1[mask]))
        g1 = 0.5 * _fast_cross(n_hat, (v0[mask] - v2[mask]))
        g2 = 0.5 * _fast_cross(n_hat, (v1[mask] - v0[mask]))

        c = coeff[:, None]
        np.add.at(grad_arr, tri_rows[mask, 0], c * g0)
        np.add.at(grad_arr, tri_rows[mask, 1], c * g1)
        np.add.at(grad_arr, tri_rows[mask, 2], c * g2)

    if tilt_in_grad_arr is not None:
        tilt_in_grad_arr = np.asarray(tilt_in_grad_arr, dtype=float)
        if tilt_in_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_in_grad_arr must have shape (N_vertices, 3)")

        vertex_areas = np.zeros(len(mesh.vertex_ids), dtype=float)
        area_thirds = areas / 3.0
        np.add.at(vertex_areas, tri_rows[mask, 0], area_thirds)
        np.add.at(vertex_areas, tri_rows[mask, 1], area_thirds)
        np.add.at(vertex_areas, tri_rows[mask, 2], area_thirds)

        tilt_in_grad_arr += k_tilt * tilts_in * vertex_areas[:, None]

    return energy


__all__ = ["compute_energy_and_gradient", "compute_energy_and_gradient_array"]
