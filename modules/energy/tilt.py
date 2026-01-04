"""Tilt energy module.

This module currently models a simple per-vertex tilt penalty:

    E = 1/2 * k_t * sum_v (|t_v|^2 * A_v)

where ``t_v`` is a 3D tangent tilt vector stored on each vertex and ``A_v`` is
an area weight.

The current discretization uses a barycentric area weight:

    A_v = (1/3) * sum_{tri incident to v} area(tri)

This keeps the pure tilt term well-defined and provides a shape gradient
through the triangle area derivatives, even when the tilt vectors are held
fixed during minimization.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver
) -> Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return tilt energy and gradients.

    Returns (E, shape_grad, tilt_grad).
    """
    k_tilt = float(param_resolver.get(None, "tilt_rigidity") or 0.0)
    shape_grad = {i: np.zeros(3, dtype=float) for i in mesh.vertices}
    tilt_grad = {i: np.zeros(3, dtype=float) for i in mesh.vertices}
    if k_tilt == 0.0:
        return 0.0, shape_grad, tilt_grad

    mesh.build_position_cache()
    positions = np.empty((len(mesh.vertex_ids), 3), dtype=float)
    for row, vid in enumerate(mesh.vertex_ids):
        positions[row] = mesh.vertices[int(vid)].position
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0, shape_grad, tilt_grad

    tilt_sq = np.zeros(len(mesh.vertex_ids), dtype=float)
    for row, vid in enumerate(mesh.vertex_ids):
        tilt_vec = np.asarray(mesh.vertices[int(vid)].tilt, dtype=float)
        tilt_sq[row] = float(np.dot(tilt_vec, tilt_vec))

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
    coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)  # dE/dA per triangle

    energy = float(np.dot(coeff, areas))

    n_hat = n[mask] / n_norm[mask][:, None]
    # Area gradient per vertex for the v0,v1,v2 ordering (matches Facet.compute_area_gradient):
    # dA/dv0 = 0.5 * n_hat x (v2 - v1), and cyclic permutations.
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

    for row, vid in enumerate(mesh.vertex_ids):
        vidx = int(vid)
        shape_grad[vidx] = grad_arr[row]
        tilt_vec = np.asarray(mesh.vertices[vidx].tilt, dtype=float)
        tilt_grad[vidx] = k_tilt * tilt_vec * vertex_areas[row]

    return energy, shape_grad, tilt_grad


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray,
    tilts: np.ndarray | None = None,
    tilt_grad_arr: np.ndarray | None = None,
) -> float:
    """Dense-array tilt energy accumulation (shape gradient; optional tilt gradient)."""
    k_tilt = float(param_resolver.get(None, "tilt_rigidity") or 0.0)
    if k_tilt == 0.0:
        return 0.0

    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0

    mesh.build_position_cache()
    if tilts is None:
        tilts = mesh.tilts_view()
    else:
        tilts = np.asarray(tilts, dtype=float)
        if tilts.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts must have shape (N_vertices, 3)")

    tilt_sq = np.einsum("ij,ij->i", tilts, tilts)

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
    coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)  # dE/dA per triangle

    energy = float(np.dot(coeff, areas))

    n_hat = n[mask] / n_norm[mask][:, None]
    # Area gradient per vertex for the v0,v1,v2 ordering (matches Facet.compute_area_gradient).
    g0 = 0.5 * _fast_cross(n_hat, (v2[mask] - v1[mask]))
    g1 = 0.5 * _fast_cross(n_hat, (v0[mask] - v2[mask]))
    g2 = 0.5 * _fast_cross(n_hat, (v1[mask] - v0[mask]))

    c = coeff[:, None]
    np.add.at(grad_arr, tri_rows[mask, 0], c * g0)
    np.add.at(grad_arr, tri_rows[mask, 1], c * g1)
    np.add.at(grad_arr, tri_rows[mask, 2], c * g2)

    if tilt_grad_arr is not None:
        tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
        if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_grad_arr must have shape (N_vertices, 3)")

        vertex_areas = np.zeros(len(mesh.vertex_ids), dtype=float)
        area_thirds = areas / 3.0
        np.add.at(vertex_areas, tri_rows[mask, 0], area_thirds)
        np.add.at(vertex_areas, tri_rows[mask, 1], area_thirds)
        np.add.at(vertex_areas, tri_rows[mask, 2], area_thirds)

        tilt_grad_arr += k_tilt * tilts * vertex_areas[:, None]

    return energy


__all__ = ["compute_energy_and_gradient", "compute_energy_and_gradient_array"]
