"""Inter-leaflet tilt coupling energy module.

This module penalizes mismatch (or anti-alignment) between ``tilt_in`` and
``tilt_out``:

    E = 1/2 * k_c * ∫ |t_out ± t_in|^2 dA

The sign is controlled by ``tilt_coupling_mode``:
    - "difference": uses |t_out - t_in|^2 (tracking)
    - "sum": uses |t_out + t_in|^2 (anti-tracking)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross

USES_TILT_LEAFLETS = True


def _resolve_coupling_mode(param_resolver) -> int | None:
    mode = param_resolver.get(None, "tilt_coupling_mode")
    if mode is None:
        mode = param_resolver.get(None, "tilt_couping_mode")
    if mode is None:
        return None
    mode_norm = str(mode).strip().lower()
    if mode_norm in ("difference", "diff", "minus", "sub"):
        return -1
    if mode_norm in ("sum", "add", "plus"):
        return 1
    return None


def _resolve_coupling_modulus(param_resolver) -> float:
    k = param_resolver.get(None, "tilt_coupling_modulus")
    return float(k or 0.0)


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver
) -> Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return inter-leaflet coupling energy and gradients."""
    sign = _resolve_coupling_mode(param_resolver)
    k_c = _resolve_coupling_modulus(param_resolver)
    shape_grad = {i: np.zeros(3, dtype=float) for i in mesh.vertices}
    tilt_grad = {i: np.zeros(3, dtype=float) for i in mesh.vertices}
    if sign is None or k_c == 0.0:
        return 0.0, shape_grad, tilt_grad

    mesh.build_position_cache()
    positions = np.empty((len(mesh.vertex_ids), 3), dtype=float)
    for row, vid in enumerate(mesh.vertex_ids):
        positions[row] = mesh.vertices[int(vid)].position
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0, shape_grad, tilt_grad

    diff_sq = np.zeros(len(mesh.vertex_ids), dtype=float)
    for row, vid in enumerate(mesh.vertex_ids):
        t_in = np.asarray(mesh.vertices[int(vid)].tilt_in, dtype=float)
        t_out = np.asarray(mesh.vertices[int(vid)].tilt_out, dtype=float)
        diff = t_out + sign * t_in
        diff_sq[row] = float(np.dot(diff, diff))

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
    tri_diff_sq_sum = diff_sq[tri_rows[mask]].sum(axis=1)
    coeff = 0.5 * k_c * (tri_diff_sq_sum / 3.0)

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

    for row, vid in enumerate(mesh.vertex_ids):
        vidx = int(vid)
        shape_grad[vidx] = grad_arr[row]
        t_in = np.asarray(mesh.vertices[vidx].tilt_in, dtype=float)
        t_out = np.asarray(mesh.vertices[vidx].tilt_out, dtype=float)
        diff = t_out + sign * t_in
        tilt_grad[vidx] = k_c * diff * vertex_areas[row]

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
    """Dense-array inter-leaflet coupling energy accumulation."""
    _ = global_params, index_map
    sign = _resolve_coupling_mode(param_resolver)
    k_c = _resolve_coupling_modulus(param_resolver)
    if sign is None or k_c == 0.0:
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

    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()
    else:
        tilts_out = np.asarray(tilts_out, dtype=float)
        if tilts_out.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")

    diff = tilts_out + sign * tilts_in
    diff_sq = np.einsum("ij,ij->i", diff, diff)

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
    tri_diff_sq_sum = diff_sq[tri_rows[mask]].sum(axis=1)
    coeff = 0.5 * k_c * (tri_diff_sq_sum / 3.0)

    energy = float(np.dot(coeff, areas))

    n_hat = n[mask] / n_norm[mask][:, None]
    g0 = 0.5 * _fast_cross(n_hat, (v2[mask] - v1[mask]))
    g1 = 0.5 * _fast_cross(n_hat, (v0[mask] - v2[mask]))
    g2 = 0.5 * _fast_cross(n_hat, (v1[mask] - v0[mask]))

    c = coeff[:, None]
    if grad_arr is not None:
        np.add.at(grad_arr, tri_rows[mask, 0], c * g0)
        np.add.at(grad_arr, tri_rows[mask, 1], c * g1)
        np.add.at(grad_arr, tri_rows[mask, 2], c * g2)

    if tilt_in_grad_arr is not None or tilt_out_grad_arr is not None:
        vertex_areas = mesh.barycentric_vertex_areas(
            positions,
            tri_rows=tri_rows,
            areas=areas,
            mask=mask,
            cache=True,
        )

        if tilt_out_grad_arr is not None:
            tilt_out_grad_arr = np.asarray(tilt_out_grad_arr, dtype=float)
            if tilt_out_grad_arr.shape != (len(mesh.vertex_ids), 3):
                raise ValueError("tilt_out_grad_arr must have shape (N_vertices, 3)")
            tilt_out_grad_arr += k_c * diff * vertex_areas[:, None]

        if tilt_in_grad_arr is not None:
            tilt_in_grad_arr = np.asarray(tilt_in_grad_arr, dtype=float)
            if tilt_in_grad_arr.shape != (len(mesh.vertex_ids), 3):
                raise ValueError("tilt_in_grad_arr must have shape (N_vertices, 3)")
            tilt_in_grad_arr += k_c * sign * diff * vertex_areas[:, None]

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
    """Dense-array inter-leaflet coupling energy (energy only)."""
    _ = global_params, index_map
    sign = _resolve_coupling_mode(param_resolver)
    k_c = _resolve_coupling_modulus(param_resolver)
    if sign is None or k_c == 0.0:
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

    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()
    else:
        tilts_out = np.asarray(tilts_out, dtype=float)
        if tilts_out.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")

    diff = tilts_out + sign * tilts_in
    diff_sq = np.einsum("ij,ij->i", diff, diff)

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
    tri_diff_sq_sum = diff_sq[tri_rows[mask]].sum(axis=1)
    coeff = 0.5 * k_c * (tri_diff_sq_sum / 3.0)

    return float(np.dot(coeff, areas))


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]
