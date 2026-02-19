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


def _resolve_tilt_mass_mode(param_resolver) -> str:
    mode = param_resolver.get(None, "tilt_mass_mode_in")
    if mode is None:
        mode = param_resolver.get(None, "tilt_mass_mode")
    txt = str(mode or "lumped").strip().lower()
    if txt not in {"lumped", "consistent"}:
        raise ValueError("tilt_mass_mode_in must be 'lumped' or 'consistent'.")
    return txt


def _triangle_geometry(
    positions: np.ndarray,
    tri_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-triangle geometric arrays for area and area gradient."""
    tri_pos = positions[tri_rows]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]

    n = _fast_cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm >= 1e-12
    return v0, v1, v2, n, n_norm, mask


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
    mode = _resolve_tilt_mass_mode(param_resolver)
    v0, v1, v2, n, n_norm, mask = _triangle_geometry(positions, tri_rows)
    if not np.any(mask):
        return 0.0, shape_grad, tilt_grad

    areas = 0.5 * n_norm[mask]
    tri_rows_m = tri_rows[mask]
    t0 = tilts_in[tri_rows_m[:, 0]]
    t1 = tilts_in[tri_rows_m[:, 1]]
    t2 = tilts_in[tri_rows_m[:, 2]]

    if mode == "lumped":
        tri_tilt_sq_sum = (
            np.einsum("ij,ij->i", t0, t0)
            + np.einsum("ij,ij->i", t1, t1)
            + np.einsum("ij,ij->i", t2, t2)
        )
        coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)
        energy = float(np.dot(coeff, areas))

        if tilt_grad:
            vertex_areas = np.zeros(len(mesh.vertex_ids), dtype=float)
            area_thirds = areas / 3.0
            np.add.at(vertex_areas, tri_rows_m[:, 0], area_thirds)
            np.add.at(vertex_areas, tri_rows_m[:, 1], area_thirds)
            np.add.at(vertex_areas, tri_rows_m[:, 2], area_thirds)
            tilt_grad_arr = k_tilt * tilts_in * vertex_areas[:, None]
        else:
            tilt_grad_arr = np.zeros_like(positions)
    else:
        s = (
            np.einsum("ij,ij->i", t0, t0)
            + np.einsum("ij,ij->i", t1, t1)
            + np.einsum("ij,ij->i", t2, t2)
            + np.einsum("ij,ij->i", t0, t1)
            + np.einsum("ij,ij->i", t1, t2)
            + np.einsum("ij,ij->i", t2, t0)
        )
        coeff = (k_tilt / 12.0) * s
        energy = float(np.dot(coeff, areas))

        tilt_grad_arr = np.zeros_like(positions)
        tri_factor = (k_tilt * areas / 12.0)[:, None]
        np.add.at(tilt_grad_arr, tri_rows_m[:, 0], tri_factor * ((2.0 * t0) + t1 + t2))
        np.add.at(tilt_grad_arr, tri_rows_m[:, 1], tri_factor * ((2.0 * t1) + t2 + t0))
        np.add.at(tilt_grad_arr, tri_rows_m[:, 2], tri_factor * ((2.0 * t2) + t0 + t1))

    n_hat = n[mask] / n_norm[mask][:, None]
    g0 = 0.5 * _fast_cross(n_hat, (v2[mask] - v1[mask]))
    g1 = 0.5 * _fast_cross(n_hat, (v0[mask] - v2[mask]))
    g2 = 0.5 * _fast_cross(n_hat, (v1[mask] - v0[mask]))

    grad_arr = np.zeros_like(positions)
    c = coeff[:, None]
    np.add.at(grad_arr, tri_rows_m[:, 0], c * g0)
    np.add.at(grad_arr, tri_rows_m[:, 1], c * g1)
    np.add.at(grad_arr, tri_rows_m[:, 2], c * g2)

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
    mode = _resolve_tilt_mass_mode(param_resolver)

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

    v0, v1, v2, n, n_norm, mask = _triangle_geometry(positions, tri_rows)
    if not np.any(mask):
        return 0.0

    areas = 0.5 * n_norm[mask]
    tri_rows_m = tri_rows[mask]
    t0 = tilts_in[tri_rows_m[:, 0]]
    t1 = tilts_in[tri_rows_m[:, 1]]
    t2 = tilts_in[tri_rows_m[:, 2]]

    if mode == "lumped":
        tri_tilt_sq_sum = (
            np.einsum("ij,ij->i", t0, t0)
            + np.einsum("ij,ij->i", t1, t1)
            + np.einsum("ij,ij->i", t2, t2)
        )
        coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)
        energy = float(np.dot(coeff, areas))
    else:
        s = (
            np.einsum("ij,ij->i", t0, t0)
            + np.einsum("ij,ij->i", t1, t1)
            + np.einsum("ij,ij->i", t2, t2)
            + np.einsum("ij,ij->i", t0, t1)
            + np.einsum("ij,ij->i", t1, t2)
            + np.einsum("ij,ij->i", t2, t0)
        )
        coeff = (k_tilt / 12.0) * s
        energy = float(np.dot(coeff, areas))

    if grad_arr is not None:
        n_hat = n[mask] / n_norm[mask][:, None]
        g0 = 0.5 * _fast_cross(n_hat, (v2[mask] - v1[mask]))
        g1 = 0.5 * _fast_cross(n_hat, (v0[mask] - v2[mask]))
        g2 = 0.5 * _fast_cross(n_hat, (v1[mask] - v0[mask]))

        c = coeff[:, None]
        np.add.at(grad_arr, tri_rows_m[:, 0], c * g0)
        np.add.at(grad_arr, tri_rows_m[:, 1], c * g1)
        np.add.at(grad_arr, tri_rows_m[:, 2], c * g2)

    if tilt_in_grad_arr is not None:
        tilt_in_grad_arr = np.asarray(tilt_in_grad_arr, dtype=float)
        if tilt_in_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_in_grad_arr must have shape (N_vertices, 3)")

        if mode == "lumped":
            vertex_areas = mesh.barycentric_vertex_areas(
                positions,
                tri_rows=tri_rows,
                areas=areas,
                mask=mask,
                cache=True,
            )
            tilt_in_grad_arr += k_tilt * tilts_in * vertex_areas[:, None]
        else:
            tri_factor = (k_tilt * areas / 12.0)[:, None]
            np.add.at(
                tilt_in_grad_arr,
                tri_rows_m[:, 0],
                tri_factor * ((2.0 * t0) + t1 + t2),
            )
            np.add.at(
                tilt_in_grad_arr,
                tri_rows_m[:, 1],
                tri_factor * ((2.0 * t1) + t2 + t0),
            )
            np.add.at(
                tilt_in_grad_arr,
                tri_rows_m[:, 2],
                tri_factor * ((2.0 * t2) + t0 + t1),
            )

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
    """Dense-array inner-leaflet tilt energy (energy only)."""
    _ = global_params, index_map, tilts_out
    k_tilt = _resolve_tilt_modulus(param_resolver)
    if k_tilt == 0.0:
        return 0.0
    mode = _resolve_tilt_mass_mode(param_resolver)

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

    v0, v1, v2, n, n_norm, mask = _triangle_geometry(positions, tri_rows)
    if not np.any(mask):
        return 0.0

    areas = 0.5 * n_norm[mask]
    tri_rows_m = tri_rows[mask]
    t0 = tilts_in[tri_rows_m[:, 0]]
    t1 = tilts_in[tri_rows_m[:, 1]]
    t2 = tilts_in[tri_rows_m[:, 2]]

    if mode == "lumped":
        tri_tilt_sq_sum = (
            np.einsum("ij,ij->i", t0, t0)
            + np.einsum("ij,ij->i", t1, t1)
            + np.einsum("ij,ij->i", t2, t2)
        )
        coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)
    else:
        s = (
            np.einsum("ij,ij->i", t0, t0)
            + np.einsum("ij,ij->i", t1, t1)
            + np.einsum("ij,ij->i", t2, t2)
            + np.einsum("ij,ij->i", t0, t1)
            + np.einsum("ij,ij->i", t1, t2)
            + np.einsum("ij,ij->i", t2, t0)
        )
        coeff = (k_tilt / 12.0) * s

    return float(np.dot(coeff, areas))


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]
