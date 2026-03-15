"""Tilt-only local interface vector matching for curved free-z disk boundaries."""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh
from modules.constraints.local_interface_shells import (
    build_local_interface_shell_data,
    radial_unit_vectors,
)


def _orthonormal_basis(
    normal: np.ndarray, preferred_u: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build a tangent basis from a normal and preferred in-plane direction."""
    u = preferred_u - float(np.dot(preferred_u, normal)) * normal
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1.0e-12:
        trial = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(trial, normal))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0], dtype=float)
        u = trial - float(np.dot(trial, normal)) * normal
        u_norm = float(np.linalg.norm(u))
        if u_norm < 1.0e-12:
            u = np.array([1.0, 0.0, 0.0], dtype=float)
            u_norm = 1.0
    u = u / u_norm
    v = np.cross(normal, u)
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1.0e-12:
        v = np.array([0.0, 1.0, 0.0], dtype=float)
        v_norm = 1.0
    return u, v / v_norm


def _build_local_interface_data(mesh: Mesh, global_params, positions: np.ndarray):
    """Build pairwise local-shell data and tangent bases for vector matching."""
    _ = global_params
    try:
        shell_data = build_local_interface_shell_data(mesh, positions=positions)
    except AssertionError:
        return None

    disk_rows = np.asarray(shell_data.disk_rows_matched, dtype=int)
    rim_rows = np.asarray(shell_data.rim_rows, dtype=int)
    outer_rows = np.asarray(shell_data.outer_rows, dtype=int)
    normals = mesh.vertex_normals(positions=positions)
    pair_normals = normals[disk_rows] + normals[rim_rows]
    pair_norms = np.linalg.norm(pair_normals, axis=1)
    bad = pair_norms < 1.0e-12
    if np.any(bad):
        pair_normals[bad] = normals[rim_rows[bad]]
        pair_norms = np.linalg.norm(pair_normals, axis=1)
    pair_normals = pair_normals / np.maximum(pair_norms[:, None], 1.0e-12)

    basis_u = np.zeros_like(pair_normals)
    basis_v = np.zeros_like(pair_normals)
    _, rim_r_hat = radial_unit_vectors(positions[rim_rows])
    for idx, normal in enumerate(pair_normals):
        u_vec, v_vec = _orthonormal_basis(normal, rim_r_hat[idx])
        basis_u[idx] = u_vec
        basis_v[idx] = v_vec

    return {
        "disk_rows": disk_rows,
        "rim_rows": rim_rows,
        "outer_rows": outer_rows,
        "basis_u": basis_u,
        "basis_v": basis_v,
        "pair_normals": pair_normals,
        "disk_radius": float(shell_data.disk_radius),
        "rim_radius": float(shell_data.rim_radius),
        "outer_radius": float(shell_data.outer_radius),
        "matching_strategy": str(shell_data.matching_strategy),
        "shell_source": str(shell_data.shell_source),
        "projection_mode": "vector_average",
    }


def _build_leaflet_constraints(
    rows_a: np.ndarray,
    rows_b: np.ndarray,
    basis: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build sparse-row equality constraints in one tangent basis direction."""
    count = int(rows_a.size)
    rows = np.empty(2 * count, dtype=int)
    rows[:count] = rows_a
    rows[count:] = rows_b
    vecs = np.empty((2 * count, 3), dtype=float)
    vecs[:count] = basis
    vecs[count:] = -basis
    return [(rows, vecs)]


def constraint_gradients_tilt_rows_array(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
):
    """Return sparse-row in-plane vector-matching constraints for both leaflets."""
    _ = index_map, tilts_in, tilts_out
    data = _build_local_interface_data(mesh, global_params, positions)
    if data is None:
        return None

    constraints = []
    for basis in (data["basis_u"], data["basis_v"]):
        rows, vecs = _build_leaflet_constraints(
            data["rim_rows"],
            data["disk_rows"],
            basis,
        )[0]
        constraints.append(((rows, vecs), None))
        constraints.append((None, (rows, vecs.copy())))
    return constraints or None


def constraint_gradients_tilt_array(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
) -> list[tuple[np.ndarray | None, np.ndarray | None]] | None:
    """Return dense tilt constraint gradients for local in-plane vector matching."""
    _ = index_map, tilts_in, tilts_out
    row_constraints = constraint_gradients_tilt_rows_array(
        mesh,
        global_params,
        positions=positions,
        index_map={},
    )
    if not row_constraints:
        return None
    constraints: list[tuple[np.ndarray | None, np.ndarray | None]] = []
    for in_part, out_part in row_constraints:
        g_in = None
        g_out = None
        if in_part is not None:
            in_rows, in_vecs = in_part
            g_in = np.zeros_like(positions)
            np.add.at(g_in, in_rows, in_vecs)
        if out_part is not None:
            out_rows, out_vecs = out_part
            g_out = np.zeros_like(positions)
            np.add.at(g_out, out_rows, out_vecs)
        constraints.append((g_in, g_out))
    return constraints or None


__all__ = [
    "_build_local_interface_data",
    "constraint_gradients_tilt_array",
    "constraint_gradients_tilt_rows_array",
]
