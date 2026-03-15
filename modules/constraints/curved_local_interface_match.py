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


def _resolve_projection_mode(global_params) -> str:
    raw = (
        None
        if global_params is None
        else global_params.get("curved_local_interface_match_mode")
    )
    mode = str(raw or "vector_average").strip().lower()
    if mode in {"local_mixed_match_v1", "mixed"}:
        return "local_mixed_match_v1"
    return "vector_average"


def _build_local_interface_data(mesh: Mesh, global_params, positions: np.ndarray):
    """Build pairwise local-shell data and tangent bases for vector matching."""
    try:
        shell_data = build_local_interface_shell_data(mesh, positions=positions)
    except AssertionError:
        return None

    mode = _resolve_projection_mode(global_params)
    if mode == "local_mixed_match_v1":
        rim_rows = np.asarray(shell_data.rim_rows_matched, dtype=int)
        phi_rim = np.mod(
            np.arctan2(positions[rim_rows, 1], positions[rim_rows, 0]), 2.0 * np.pi
        )
        phi_disk = np.mod(
            np.arctan2(
                positions[shell_data.disk_rows, 1], positions[shell_data.disk_rows, 0]
            ),
            2.0 * np.pi,
        )
        dphi_disk = np.abs(phi_rim[:, None] - phi_disk[None, :])
        dphi_disk = np.minimum(dphi_disk, 2.0 * np.pi - dphi_disk)
        disk_rows = np.asarray(
            shell_data.disk_rows[np.argmin(dphi_disk, axis=1)], dtype=int
        )
    else:
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

    dr = np.maximum(
        np.linalg.norm(positions[outer_rows, :2], axis=1)
        - np.linalg.norm(positions[shell_data.rim_rows_matched, :2], axis=1),
        1.0e-6,
    )
    phi = (positions[outer_rows, 2] - positions[shell_data.rim_rows_matched, 2]) / dr

    return {
        "disk_rows": disk_rows,
        "rim_rows": rim_rows,
        "outer_rows": outer_rows,
        "basis_u": basis_u,
        "basis_v": basis_v,
        "pair_normals": pair_normals,
        "phi": phi,
        "disk_radius": float(shell_data.disk_radius),
        "rim_radius": float(shell_data.rim_radius),
        "outer_radius": float(shell_data.outer_radius),
        "matching_strategy": str(shell_data.matching_strategy),
        "shell_source": str(shell_data.shell_source),
        "projection_mode": mode,
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
    basis_list = (data["basis_u"], data["basis_v"])
    if str(data["projection_mode"]) == "local_mixed_match_v1":
        basis_list = (data["basis_v"],)
    for basis in basis_list:
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


def _project_pair(
    *,
    disk_vec: np.ndarray,
    rim_vec: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    disk_fixed: bool,
    rim_fixed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one matched pair onto the local in-plane continuity manifold."""
    coeff_disk = np.array(
        [float(np.dot(disk_vec, basis_u)), float(np.dot(disk_vec, basis_v))],
        dtype=float,
    )
    coeff_rim = np.array(
        [float(np.dot(rim_vec, basis_u)), float(np.dot(rim_vec, basis_v))],
        dtype=float,
    )

    if disk_fixed and rim_fixed:
        target = coeff_rim
    elif disk_fixed:
        target = coeff_disk
    elif rim_fixed:
        target = coeff_rim
    else:
        target = 0.5 * (coeff_disk + coeff_rim)

    def _apply(vec: np.ndarray, coeff_now: np.ndarray) -> np.ndarray:
        delta = (target[0] - coeff_now[0]) * basis_u + (
            target[1] - coeff_now[1]
        ) * basis_v
        return vec + delta

    new_disk = disk_vec if disk_fixed else _apply(disk_vec, coeff_disk)
    new_rim = rim_vec if rim_fixed else _apply(rim_vec, coeff_rim)
    return new_disk, new_rim


def _project_pair_mixed(
    *,
    disk_vec: np.ndarray,
    rim_vec: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    phi_target: float,
    disk_fixed: bool,
    rim_fixed: bool,
    leaflet: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one pair with tangential continuity and leaflet-specific radial targets."""
    coeff_disk = np.array(
        [float(np.dot(disk_vec, basis_u)), float(np.dot(disk_vec, basis_v))],
        dtype=float,
    )
    coeff_rim = np.array(
        [float(np.dot(rim_vec, basis_u)), float(np.dot(rim_vec, basis_v))],
        dtype=float,
    )
    tangent_target = 0.5 * (coeff_disk[1] + coeff_rim[1])
    if disk_fixed:
        tangent_target = coeff_disk[1]
    if rim_fixed:
        tangent_target = coeff_rim[1]

    radial_target = float(phi_target if leaflet == "out" else -phi_target)

    def _apply(vec: np.ndarray, coeff_now: np.ndarray) -> np.ndarray:
        delta = (radial_target - coeff_now[0]) * basis_u + (
            tangent_target - coeff_now[1]
        ) * basis_v
        return vec + delta

    new_disk = disk_vec if disk_fixed else _apply(disk_vec, coeff_disk)
    new_rim = rim_vec if rim_fixed else _apply(rim_vec, coeff_rim)
    return new_disk, new_rim


def enforce_tilt_constraint(mesh: Mesh, global_params=None, **_kwargs) -> None:
    """Project both leaflet tilt fields onto local in-plane continuity."""
    positions = mesh.positions_view()
    data = _build_local_interface_data(mesh, global_params, positions)
    if data is None:
        return

    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")
    mode = str(data["projection_mode"])
    for idx, rim_row in enumerate(data["rim_rows"]):
        disk_row = int(data["disk_rows"][idx])
        rim_row = int(rim_row)
        disk_vid = int(mesh.vertex_ids[disk_row])
        rim_vid = int(mesh.vertex_ids[rim_row])
        basis_u = data["basis_u"][idx]
        basis_v = data["basis_v"][idx]
        phi_target = float(data["phi"][idx])

        if mode == "local_mixed_match_v1":
            new_disk_in, new_rim_in = _project_pair_mixed(
                disk_vec=tilts_in[disk_row],
                rim_vec=tilts_in[rim_row],
                basis_u=basis_u,
                basis_v=basis_v,
                phi_target=phi_target,
                disk_fixed=bool(
                    getattr(mesh.vertices[disk_vid], "tilt_fixed_in", False)
                ),
                rim_fixed=bool(getattr(mesh.vertices[rim_vid], "tilt_fixed_in", False)),
                leaflet="in",
            )
            tilts_in[disk_row] = new_disk_in
            tilts_in[rim_row] = new_rim_in

            new_disk_out, new_rim_out = _project_pair_mixed(
                disk_vec=tilts_out[disk_row],
                rim_vec=tilts_out[rim_row],
                basis_u=basis_u,
                basis_v=basis_v,
                phi_target=phi_target,
                disk_fixed=bool(
                    getattr(mesh.vertices[disk_vid], "tilt_fixed_out", False)
                ),
                rim_fixed=bool(
                    getattr(mesh.vertices[rim_vid], "tilt_fixed_out", False)
                ),
                leaflet="out",
            )
            tilts_out[disk_row] = new_disk_out
            tilts_out[rim_row] = new_rim_out
            continue

        new_disk_in, new_rim_in = _project_pair(
            disk_vec=tilts_in[disk_row],
            rim_vec=tilts_in[rim_row],
            basis_u=basis_u,
            basis_v=basis_v,
            disk_fixed=bool(getattr(mesh.vertices[disk_vid], "tilt_fixed_in", False)),
            rim_fixed=bool(getattr(mesh.vertices[rim_vid], "tilt_fixed_in", False)),
        )
        tilts_in[disk_row] = new_disk_in
        tilts_in[rim_row] = new_rim_in

        new_disk_out, new_rim_out = _project_pair(
            disk_vec=tilts_out[disk_row],
            rim_vec=tilts_out[rim_row],
            basis_u=basis_u,
            basis_v=basis_v,
            disk_fixed=bool(getattr(mesh.vertices[disk_vid], "tilt_fixed_out", False)),
            rim_fixed=bool(getattr(mesh.vertices[rim_vid], "tilt_fixed_out", False)),
        )
        tilts_out[disk_row] = new_disk_out
        tilts_out[rim_row] = new_rim_out

    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)


__all__ = [
    "_build_local_interface_data",
    "constraint_gradients_tilt_array",
    "constraint_gradients_tilt_rows_array",
    "enforce_tilt_constraint",
]
