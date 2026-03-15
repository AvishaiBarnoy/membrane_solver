"""Low-dimensional hard constraint on the true local shell family near r=R."""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh
from modules.constraints.local_interface_shells import build_local_interface_shell_data


def _build_matching_data(mesh: Mesh, positions: np.ndarray):
    """Build ring-averaged outer-radial matching data on the local shell family."""
    try:
        shell_data = build_local_interface_shell_data(mesh, positions=positions)
    except AssertionError:
        return None

    rim_rows = np.asarray(shell_data.rim_rows_matched, dtype=int)
    outer_rows = np.asarray(shell_data.outer_rows, dtype=int)
    if rim_rows.size == 0 or outer_rows.size == 0:
        return None

    normals = mesh.vertex_normals(positions=positions)
    r_hat = np.asarray(shell_data.rim_r_hat, dtype=float)
    n_rim = normals[rim_rows]
    r_dir = r_hat - np.einsum("ij,ij->i", r_hat, n_rim)[:, None] * n_rim
    r_norm = np.linalg.norm(r_dir, axis=1)
    valid = r_norm > 1.0e-12
    if not np.any(valid):
        return None
    r_dir[valid] = r_dir[valid] / r_norm[valid, None]

    radii = np.linalg.norm(positions[:, :2], axis=1)
    dr = radii[outer_rows] - radii[rim_rows]
    valid &= np.abs(dr) > 1.0e-12
    if not np.any(valid):
        return None

    phi = np.zeros(rim_rows.size, dtype=float)
    phi[valid] = (positions[outer_rows[valid], 2] - positions[rim_rows[valid], 2]) / dr[
        valid
    ]
    return {
        "rim_rows": rim_rows,
        "outer_rows": outer_rows,
        "r_dir": r_dir,
        "phi": phi,
        "valid": valid,
        "rim_radius": float(shell_data.rim_radius),
        "outer_radius": float(shell_data.outer_radius),
        "matching_strategy": str(shell_data.matching_strategy),
        "shell_source": str(shell_data.shell_source),
    }


def constraint_gradients_tilt_rows_array(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
):
    """Return one sparse ring-averaged outer-radial tilt constraint."""
    _ = global_params, index_map, tilts_in, tilts_out
    data = _build_matching_data(mesh, positions)
    if data is None:
        return None
    valid = np.asarray(data["valid"], dtype=bool)
    if not np.any(valid):
        return None
    rows = np.asarray(data["rim_rows"][valid], dtype=int)
    vecs = np.asarray(data["r_dir"][valid], dtype=float) / float(rows.size)
    return [(None, (rows, vecs))]


def constraint_gradients_tilt_array(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
):
    """Return one dense ring-averaged outer-radial tilt constraint."""
    _ = global_params, index_map, tilts_in, tilts_out
    row_constraints = constraint_gradients_tilt_rows_array(
        mesh,
        global_params,
        positions=positions,
        index_map={},
    )
    if not row_constraints:
        return None
    _, out_part = row_constraints[0]
    assert out_part is not None
    out_rows, out_vecs = out_part
    g_out = np.zeros_like(positions)
    np.add.at(g_out, out_rows, out_vecs)
    return [(None, g_out)]


def enforce_tilt_constraint(mesh: Mesh, global_params=None, **_kwargs) -> None:
    """Project outer-radial tilt so the ring-averaged mismatch vanishes."""
    _ = global_params
    positions = mesh.positions_view()
    data = _build_matching_data(mesh, positions)
    if data is None:
        return

    rim_rows = np.asarray(data["rim_rows"], dtype=int)
    r_dir = np.asarray(data["r_dir"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)
    valid = np.asarray(data["valid"], dtype=bool)

    tilts_out = mesh.tilts_out_view().copy(order="F")
    residuals: list[float] = []
    update_rows: list[int] = []
    update_dirs: list[np.ndarray] = []

    for i, ok in enumerate(valid):
        if not ok:
            continue
        row = int(rim_rows[i])
        vid = int(mesh.vertex_ids[row])
        if bool(getattr(mesh.vertices[vid], "tilt_fixed_out", False)):
            continue
        t_out_rad = float(np.dot(tilts_out[row], r_dir[i]))
        residuals.append(t_out_rad - float(phi[i]))
        update_rows.append(row)
        update_dirs.append(r_dir[i])

    if not residuals:
        return

    mean_residual = float(np.mean(np.asarray(residuals, dtype=float)))
    for row, direction in zip(update_rows, update_dirs):
        tilts_out[row] = tilts_out[row] - mean_residual * direction
    mesh.set_tilts_out_from_array(tilts_out)


__all__ = [
    "_build_matching_data",
    "constraint_gradients_tilt_array",
    "constraint_gradients_tilt_rows_array",
    "enforce_tilt_constraint",
]
