"""Local-shell rim matching constraint for curved free-z interfaces.

This module enforces the same radial tilt conditions as `rim_slope_match_out`,
but derives the matching rings from the local shell family immediately outside
the disk boundary instead of relying on coarse tagged rim/outer groups.
"""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh
from modules.constraints.local_interface_shells import (
    build_local_interface_shell_data,
    radial_unit_vectors,
)


def _resolve_theta_scalar(global_params) -> float | None:
    if global_params is None:
        return None
    theta_param = global_params.get("rim_slope_match_thetaB_param")
    if theta_param is not None:
        val = global_params.get(str(theta_param))
        if val is not None:
            return float(val)
    val = global_params.get("tilt_thetaB_value")
    if val is None:
        return None
    return float(val)


def _build_matching_data(mesh: Mesh, global_params, positions: np.ndarray):
    try:
        shell_data = build_local_interface_shell_data(mesh, positions=positions)
    except AssertionError:
        return None

    rim_rows = shell_data.rim_rows_matched
    outer_rows = shell_data.outer_rows
    disk_rows_matched = shell_data.disk_rows_matched
    _, r_hat = radial_unit_vectors(positions[rim_rows])
    _, disk_r_hat = radial_unit_vectors(positions[disk_rows_matched])
    dr = np.maximum(
        np.linalg.norm(positions[outer_rows, :2], axis=1)
        - np.linalg.norm(positions[rim_rows, :2], axis=1),
        1.0e-6,
    )
    phi = (positions[outer_rows, 2] - positions[rim_rows, 2]) / dr
    valid = np.isfinite(phi)
    return {
        "disk_rows_matched": disk_rows_matched,
        "rim_rows": rim_rows,
        "outer_rows": outer_rows,
        "r_hat": r_hat,
        "disk_r_hat": disk_r_hat,
        "phi": phi,
        "valid": valid,
        "theta_scalar": _resolve_theta_scalar(global_params),
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
    _ = index_map, tilts_in, tilts_out
    data = _build_matching_data(mesh, global_params, positions)
    if data is None:
        return None
    normals = mesh.vertex_normals(positions=positions)
    constraints = []
    theta_scalar = data["theta_scalar"]
    for i, ok in enumerate(data["valid"]):
        if not ok:
            continue
        row = int(data["rim_rows"][i])
        n = normals[row]
        r_dir = data["r_hat"][i] - float(np.dot(data["r_hat"][i], n)) * n
        nrm = float(np.linalg.norm(r_dir))
        if nrm < 1.0e-12:
            continue
        r_dir = r_dir / nrm
        out_part = (
            np.asarray([row], dtype=int),
            np.asarray([r_dir], dtype=float),
        )
        in_rows = np.asarray([row], dtype=int)
        in_vecs = np.asarray([r_dir], dtype=float)
        if theta_scalar is None:
            in_rows = np.asarray([row, int(data["disk_rows_matched"][i])], dtype=int)
            in_vecs = np.asarray([r_dir, -data["disk_r_hat"][i]], dtype=float)
        constraints.append((None, out_part))
        constraints.append(((in_rows, in_vecs), None))
    return constraints or None


def enforce_tilt_constraint(mesh: Mesh, global_params=None, **_kwargs) -> None:
    positions = mesh.positions_view()
    data = _build_matching_data(mesh, global_params, positions)
    if data is None:
        return
    normals = mesh.vertex_normals(positions=positions)
    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")
    theta_scalar = data["theta_scalar"]
    if theta_scalar is None:
        theta_disk = np.einsum(
            "ij,ij->i", tilts_in[data["disk_rows_matched"]], data["disk_r_hat"]
        )
    else:
        theta_disk = theta_scalar

    for i, ok in enumerate(data["valid"]):
        if not ok:
            continue
        row = int(data["rim_rows"][i])
        vid = int(mesh.vertex_ids[row])
        n = normals[row]
        r_dir = data["r_hat"][i] - float(np.dot(data["r_hat"][i], n)) * n
        nrm = float(np.linalg.norm(r_dir))
        if nrm < 1.0e-12:
            continue
        r_dir = r_dir / nrm
        if not bool(getattr(mesh.vertices[vid], "tilt_fixed_out", False)):
            t_out = tilts_out[row]
            tilts_out[row] = (
                t_out + (float(data["phi"][i]) - float(np.dot(t_out, r_dir))) * r_dir
            )
        target_in = (
            float(theta_disk[i] - data["phi"][i])
            if isinstance(theta_disk, np.ndarray)
            else float(theta_disk - data["phi"][i])
        )
        if not bool(getattr(mesh.vertices[vid], "tilt_fixed_in", False)):
            t_in = tilts_in[row]
            tilts_in[row] = t_in + (target_in - float(np.dot(t_in, r_dir))) * r_dir

    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)


__all__ = [
    "_build_matching_data",
    "constraint_gradients_tilt_rows_array",
    "enforce_tilt_constraint",
]
