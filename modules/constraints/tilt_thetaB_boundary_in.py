"""Hard thetaB boundary condition for the inner-leaflet tilt field.

This constraint enforces the Kozlov-style boundary condition at the disk
interface:

    t_in · r_hat = thetaB

where thetaB is a *scalar* global DOF stored in
``global_params['tilt_thetaB_value']``.

Unlike the legacy penalty approach (which adds a large quadratic term to the
energy), this module performs a tilt-only projection and therefore does *not*
contribute to the reported energy breakdown.

Vertex selection
----------------
The boundary ring is selected by a group name. The group is resolved from:

  1) ``global_params['tilt_thetaB_group_in']`` (preferred), else
  2) ``global_params['rim_slope_match_disk_group']`` (fallback).

Vertices are included when any of the following vertex options match the group:

  - ``rim_slope_match_group``
  - ``tilt_thetaB_group``
  - ``tilt_thetaB_group_in`` (legacy tagging convenience)

Notes
-----
The projection respects ``vertex.tilt_fixed_in`` and leaves all other tilt
components unchanged (only the radial component is adjusted).
"""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh


def _normalize(vec: np.ndarray) -> np.ndarray | None:
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-15:
        return None
    return vec / nrm


def _fit_plane_normal(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    centroid = np.mean(points, axis=0)
    X = points - centroid
    try:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    return _normalize(vh[-1, :])


def _resolve_group(global_params) -> str | None:
    raw = None if global_params is None else global_params.get("tilt_thetaB_group_in")
    if raw is None and global_params is not None:
        raw = global_params.get("rim_slope_match_disk_group")
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _resolve_center(global_params) -> np.ndarray:
    center = None if global_params is None else global_params.get("tilt_thetaB_center")
    if center is None:
        center = [0.0, 0.0, 0.0]
    return np.asarray(center, dtype=float).reshape(3)


def _resolve_normal(global_params, points: np.ndarray) -> np.ndarray:
    raw = None if global_params is None else global_params.get("tilt_thetaB_normal")
    if raw is not None:
        n = _normalize(np.asarray(raw, dtype=float).reshape(3))
        if n is not None:
            return n
    fitted = _fit_plane_normal(points)
    if fitted is not None:
        return fitted
    return np.array([0.0, 0.0, 1.0], dtype=float)


def _collect_group_rows(mesh: Mesh, group: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if (
            opts.get("rim_slope_match_group") == group
            or opts.get("tilt_thetaB_group") == group
            or opts.get("tilt_thetaB_group_in") == group
        ):
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    return np.asarray(rows, dtype=int)


def enforce_tilt_constraint(mesh: Mesh, global_params=None, **_kwargs) -> None:
    """Project inner-leaflet radial tilt on the disk boundary to thetaB."""
    group = _resolve_group(global_params)
    if group is None:
        return

    thetaB = (
        0.0
        if global_params is None
        else float(global_params.get("tilt_thetaB_value") or 0.0)
    )

    positions = mesh.positions_view()
    rows = _collect_group_rows(mesh, group)
    if rows.size == 0:
        return

    pts = positions[rows]
    center = _resolve_center(global_params)
    normal = _resolve_normal(global_params, pts)

    # In-plane radial vectors from center to the boundary ring.
    r_vec = pts - center[None, :]
    r_vec = r_vec - np.einsum("ij,j->i", r_vec, normal)[:, None] * normal[None, :]
    r_len = np.linalg.norm(r_vec, axis=1)
    good = r_len > 1e-12
    if not np.any(good):
        return

    r_hat = np.zeros_like(r_vec)
    r_hat[good] = r_vec[good] / r_len[good][:, None]

    tilts_in = mesh.tilts_in_view().copy(order="F")
    normals_v = mesh.vertex_normals(positions=positions)

    for i, ok in enumerate(good):
        if not ok:
            continue
        row = int(rows[i])
        vid = int(mesh.vertex_ids[row])
        if getattr(mesh.vertices[vid], "tilt_fixed_in", False):
            continue

        # Use the radial direction projected into the local tangent plane.
        n = normals_v[row]
        r_dir = r_hat[i] - float(np.dot(r_hat[i], n)) * n
        r_norm = float(np.linalg.norm(r_dir))
        if r_norm < 1e-12:
            continue
        r_dir = r_dir / r_norm

        t = tilts_in[row]
        t_rad = float(np.dot(t, r_dir))
        tilts_in[row] = t + (thetaB - t_rad) * r_dir

    mesh.set_tilts_in_from_array(tilts_in)


__all__ = ["enforce_tilt_constraint"]
