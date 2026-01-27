"""Hard constraint: match inner/outer leaflet in-plane tilt on a rim.

This constraint encodes the tensionless (γ=0) director-continuity condition at
the disk boundary used in `docs/tex/1_disk_3d.pdf`: distal/proximal monolayer
tilts match at r=R. In the dual-leaflet discretization, this is implemented as
an equality constraint on the in-plane components of `tilt_in` and `tilt_out`
for vertices on a tagged rim ring.

Tagging
-------
Tag rim vertices with:
  - ``tilt_leaflet_match_group``: group name (string)

Global parameters
-----------------
- ``tilt_leaflet_match_group``: group name (string; required)
  (used to select tagged vertices; matches the vertex option key)
- ``tilt_leaflet_match_mode``: projection mode, one of:
    * ``average`` (default): set both to the average when both are free
    * ``in_to_out``: set `tilt_out` to `tilt_in`
    * ``out_to_in``: set `tilt_in` to `tilt_out`

Notes
-----
- This module provides only *tilt* constraints (no shape constraints).
- Matching is done in a fitted local tangent basis (u,v) built from the rim’s
  best-fit plane normal, so it generalizes to disks on curved surfaces.
"""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh


def _resolve_group(global_params) -> str | None:
    raw = (
        None if global_params is None else global_params.get("tilt_leaflet_match_group")
    )
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _resolve_mode(global_params) -> str:
    raw = (
        None if global_params is None else global_params.get("tilt_leaflet_match_mode")
    )
    mode = str(raw or "average").strip().lower()
    if mode in {"in_to_out", "in2out"}:
        return "in_to_out"
    if mode in {"out_to_in", "out2in"}:
        return "out_to_in"
    return "average"


def _orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(trial, normal)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    u = trial - np.dot(trial, normal) * normal
    nrm = float(np.linalg.norm(u))
    if nrm < 1e-15:
        u = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        u /= nrm
    v = np.cross(normal, u)
    nrm_v = float(np.linalg.norm(v))
    if nrm_v < 1e-15:
        v = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        v /= nrm_v
    return u, v


def _fit_plane_normal(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    centroid = np.mean(points, axis=0)
    X = points - centroid
    try:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1, :]
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-15:
        return None
    return normal / nrm


def _collect_rows(mesh: Mesh, group: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        v = mesh.vertices[int(vid)]
        opts = getattr(v, "options", None) or {}
        if opts.get("tilt_leaflet_match_group") != group:
            continue
        row = mesh.vertex_index_to_row.get(int(vid))
        if row is not None:
            rows.append(int(row))
    return np.asarray(rows, dtype=int)


def constraint_gradients_tilt_array(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
) -> list[tuple[np.ndarray | None, np.ndarray | None]] | None:
    """Return tilt constraint gradients enforcing `tilt_in == tilt_out` on the rim."""
    _ = index_map, tilts_in, tilts_out
    group = _resolve_group(global_params)
    if group is None:
        return None

    rows = _collect_rows(mesh, group)
    if rows.size == 0:
        return None

    pts = positions[rows]
    normal = _fit_plane_normal(pts)
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    u, v = _orthonormal_basis(normal)

    constraints: list[tuple[np.ndarray | None, np.ndarray | None]] = []
    for dvec in (u, v):
        g_in = np.zeros_like(positions)
        g_out = np.zeros_like(positions)
        g_in[rows] += dvec
        g_out[rows] += -dvec
        constraints.append((g_in, g_out))
    return constraints


def enforce_tilt_constraint(mesh: Mesh, global_params=None, **_kwargs) -> None:
    """Project in-plane tilt components so `tilt_in == tilt_out` on the tagged rim."""
    group = _resolve_group(global_params)
    if group is None:
        return

    positions = mesh.positions_view()
    rows = _collect_rows(mesh, group)
    if rows.size == 0:
        return

    pts = positions[rows]
    normal = _fit_plane_normal(pts)
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    u, v = _orthonormal_basis(normal)
    mode = _resolve_mode(global_params)

    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")

    for row in rows:
        vid = int(mesh.vertex_ids[int(row)])
        fixed_in = bool(getattr(mesh.vertices[vid], "tilt_fixed_in", False))
        fixed_out = bool(getattr(mesh.vertices[vid], "tilt_fixed_out", False))
        if fixed_in and fixed_out:
            continue

        t_in = tilts_in[row]
        t_out = tilts_out[row]

        d_in = np.array([float(np.dot(t_in, u)), float(np.dot(t_in, v))])
        d_out = np.array([float(np.dot(t_out, u)), float(np.dot(t_out, v))])

        if mode == "in_to_out":
            target = d_in
        elif mode == "out_to_in":
            target = d_out
        else:
            target = 0.5 * (d_in + d_out)
            if fixed_in:
                target = d_in
            if fixed_out:
                target = d_out

        if not fixed_in:
            delta = target - d_in
            tilts_in[row] = t_in + delta[0] * u + delta[1] * v
        if not fixed_out:
            delta = target - d_out
            tilts_out[row] = t_out + delta[0] * u + delta[1] * v

    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)


__all__ = ["constraint_gradients_tilt_array", "enforce_tilt_constraint"]
