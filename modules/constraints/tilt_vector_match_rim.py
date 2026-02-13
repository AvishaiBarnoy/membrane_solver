"""Hard per-leaflet in-plane tilt matching between a disk ring and rim ring.

Purpose
-------
For multi-disk geometries (e.g. disks on a sphere), "radial-only" rim matching
is not sufficient to ensure that the *full* in-plane director field is
continuous across the disk/annulus interface. This constraint enforces, for
each disk group, that the in-plane tilt vectors match between:

  - a "disk-side" ring just inside the rim (role: ``disk``), and
  - the rim ring at r=R (role: ``rim``).

The match is performed separately for each leaflet (tilt_in and tilt_out).

Tagging
-------
Attach these options to vertices that participate in matching:

  - ``tilt_vector_match_group``: string disk id (e.g. "cav1", "cav2", ...)
  - ``tilt_vector_match_role``: "disk" or "rim"

Vertices are paired per-group by sorting both rings by polar angle in the
group's local disk frame (center + fitted plane normal).

Constraints
-----------
For each paired (disk_i, rim_i) vertex and each leaflet, enforce equality of
the two in-plane components in the local disk frame basis (u, v):

  (t_rim - t_disk) · u = 0
  (t_rim - t_disk) · v = 0

We intentionally ignore the dependence of the local basis on positions for the
constraint gradients, so this module provides only *tilt* constraints (no shape
constraints).

Optional projection mode
------------------------
This module provides ``enforce_tilt_constraint`` to "snap" the tilts onto the
constraint manifold after accepted tilt steps. By default it projects by
averaging the in-plane components when both endpoints are free, and otherwise
adjusts the non-fixed endpoint.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from geometry.entities import Mesh


def _as_str(val) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


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


def _collect_groups(mesh: Mesh) -> dict[str, dict[str, np.ndarray]]:
    """Return group -> {"disk": rows, "rim": rows}."""
    grouped: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"disk": [], "rim": []}
    )
    for vid in mesh.vertex_ids:
        v = mesh.vertices[int(vid)]
        opts = getattr(v, "options", None) or {}
        group = _as_str(opts.get("tilt_vector_match_group"))
        role = _as_str(opts.get("tilt_vector_match_role"))
        if group is None or role is None:
            continue
        role_l = role.lower()
        if role_l not in {"disk", "rim"}:
            continue
        row = mesh.vertex_index_to_row.get(int(vid))
        if row is None:
            continue
        grouped[group][role_l].append(int(row))

    out: dict[str, dict[str, np.ndarray]] = {}
    for g, roles in grouped.items():
        out[g] = {
            "disk": np.asarray(roles["disk"], dtype=int),
            "rim": np.asarray(roles["rim"], dtype=int),
        }
    return out


def _order_by_angle(
    positions: np.ndarray, *, center: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    u, v = _orthonormal_basis(normal)
    rel = positions - center[None, :]
    rel_plane = rel - np.einsum("ij,j->i", rel, normal)[:, None] * normal[None, :]
    x = rel_plane @ u
    y = rel_plane @ v
    return np.argsort(np.arctan2(y, x))


def _resolve_projection_mode(global_params) -> str:
    raw = None if global_params is None else global_params.get("tilt_vector_match_mode")
    mode = str(raw or "average").strip().lower()
    if mode in {"rim_to_disk", "rim2disk"}:
        return "rim_to_disk"
    if mode in {"disk_to_rim", "disk2rim"}:
        return "disk_to_rim"
    return "average"


def _build_tilt_row_constraints(
    mesh: Mesh,
    positions: np.ndarray,
) -> (
    list[
        tuple[
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]
    ]
    | None
):
    groups = _collect_groups(mesh)
    if not groups:
        return None

    constraints: list[
        tuple[
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]
    ] = []
    for group, roles in groups.items():
        disk_rows = roles["disk"]
        rim_rows = roles["rim"]
        if disk_rows.size == 0 or rim_rows.size == 0:
            continue
        if disk_rows.size != rim_rows.size:
            continue

        disk_pos = positions[disk_rows]
        rim_pos = positions[rim_rows]
        center = np.mean(np.vstack([disk_pos, rim_pos]), axis=0)
        normal = _fit_plane_normal(disk_pos)
        if normal is None:
            normal = _fit_plane_normal(rim_pos)
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0], dtype=float)

        disk_order = _order_by_angle(disk_pos, center=center, normal=normal)
        rim_order = _order_by_angle(rim_pos, center=center, normal=normal)
        disk_rows = disk_rows[disk_order]
        rim_rows = rim_rows[rim_order]

        u, v = _orthonormal_basis(normal)
        for dvec in (u, v):
            n = int(rim_rows.size)
            in_rows = np.empty(2 * n, dtype=int)
            in_rows[:n] = rim_rows
            in_rows[n:] = disk_rows
            in_vecs = np.empty((2 * n, 3), dtype=float)
            in_vecs[:n] = dvec
            in_vecs[n:] = -dvec
            constraints.append(((in_rows, in_vecs), None))

            out_rows = np.empty(2 * n, dtype=int)
            out_rows[:n] = rim_rows
            out_rows[n:] = disk_rows
            out_vecs = np.empty((2 * n, 3), dtype=float)
            out_vecs[:n] = dvec
            out_vecs[n:] = -dvec
            constraints.append((None, (out_rows, out_vecs)))

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
    """Return tilt constraint gradients enforcing per-leaflet in-plane matching."""
    _ = global_params, index_map, tilts_in, tilts_out
    row_constraints = _build_tilt_row_constraints(mesh, positions)
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


def constraint_gradients_tilt_rows_array(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
) -> (
    list[
        tuple[
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]
    ]
    | None
):
    """Return sparse-row tilt constraints for per-leaflet in-plane matching."""
    _ = global_params, index_map, tilts_in, tilts_out
    return _build_tilt_row_constraints(mesh, positions)


def enforce_tilt_constraint(mesh: Mesh, global_params=None, **_kwargs) -> None:
    """Project in-plane tilt components to satisfy matching constraints."""
    positions = mesh.positions_view()
    mode = _resolve_projection_mode(global_params)

    groups = _collect_groups(mesh)
    if not groups:
        return

    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")

    for group, roles in groups.items():
        disk_rows = roles["disk"]
        rim_rows = roles["rim"]
        if disk_rows.size == 0 or rim_rows.size == 0:
            continue
        if disk_rows.size != rim_rows.size:
            continue

        disk_pos = positions[disk_rows]
        rim_pos = positions[rim_rows]
        center = np.mean(np.vstack([disk_pos, rim_pos]), axis=0)
        normal = _fit_plane_normal(disk_pos)
        if normal is None:
            normal = _fit_plane_normal(rim_pos)
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
        u, v = _orthonormal_basis(normal)

        disk_order = _order_by_angle(disk_pos, center=center, normal=normal)
        rim_order = _order_by_angle(rim_pos, center=center, normal=normal)
        disk_rows = disk_rows[disk_order]
        rim_rows = rim_rows[rim_order]

        for leaflet, tilts in (("in", tilts_in), ("out", tilts_out)):
            for dr, rr in zip(disk_rows, rim_rows):
                vid_disk = int(mesh.vertex_ids[int(dr)])
                vid_rim = int(mesh.vertex_ids[int(rr)])
                fixed_disk = bool(
                    getattr(mesh.vertices[vid_disk], f"tilt_fixed_{leaflet}", False)
                )
                fixed_rim = bool(
                    getattr(mesh.vertices[vid_rim], f"tilt_fixed_{leaflet}", False)
                )
                if fixed_disk and fixed_rim:
                    continue

                t_disk = tilts[dr]
                t_rim = tilts[rr]

                # Work only with the in-plane components in the (u,v) basis.
                d_disk = np.array([float(np.dot(t_disk, u)), float(np.dot(t_disk, v))])
                d_rim = np.array([float(np.dot(t_rim, u)), float(np.dot(t_rim, v))])

                if mode == "rim_to_disk":
                    target = d_disk
                    if not fixed_rim:
                        delta = target - d_rim
                        tilts[rr] = t_rim + delta[0] * u + delta[1] * v
                    elif not fixed_disk:
                        delta = target - d_rim
                        tilts[dr] = t_disk - delta[0] * u - delta[1] * v
                    continue

                if mode == "disk_to_rim":
                    target = d_rim
                    if not fixed_disk:
                        delta = target - d_disk
                        tilts[dr] = t_disk + delta[0] * u + delta[1] * v
                    elif not fixed_rim:
                        delta = target - d_disk
                        tilts[rr] = t_rim - delta[0] * u - delta[1] * v
                    continue

                # average
                avg = 0.5 * (d_disk + d_rim)
                if fixed_disk:
                    avg = d_disk
                if fixed_rim:
                    avg = d_rim
                if not fixed_disk:
                    delta = avg - d_disk
                    tilts[dr] = t_disk + delta[0] * u + delta[1] * v
                if not fixed_rim:
                    delta = avg - d_rim
                    tilts[rr] = t_rim + delta[0] * u + delta[1] * v

    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)


__all__ = [
    "constraint_gradients_tilt_array",
    "constraint_gradients_tilt_rows_array",
    "enforce_tilt_constraint",
]
