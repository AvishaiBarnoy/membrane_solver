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
    cache_key = (mesh._vertex_ids_version, str(group))
    cache_attr = "_tilt_thetaB_boundary_group_rows_cache"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["rows"]

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
    out = np.asarray(rows, dtype=int)
    setattr(mesh, cache_attr, {"key": cache_key, "rows": out})
    return out


def _augment_disk_ring_rows_geometrically(
    rows: np.ndarray,
    *,
    positions: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """Expand tagged disk rows to include same-plane, same-radius ring vertices."""
    if rows.size == 0:
        return rows

    pts = positions[rows]
    rel = pts - center[None, :]
    proj = rel - np.einsum("ij,j->i", rel, normal)[:, None] * normal[None, :]
    radii = np.linalg.norm(proj, axis=1)
    good = radii > 1e-12
    if not np.any(good):
        return rows

    target_radius = float(np.median(radii[good]))
    ring_pts = pts[good]
    spacing = 0.0
    if ring_pts.shape[0] >= 2:
        diff = ring_pts[:, None, :] - ring_pts[None, :, :]
        dmat = np.linalg.norm(diff, axis=2)
        dmat[dmat < 1e-12] = np.inf
        nearest = np.min(dmat, axis=1)
        nearest = nearest[np.isfinite(nearest)]
        if nearest.size:
            spacing = float(np.median(nearest))

    radial_tol = max(1e-8, 2e-3 * max(target_radius, 1.0), 0.2 * spacing)
    plane_tol = max(1e-8, 0.2 * radial_tol)

    all_rel = positions - center[None, :]
    plane_dist = np.abs(np.einsum("ij,j->i", all_rel, normal))
    all_proj = (
        all_rel - np.einsum("ij,j->i", all_rel, normal)[:, None] * normal[None, :]
    )
    all_radii = np.linalg.norm(all_proj, axis=1)

    candidate_mask = (np.abs(all_radii - target_radius) <= radial_tol) & (
        plane_dist <= plane_tol
    )
    candidate_rows = np.flatnonzero(candidate_mask).astype(int, copy=False)
    if candidate_rows.size == 0:
        return rows
    return np.unique(np.concatenate((rows.astype(int, copy=False), candidate_rows)))


def _boundary_directions(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return boundary rows and tangent-plane radial directions."""
    group = _resolve_group(global_params)
    if group is None:
        return None
    center = _resolve_center(global_params)
    raw_normal = (
        None if global_params is None else global_params.get("tilt_thetaB_normal")
    )
    normal_token = (
        None
        if raw_normal is None
        else tuple(np.asarray(raw_normal, dtype=float).reshape(3).tolist())
    )
    cache_key = (
        int(mesh._version),
        int(mesh._vertex_ids_version),
        str(group),
        tuple(center.tolist()),
        normal_token,
        id(positions),
    )
    cache_attr = "_tilt_thetaB_boundary_directions_cache"
    if mesh._geometry_cache_active(positions):
        cached = getattr(mesh, cache_attr, None)
        if cached is not None and cached.get("key") == cache_key:
            return cached.get("value")

    rows = _collect_group_rows(mesh, group)
    if rows.size == 0:
        return None

    tagged_pts = positions[rows]
    normal = _resolve_normal(global_params, tagged_pts)
    if str(group) == "disk":
        rows = _augment_disk_ring_rows_geometrically(
            rows,
            positions=positions,
            center=center,
            normal=normal,
        )
    pts = positions[rows]

    r_vec = pts - center[None, :]
    r_vec = r_vec - np.einsum("ij,j->i", r_vec, normal)[:, None] * normal[None, :]
    r_len = np.linalg.norm(r_vec, axis=1)
    good = r_len > 1e-12
    if not np.any(good):
        return None

    r_hat = np.zeros_like(r_vec)
    r_hat[good] = r_vec[good] / r_len[good][:, None]

    normals_v = mesh.vertex_normals(positions=positions)[rows]
    r_dir = r_hat - np.einsum("ij,ij->i", r_hat, normals_v)[:, None] * normals_v
    nrm = np.linalg.norm(r_dir, axis=1)
    ok = nrm > 1e-12
    r_dir[ok] = r_dir[ok] / nrm[ok][:, None]
    valid = good & ok
    if not np.any(valid):
        return None

    result = (rows[valid], r_dir[valid])
    if mesh._geometry_cache_active(positions):
        setattr(mesh, cache_attr, {"key": cache_key, "value": result})
    return result


def constraint_gradients(mesh: Mesh, global_params=None) -> None:
    """No shape constraints for this tilt-only module."""
    _ = mesh, global_params
    return None


def constraint_gradients_array(
    mesh: Mesh,
    global_params=None,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
) -> None:
    """No shape constraints for this tilt-only module."""
    _ = mesh, global_params, positions, index_map
    return None


def constraint_gradients_tilt_array(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
) -> list[tuple[np.ndarray | None, np.ndarray | None]] | None:
    """Return tilt-space KKT gradients for `t_in·r_dir = thetaB` constraints."""
    row_constraints = constraint_gradients_tilt_rows_array(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
        tilts_out=tilts_out,
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
            g_in[in_rows] += in_vecs
        if out_part is not None:
            out_rows, out_vecs = out_part
            g_out = np.zeros_like(positions)
            g_out[out_rows] += out_vecs
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
    """Return sparse tilt-space KKT gradients for `t_in·r_dir = thetaB`."""
    _ = index_map, tilts_in, tilts_out
    data = _boundary_directions(mesh, global_params, positions=positions)
    if data is None:
        return None
    rows, r_dir = data

    constraints: list[
        tuple[
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]
    ] = []
    for row, dvec in zip(rows, r_dir):
        vid = int(mesh.vertex_ids[int(row)])
        if getattr(mesh.vertices[vid], "tilt_fixed_in", False):
            continue
        constraints.append(
            (
                (
                    np.asarray([int(row)], dtype=int),
                    np.asarray(dvec, dtype=float).reshape(1, 3),
                ),
                None,
            )
        )
    return constraints or None


def enforce_tilt_constraint(mesh: Mesh, global_params=None, **_kwargs) -> None:
    """Project inner-leaflet radial tilt on the disk boundary to thetaB."""
    thetaB = (
        0.0
        if global_params is None
        else float(global_params.get("tilt_thetaB_value") or 0.0)
    )

    positions = mesh.positions_view()
    data = _boundary_directions(mesh, global_params, positions=positions)
    if data is None:
        return
    rows, r_dir = data

    rows_i = np.asarray(rows, dtype=int)
    if rows_i.size == 0:
        return

    vids = mesh.vertex_ids[rows_i]
    fixed_mask = np.fromiter(
        (
            bool(getattr(mesh.vertices[int(vid)], "tilt_fixed_in", False))
            for vid in vids
        ),
        dtype=bool,
        count=rows_i.size,
    )
    free = ~fixed_mask
    if not np.any(free):
        return

    rows_free = rows_i[free]
    d_free = np.asarray(r_dir, dtype=float)[free]

    # Update in-place to avoid full-array copy + scatter-back each relaxation call.
    tilts_in = mesh.tilts_in_view()
    t_free = tilts_in[rows_free]
    t_rad = np.einsum("ij,ij->i", t_free, d_free)
    tilts_in[rows_free] = t_free + (thetaB - t_rad)[:, None] * d_free
    mesh.touch_tilts_in()


__all__ = [
    "constraint_gradients",
    "constraint_gradients_array",
    "constraint_gradients_tilt_array",
    "constraint_gradients_tilt_rows_array",
    "enforce_tilt_constraint",
]
