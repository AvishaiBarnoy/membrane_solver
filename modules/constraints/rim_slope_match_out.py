"""Hard rim matching constraint for outer-leaflet tilt vs slope.

This constraint enforces the (gamma=0) rim-matching conditions described in
`docs/tex/1_disk_3d.tex` using Lagrange-style gradient projection plus an
optional tilt-only projection hook:

    t_out · r_hat = phi
    t_in  · r_hat = theta_disk - phi

where
    phi = (h_out - h_rim) / (r_out - r_rim)

Implementation notes
--------------------
- Uses a small-slope approximation and only includes shape derivatives along
  the plane normal (ignores r_hat and radial-distance derivatives), matching
  the energy counterpart in `modules/energy/rim_slope_match_out.py`.
- Constraints are applied per-vertex when the disk ring has the same number of
  vertices as the rim; otherwise the disk tilt uses an arc-length-weighted mean.
- The constraint gradients are weighted by sqrt(arc-length) to approximate a
  line-integral weighting.
- This module provides:
    * constraint_gradients_array: shape-gradient constraints (positions).
    * constraint_gradients_tilt_array: tilt-gradient constraints (leaflets).
    * enforce_tilt_constraint: tilt-only projection to satisfy constraints.
"""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh


def _resolve_group(global_params, key: str) -> str | None:
    raw = None if global_params is None else global_params.get(key)
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _resolve_center(global_params) -> np.ndarray:
    center = (
        None if global_params is None else global_params.get("rim_slope_match_center")
    )
    if center is None:
        center = [0.0, 0.0, 0.0]
    return np.asarray(center, dtype=float).reshape(3)


def _resolve_normal(global_params) -> np.ndarray | None:
    raw = None if global_params is None else global_params.get("rim_slope_match_normal")
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float).reshape(3)
    nrm = float(np.linalg.norm(arr))
    if nrm < 1e-15:
        return None
    return arr / nrm


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


def _orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(trial, normal)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    u = trial - np.dot(trial, normal) * normal
    nrm = float(np.linalg.norm(u))
    if nrm < 1e-15:
        u = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        u = u / nrm
    v = np.cross(normal, u)
    nrm_v = float(np.linalg.norm(v))
    if nrm_v < 1e-15:
        v = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        v = v / nrm_v
    return u, v


def _collect_group_rows(mesh: Mesh, group: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") == group:
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    return np.asarray(rows, dtype=int)


def _order_by_angle(
    positions: np.ndarray, *, center: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    u, v = _orthonormal_basis(normal)
    rel = positions - center[None, :]
    rel_plane = rel - np.einsum("ij,j->i", rel, normal)[:, None] * normal[None, :]
    x = rel_plane @ u
    y = rel_plane @ v
    angles = np.arctan2(y, x)
    return np.argsort(angles)


def _arc_length_weights(positions: np.ndarray, order: np.ndarray) -> np.ndarray:
    n = len(order)
    if n == 0:
        return np.zeros(0, dtype=float)
    pos = positions[order]
    diffs_next = pos[(np.arange(n) + 1) % n] - pos
    diffs_prev = pos - pos[(np.arange(n) - 1) % n]
    l_next = np.linalg.norm(diffs_next, axis=1)
    l_prev = np.linalg.norm(diffs_prev, axis=1)
    return 0.5 * (l_next + l_prev)


def _build_matching_data(mesh: Mesh, global_params, positions: np.ndarray):
    group = _resolve_group(global_params, "rim_slope_match_group")
    outer_group = _resolve_group(global_params, "rim_slope_match_outer_group")
    disk_group = _resolve_group(global_params, "rim_slope_match_disk_group")
    theta_param = None
    if global_params is not None:
        theta_param = global_params.get("rim_slope_match_thetaB_param")
    if group is None or outer_group is None:
        return None

    rim_rows = _collect_group_rows(mesh, group)
    outer_rows = _collect_group_rows(mesh, outer_group)
    if rim_rows.size == 0 or outer_rows.size == 0:
        return None
    if rim_rows.size != outer_rows.size:
        return None

    center = _resolve_center(global_params)
    normal = _resolve_normal(global_params)
    if normal is None:
        normal = _fit_plane_normal(positions[rim_rows])
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    rim_pos = positions[rim_rows]
    outer_pos = positions[outer_rows]

    rim_order = _order_by_angle(rim_pos, center=center, normal=normal)
    outer_order = _order_by_angle(outer_pos, center=center, normal=normal)

    rim_rows = rim_rows[rim_order]
    outer_rows = outer_rows[outer_order]
    rim_pos = positions[rim_rows]
    outer_pos = positions[outer_rows]

    r_vec = rim_pos - center[None, :]
    r_vec = r_vec - np.einsum("ij,j->i", r_vec, normal)[:, None] * normal[None, :]
    r_len = np.linalg.norm(r_vec, axis=1)
    good = r_len > 1e-12
    if not np.any(good):
        return None

    r_hat = np.zeros_like(r_vec)
    r_hat[good] = r_vec[good] / r_len[good][:, None]

    h_rim = np.einsum("ij,j->i", rim_pos - center[None, :], normal)
    h_out = np.einsum("ij,j->i", outer_pos - center[None, :], normal)
    r_out_vec = outer_pos - center[None, :]
    r_out_vec = (
        r_out_vec - np.einsum("ij,j->i", r_out_vec, normal)[:, None] * normal[None, :]
    )
    r_out = np.linalg.norm(r_out_vec, axis=1)
    dr = r_out - r_len
    inv_dr = np.zeros_like(dr)
    valid = good & (np.abs(dr) > 1e-8)
    inv_dr[valid] = 1.0 / dr[valid]

    phi = np.zeros_like(dr)
    phi[valid] = (h_out[valid] - h_rim[valid]) * inv_dr[valid]

    weights = _arc_length_weights(rim_pos, np.arange(len(rim_rows)))
    weights = np.where(valid, weights, 0.0)
    weight_sqrt = np.sqrt(weights)

    disk_rows = None
    disk_r_hat = None
    disk_weights = None
    local_disk = False
    theta_scalar = None
    if theta_param is not None and global_params is not None:
        theta_scalar = float(global_params.get(str(theta_param)) or 0.0)
    if disk_group is not None:
        disk_rows = _collect_group_rows(mesh, disk_group)
        if disk_rows.size:
            disk_pos = positions[disk_rows]
            disk_order = _order_by_angle(disk_pos, center=center, normal=normal)
            disk_rows = disk_rows[disk_order]
            disk_pos = positions[disk_rows]
            r_vec_disk = disk_pos - center[None, :]
            r_vec_disk = (
                r_vec_disk
                - np.einsum("ij,j->i", r_vec_disk, normal)[:, None] * normal[None, :]
            )
            r_len_disk = np.linalg.norm(r_vec_disk, axis=1)
            disk_r_hat = np.zeros_like(r_vec_disk)
            good_disk = r_len_disk > 1e-12
            disk_r_hat[good_disk] = (
                r_vec_disk[good_disk] / r_len_disk[good_disk][:, None]
            )
            if disk_rows.size == rim_rows.size:
                local_disk = True
            else:
                disk_weights = _arc_length_weights(disk_pos, np.arange(len(disk_rows)))
                disk_weights = np.where(good_disk, disk_weights, 0.0)

    return {
        "rim_rows": rim_rows,
        "outer_rows": outer_rows,
        "disk_rows": disk_rows,
        "r_hat": r_hat,
        "disk_r_hat": disk_r_hat,
        "weight_sqrt": weight_sqrt,
        "inv_dr": inv_dr,
        "phi": phi,
        "valid": valid,
        "local_disk": local_disk,
        "disk_weights": disk_weights,
        "normal": normal,
        "theta_scalar": theta_scalar,
    }


def constraint_gradients_array(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
) -> list[np.ndarray] | None:
    """Return dense constraint gradients for rim matching (shape only)."""
    _ = index_map
    data = _build_matching_data(mesh, global_params, positions)
    if data is None:
        return None

    rim_rows = data["rim_rows"]
    outer_rows = data["outer_rows"]
    disk_rows = data["disk_rows"]
    weight_sqrt = data["weight_sqrt"]
    inv_dr = data["inv_dr"]
    valid = data["valid"]
    normal = data["normal"]

    constraints: list[np.ndarray] = []
    # Note: theta_scalar is available but does not change the gradient structure
    # for the asymmetric constraints:
    # 1. t_out . r = phi  => C = t_out.r - phi = 0
    #    Gradient is -grad(phi).
    # 2. t_in . r = theta_B - phi => C = t_in.r + phi - theta_B = 0
    #    Gradient is +grad(phi).

    for i, ok in enumerate(valid):
        if not ok or weight_sqrt[i] == 0.0:
            continue
        coeff = weight_sqrt[i] * inv_dr[i]

        g_out = np.zeros_like(positions)
        # Gradient of phi w.r.t z: rim (-inv_dr), outer (+inv_dr)
        # Gradient of -phi: rim (+inv_dr), outer (-inv_dr)
        # Multiplied by weight_sqrt (captured in coeff)
        g_out[rim_rows[i]] += coeff * normal
        g_out[outer_rows[i]] += -coeff * normal
        constraints.append(g_out)

        if disk_rows is not None:
            g_in = np.zeros_like(positions)
            # Gradient of +phi: rim (-inv_dr), outer (+inv_dr)
            g_in[rim_rows[i]] += -coeff * normal
            g_in[outer_rows[i]] += coeff * normal
            constraints.append(g_in)

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
    """Return dense constraint gradients for rim matching (tilts)."""
    _ = index_map, tilts_in, tilts_out
    data = _build_matching_data(mesh, global_params, positions)
    if data is None:
        return None

    rim_rows = data["rim_rows"]
    disk_rows = data["disk_rows"]
    r_hat = data["r_hat"]
    disk_r_hat = data["disk_r_hat"]
    weight_sqrt = data["weight_sqrt"]
    valid = data["valid"]
    local_disk = data["local_disk"]
    disk_weights = data["disk_weights"]
    theta_scalar = data["theta_scalar"]

    normals = mesh.vertex_normals(positions=positions)

    constraints: list[tuple[np.ndarray | None, np.ndarray | None]] = []

    for i, ok in enumerate(valid):
        if not ok or weight_sqrt[i] == 0.0:
            continue
        coeff = weight_sqrt[i]

        n = normals[rim_rows[i]]
        r_dir = r_hat[i] - float(np.dot(r_hat[i], n)) * n
        r_norm = float(np.linalg.norm(r_dir))
        if r_norm < 1e-12:
            continue
        r_dir = r_dir / r_norm

        g_out = np.zeros_like(positions)
        g_out[rim_rows[i]] += coeff * r_dir
        constraints.append((None, g_out))

        if disk_rows is None or disk_r_hat is None:
            continue

        # If theta_scalar is present, we still need a constraint on t_in.
        # But we don't need to project onto disk_rows (since theta_B is scalar).
        # However, the legacy code structure couples t_in to disk rows if local_disk is set.
        # But for scalar theta_B, the constraint is t_in . r = theta_B - phi.
        # The gradient w.r.t t_in is just r_dir.
        # The gradient w.r.t disk tilts is 0 (since theta_B is scalar).

        g_in = np.zeros_like(positions)
        g_in[rim_rows[i]] += coeff * r_dir

        if theta_scalar is None:
            # Legacy: theta_disk depends on disk tilts. Add gradient.
            if local_disk:
                g_in[disk_rows[i]] += -coeff * disk_r_hat[i]
            else:
                if disk_weights is None:
                    continue
                weight_sum = float(np.sum(disk_weights))
                if weight_sum > 0.0:
                    factors = (disk_weights / weight_sum)[:, None] * disk_r_hat
                    g_in[disk_rows] += -coeff * factors

        constraints.append((g_in, None))

    return constraints or None


def enforce_tilt_constraint(mesh: Mesh, global_params=None, **_kwargs) -> None:
    """Project rim tilt components to satisfy the matching condition."""
    positions = mesh.positions_view()
    data = _build_matching_data(mesh, global_params, positions)
    if data is None:
        return

    rim_rows = data["rim_rows"]
    disk_rows = data["disk_rows"]
    r_hat = data["r_hat"]
    disk_r_hat = data["disk_r_hat"]
    phi = data["phi"]
    valid = data["valid"]
    local_disk = data["local_disk"]
    disk_weights = data["disk_weights"]
    theta_scalar = data["theta_scalar"]

    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")
    normals = mesh.vertex_normals(positions=positions)

    if theta_scalar is not None:
        theta_disk = theta_scalar
    elif disk_rows is not None and disk_r_hat is not None:
        if local_disk:
            theta_disk = np.einsum("ij,ij->i", tilts_in[disk_rows], disk_r_hat)
        else:
            weight_sum = (
                float(np.sum(disk_weights)) if disk_weights is not None else 0.0
            )
            if weight_sum > 0.0:
                theta_disk = float(
                    np.sum(
                        disk_weights
                        * np.einsum("ij,ij->i", tilts_in[disk_rows], disk_r_hat)
                    )
                    / weight_sum
                )
            else:
                theta_disk = None
    else:
        theta_disk = None

    for i, ok in enumerate(valid):
        if not ok:
            continue
        vid_rim = mesh.vertex_ids[rim_rows[i]]
        n = normals[rim_rows[i]]
        r_dir = r_hat[i] - float(np.dot(r_hat[i], n)) * n
        r_norm = float(np.linalg.norm(r_dir))
        if r_norm < 1e-12:
            continue
        r_dir = r_dir / r_norm

        # Constraint 1: t_out . r = phi
        target_out = phi[i]

        if not getattr(mesh.vertices[int(vid_rim)], "tilt_fixed_out", False):
            t_out = tilts_out[rim_rows[i]]
            t_out_rad = float(np.dot(t_out, r_dir))
            tilts_out[rim_rows[i]] = t_out + (target_out - t_out_rad) * r_dir

        if theta_disk is None:
            continue

        # Constraint 2: t_in . r = theta_B - phi
        if theta_scalar is not None:
            target_in = float(theta_disk - phi[i])
        elif local_disk:
            target_in = float(theta_disk[i] - phi[i])
        else:
            target_in = float(theta_disk - phi[i])

        if not getattr(mesh.vertices[int(vid_rim)], "tilt_fixed_in", False):
            t_in = tilts_in[rim_rows[i]]
            t_in_rad = float(np.dot(t_in, r_dir))
            tilts_in[rim_rows[i]] = t_in + (target_in - t_in_rad) * r_dir

    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)


__all__ = [
    "constraint_gradients_array",
    "constraint_gradients_tilt_array",
    "enforce_tilt_constraint",
]
