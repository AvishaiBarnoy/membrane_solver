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

import logging

import numpy as np

from geometry.entities import Mesh

logger = logging.getLogger("membrane_solver")
_WARNED_DISK_EQUALS_RIM = False


def _resolve_group(global_params, key: str) -> str | None:
    raw = None if global_params is None else global_params.get(key)
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _sanitize_disk_group(
    *, rim_group: str | None, disk_group: str | None
) -> str | None:
    """Disable degenerate disk-group coupling when it equals the rim group."""
    if rim_group is None or disk_group is None:
        return disk_group
    if disk_group != rim_group:
        return disk_group
    global _WARNED_DISK_EQUALS_RIM
    if not _WARNED_DISK_EQUALS_RIM:
        logger.warning(
            "rim_slope_match_disk_group matches rim_slope_match_group (%s); "
            "skipping disk-side coupling to avoid degenerate constraints.",
            rim_group,
        )
        _WARNED_DISK_EQUALS_RIM = True
    return None


def _resolve_center(global_params) -> np.ndarray:
    center = (
        None if global_params is None else global_params.get("rim_slope_match_center")
    )
    if center is None:
        center = [0.0, 0.0, 0.0]
    return np.asarray(center, dtype=float).reshape(3)


def _resolve_matching_mode(global_params) -> str:
    raw = None if global_params is None else global_params.get("rim_slope_match_mode")
    mode = "pointwise_radial_v1" if raw is None else str(raw).strip().lower()
    if mode not in {
        "pointwise_radial_v1",
        "ring_average_radial_v1",
        "shared_rim_staggered_v1",
    }:
        raise ValueError(
            "rim_slope_match_mode must be 'pointwise_radial_v1' or "
            "'ring_average_radial_v1' or 'shared_rim_staggered_v1'."
        )
    return mode


def _tilt_target_rows_weights_and_direction(
    *,
    data: dict,
    positions: np.ndarray,
    normals: np.ndarray,
    i: int,
    matching_mode: str,
) -> tuple[list[int], list[float], np.ndarray] | None:
    """Return target rows and tangent radial direction for tilt matching."""
    r_hat = data["r_hat"]
    if matching_mode == "shared_rim_staggered_v1":
        outer_rows = np.asarray(data["outer_rows"], dtype=int)
        row0 = int(outer_rows[np.asarray(data["outer_idx0"], dtype=int)[i]])
        row1 = int(outer_rows[np.asarray(data["outer_idx1"], dtype=int)[i]])
        w0 = float(np.asarray(data["outer_w0"], dtype=float)[i])
        w1 = float(np.asarray(data["outer_w1"], dtype=float)[i])
        normal = w0 * np.asarray(normals[row0], dtype=float) + w1 * np.asarray(
            normals[row1], dtype=float
        )
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1.0e-12:
            return None
        normal = normal / normal_norm
        rows = [row0]
        weights = [w0]
        if row1 != row0 or abs(w1) > 1.0e-12:
            rows.append(row1)
            weights.append(w1)
    else:
        row = int(np.asarray(data["rim_rows"], dtype=int)[i])
        normal = np.asarray(normals[row], dtype=float)
        rows = [row]
        weights = [1.0]

    r_dir = np.asarray(r_hat[i], dtype=float) - float(np.dot(r_hat[i], normal)) * normal
    r_norm = float(np.linalg.norm(r_dir))
    if r_norm < 1e-12:
        return None
    r_dir = r_dir / r_norm
    return rows, weights, r_dir


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
    cache_key = (mesh._vertex_ids_version, str(group))
    cache_attr = "_rim_slope_match_group_rows_cache"
    cached = getattr(mesh, cache_attr, None)
    if isinstance(cached, dict) and "entries" in cached:
        if cached.get("key") == cache_key and "rows" in cached:
            return cached["rows"]
        entries = cached["entries"]
        rows = entries.get(cache_key)
        if rows is not None:
            return rows
    else:
        entries = {}
        if isinstance(cached, dict) and cached.get("key") == cache_key:
            return cached["rows"]

    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") == group:
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    out = np.asarray(rows, dtype=int)
    entries[cache_key] = out
    if len(entries) > 32:
        # Keep the cache bounded while preserving recent keys.
        entries = dict(list(entries.items())[-16:])
    setattr(mesh, cache_attr, {"entries": entries, "key": cache_key, "rows": out})
    return out


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


def _arc_length_params(positions: np.ndarray) -> tuple[np.ndarray, float]:
    n = len(positions)
    if n == 0:
        return np.zeros(0, dtype=float), 0.0
    diffs = positions[(np.arange(n) + 1) % n] - positions
    seg = np.linalg.norm(diffs, axis=1)
    total = float(np.sum(seg))
    if total <= 0.0:
        return np.zeros(n, dtype=float), 0.0
    s = np.concatenate(([0.0], np.cumsum(seg[:-1], dtype=float)))
    s /= total
    return s, total


def _interp_ring_positions(
    positions: np.ndarray, s_targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    n = len(positions)
    if n == 0 or s_targets.size == 0:
        return None
    s_out, total = _arc_length_params(positions)
    if total <= 0.0 or s_out.size < 2:
        return None
    idx1 = np.searchsorted(s_out, s_targets, side="right") % n
    idx0 = (idx1 - 1) % n
    s0 = s_out[idx0]
    s1 = s_out[idx1]
    s1_adj = s1.copy()
    s_targets_adj = s_targets.copy()
    wrap = s1_adj <= s0
    s1_adj[wrap] += 1.0
    s_targets_adj = np.where(s_targets_adj < s0, s_targets_adj + 1.0, s_targets_adj)
    denom = s1_adj - s0
    t = np.zeros_like(s_targets_adj)
    mask = denom > 1e-12
    t[mask] = (s_targets_adj[mask] - s0[mask]) / denom[mask]
    w1 = t
    w0 = 1.0 - t
    interp = positions[idx0] * w0[:, None] + positions[idx1] * w1[:, None]
    return interp, idx0, idx1, w0, w1


def _build_matching_data(mesh: Mesh, global_params, positions: np.ndarray):
    group = _resolve_group(global_params, "rim_slope_match_group")
    outer_group = _resolve_group(global_params, "rim_slope_match_outer_group")
    disk_group = _resolve_group(global_params, "rim_slope_match_disk_group")
    disk_group = _sanitize_disk_group(rim_group=group, disk_group=disk_group)
    theta_param = None
    if global_params is not None:
        theta_param = global_params.get("rim_slope_match_thetaB_param")
    if group is None or outer_group is None:
        return None
    center = _resolve_center(global_params)
    raw_normal = _resolve_normal(global_params)
    normal_token = (
        None
        if raw_normal is None
        else tuple(np.asarray(raw_normal, dtype=float).reshape(3).tolist())
    )
    theta_token = None
    if theta_param is not None and global_params is not None:
        theta_token = float(global_params.get(str(theta_param)) or 0.0)
    cache_key = (
        int(mesh._version),
        int(mesh._vertex_ids_version),
        str(group),
        str(outer_group),
        None if disk_group is None else str(disk_group),
        None if theta_param is None else str(theta_param),
        theta_token,
        tuple(center.tolist()),
        normal_token,
        id(positions),
    )
    cache_attr = "_rim_slope_match_data_cache"
    if mesh._geometry_cache_active(positions):
        cached = getattr(mesh, cache_attr, None)
        if isinstance(cached, dict) and "entries" in cached:
            value = cached["entries"].get(cache_key)
            if value is not None:
                return value
        elif cached is not None and cached.get("key") == cache_key:
            return cached.get("value")

    rim_rows = _collect_group_rows(mesh, group)
    outer_rows = _collect_group_rows(mesh, outer_group)
    if rim_rows.size == 0 or outer_rows.size == 0:
        return None

    normal = raw_normal
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
    outer_idx0 = np.arange(len(rim_rows), dtype=int)
    outer_idx1 = np.arange(len(rim_rows), dtype=int)
    outer_w0 = np.ones(len(rim_rows), dtype=float)
    outer_w1 = np.zeros(len(rim_rows), dtype=float)
    if rim_rows.size != outer_rows.size:
        s_rim, total_rim = _arc_length_params(rim_pos)
        if total_rim <= 0.0:
            return None
        interp = _interp_ring_positions(outer_pos, s_rim)
        if interp is None:
            return None
        outer_pos, outer_idx0, outer_idx1, outer_w0, outer_w1 = interp

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
    theta_scalar = theta_token
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

    result = {
        "rim_rows": rim_rows,
        "outer_rows": outer_rows,
        "outer_idx0": outer_idx0,
        "outer_idx1": outer_idx1,
        "outer_w0": outer_w0,
        "outer_w1": outer_w1,
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
    if mesh._geometry_cache_active(positions):
        cached = getattr(mesh, cache_attr, None)
        entries = cached.get("entries", {}) if isinstance(cached, dict) else {}
        entries[cache_key] = result
        if len(entries) > 16:
            entries = dict(list(entries.items())[-8:])
        setattr(mesh, cache_attr, {"entries": entries})
    return result


def matching_ring_diagnostics(mesh: Mesh, global_params, positions: np.ndarray) -> dict:
    """Report tagged ring candidates seen by the rim-matching constraint path."""
    group = _resolve_group(global_params, "rim_slope_match_group")
    outer_group = _resolve_group(global_params, "rim_slope_match_outer_group")
    disk_group = _resolve_group(global_params, "rim_slope_match_disk_group")
    disk_group = _sanitize_disk_group(rim_group=group, disk_group=disk_group)
    center = _resolve_center(global_params)
    normal = _resolve_normal(global_params)

    out: dict[str, object] = {
        "available": False,
        "reason": "missing_groups",
        "rim_group": group,
        "outer_group": outer_group,
        "disk_group": disk_group,
        "outer_source": "tagged_group",
    }
    if group is None or outer_group is None:
        return out

    rim_rows = _collect_group_rows(mesh, group)
    outer_rows = _collect_group_rows(mesh, outer_group)
    disk_rows = (
        np.zeros(0, dtype=int)
        if disk_group is None
        else _collect_group_rows(mesh, disk_group)
    )
    if rim_rows.size == 0:
        out["reason"] = "missing_rim_group_rows"
    elif outer_rows.size == 0:
        out["reason"] = "missing_outer_group_rows"
    else:
        out["available"] = True
        out["reason"] = "ok"

    if normal is None and rim_rows.size > 0:
        normal = _fit_plane_normal(positions[rim_rows])
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    rel_all = positions - center[None, :]
    rel_all = rel_all - np.einsum("ij,j->i", rel_all, normal)[:, None] * normal[None, :]
    r_all = np.linalg.norm(rel_all, axis=1)

    def _median_radius(rows: np.ndarray) -> float:
        if rows.size == 0:
            return float("nan")
        return float(np.median(r_all[rows]))

    out["rim_count"] = int(rim_rows.size)
    out["outer_count"] = int(outer_rows.size)
    out["disk_count"] = int(disk_rows.size)
    out["rim_radius"] = _median_radius(rim_rows)
    out["outer_radius"] = _median_radius(outer_rows)
    out["disk_radius"] = _median_radius(disk_rows)
    return out


def coarse_rim_family_diagnostics(
    mesh: Mesh, global_params, positions: np.ndarray
) -> dict[str, object]:
    """Describe the coarse tagged rim family and its immediate radial neighbors."""
    group = _resolve_group(global_params, "rim_slope_match_group")
    center = _resolve_center(global_params)
    normal = _resolve_normal(global_params)
    out: dict[str, object] = {
        "available": False,
        "reason": "missing_rim_group",
        "rim_group": group,
    }
    if group is None:
        return out

    rim_rows = _collect_group_rows(mesh, group)
    if rim_rows.size == 0:
        out["reason"] = "missing_rim_group_rows"
        return out

    if normal is None:
        normal = _fit_plane_normal(positions[rim_rows])
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    rel_all = positions - center[None, :]
    rel_all = rel_all - np.einsum("ij,j->i", rel_all, normal)[:, None] * normal[None, :]
    r_all = np.linalg.norm(rel_all, axis=1)
    rim_radii = r_all[rim_rows]
    rim_radius_unique, rim_radius_counts = np.unique(
        np.round(rim_radii, 12), return_counts=True
    )

    rim_vids = {int(mesh.vertex_ids[row]) for row in rim_rows.tolist()}
    neighbor_rows: set[int] = set()
    for edge in mesh.edges.values():
        tail = int(edge.tail_index)
        head = int(edge.head_index)
        if tail in rim_vids and head not in rim_vids:
            row = mesh.vertex_index_to_row.get(head)
            if row is not None:
                neighbor_rows.add(int(row))
        if head in rim_vids and tail not in rim_vids:
            row = mesh.vertex_index_to_row.get(tail)
            if row is not None:
                neighbor_rows.add(int(row))

    neighbor_rows_arr = np.asarray(sorted(neighbor_rows), dtype=int)
    inner_rows = np.zeros(0, dtype=int)
    outer_rows = np.zeros(0, dtype=int)
    if neighbor_rows_arr.size:
        tol = max(1.0e-9, 1.0e-5 * max(1.0, float(np.max(rim_radii))))
        rim_r_min = float(np.min(rim_radii))
        rim_r_max = float(np.max(rim_radii))
        neighbor_r = r_all[neighbor_rows_arr]
        inner_rows = neighbor_rows_arr[neighbor_r < (rim_r_min - tol)]
        outer_rows = neighbor_rows_arr[neighbor_r > (rim_r_max + tol)]

    def _radius_block(rows: np.ndarray) -> dict[str, object]:
        if rows.size == 0:
            return {
                "count": 0,
                "radius_min": float("nan"),
                "radius_max": float("nan"),
                "radius_unique": [],
            }
        vals = r_all[rows]
        unique_vals = np.unique(np.round(vals, 12))
        return {
            "count": int(rows.size),
            "radius_min": float(np.min(vals)),
            "radius_max": float(np.max(vals)),
            "radius_unique": [float(v) for v in unique_vals.tolist()],
        }

    out.update(
        {
            "available": True,
            "reason": "ok",
            "rim_count": int(rim_rows.size),
            "rim_radius_min": float(np.min(rim_radii)),
            "rim_radius_max": float(np.max(rim_radii)),
            "rim_radius_unique": [float(v) for v in rim_radius_unique.tolist()],
            "rim_radius_unique_counts": [int(v) for v in rim_radius_counts.tolist()],
            "inner_neighbor": _radius_block(inner_rows),
            "outer_neighbor": _radius_block(outer_rows),
        }
    )
    return out


def _residual_summary(values: np.ndarray, angles: np.ndarray) -> dict[str, float]:
    """Summarize residual structure with bulk stats and first azimuthal mode."""
    if values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "max_abs": float("nan"),
            "first_mode_cos": float("nan"),
            "first_mode_sin": float("nan"),
            "first_mode_amplitude": float("nan"),
        }
    mean = float(np.mean(values))
    median = float(np.median(values))
    std = float(np.std(values))
    max_abs = float(np.max(np.abs(values)))
    cos1 = float(np.mean(values * np.cos(angles)))
    sin1 = float(np.mean(values * np.sin(angles)))
    amp1 = float(np.hypot(cos1, sin1))
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "max_abs": max_abs,
        "first_mode_cos": cos1,
        "first_mode_sin": sin1,
        "first_mode_amplitude": amp1,
    }


def matching_residual_diagnostics(
    mesh: Mesh, global_params, positions: np.ndarray
) -> dict[str, object]:
    """Report residuals for the active coarse rim-matching law without enforcing it."""
    ring_diag = matching_ring_diagnostics(mesh, global_params, positions)
    data = _build_matching_data(mesh, global_params, positions)
    out: dict[str, object] = {
        "available": False,
        "reason": str(ring_diag.get("reason", "missing_matching_data")),
        "phi_estimator": "tagged_group_secant_dr",
        "matching_mode": _resolve_matching_mode(global_params),
        "disk_theta_source": "unavailable",
    }
    if data is None:
        return out

    rim_rows = data["rim_rows"]
    disk_rows = data["disk_rows"]
    outer_rows = data["outer_rows"]
    r_hat = data["r_hat"]
    disk_r_hat = data["disk_r_hat"]
    phi = data["phi"]
    valid = np.asarray(data["valid"], dtype=bool)
    local_disk = bool(data["local_disk"])
    disk_weights = data["disk_weights"]
    normal = np.asarray(data["normal"], dtype=float)
    theta_scalar = data["theta_scalar"]
    center = _resolve_center(global_params)

    rim_pos = positions[rim_rows]
    u, v = _orthonormal_basis(normal)
    rel = rim_pos - center[None, :]
    rel_plane = rel - np.einsum("ij,j->i", rel, normal)[:, None] * normal[None, :]
    rim_angles = np.arctan2(rel_plane @ v, rel_plane @ u)

    normals = mesh.vertex_normals(positions=positions)
    matching_mode = str(out["matching_mode"])
    r_dir = np.zeros_like(r_hat)
    good_dir = np.zeros(r_hat.shape[0], dtype=bool)
    tilt_rows0 = np.asarray(data["rim_rows"], dtype=int)
    tilt_rows1 = np.asarray(data["rim_rows"], dtype=int)
    tilt_w0 = np.ones(r_hat.shape[0], dtype=float)
    tilt_w1 = np.zeros(r_hat.shape[0], dtype=float)
    if matching_mode == "shared_rim_staggered_v1":
        outer_rows = np.asarray(data["outer_rows"], dtype=int)
        tilt_rows0 = outer_rows[np.asarray(data["outer_idx0"], dtype=int)]
        tilt_rows1 = outer_rows[np.asarray(data["outer_idx1"], dtype=int)]
        tilt_w0 = np.asarray(data["outer_w0"], dtype=float)
        tilt_w1 = np.asarray(data["outer_w1"], dtype=float)
    for i in range(r_hat.shape[0]):
        n = tilt_w0[i] * normals[tilt_rows0[i]] + tilt_w1[i] * normals[tilt_rows1[i]]
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        tangent_dir = r_hat[i] - float(np.dot(r_hat[i], n)) * n
        tangent_norm = float(np.linalg.norm(tangent_dir))
        if tangent_norm < 1e-12:
            continue
        r_dir[i] = tangent_dir / tangent_norm
        good_dir[i] = True

    valid = valid & good_dir
    out.update(
        {
            "available": bool(np.any(valid)),
            "reason": "ok" if bool(np.any(valid)) else "no_valid_rows",
            "sample_count": int(rim_rows.size),
            "valid_count": int(np.count_nonzero(valid)),
            "rim_count": int(rim_rows.size),
            "disk_count": 0 if disk_rows is None else int(disk_rows.size),
            "local_disk": bool(local_disk),
            "theta_param_used": bool(theta_scalar is not None),
        }
    )
    if not np.any(valid):
        return out

    phi_valid = phi[valid]
    angles_valid = rim_angles[valid]
    tilt_out_rad = (
        tilt_w0 * np.einsum("ij,ij->i", mesh.tilts_out_view()[tilt_rows0], r_dir)
        + tilt_w1 * np.einsum("ij,ij->i", mesh.tilts_out_view()[tilt_rows1], r_dir)
    )[valid]
    outer_residual = tilt_out_rad - phi_valid

    out["phi_mean"] = float(np.mean(phi_valid))
    out["phi_median"] = float(np.median(phi_valid))
    out["phi_std"] = float(np.std(phi_valid))
    out["phi_max_abs"] = float(np.max(np.abs(phi_valid)))
    out["outer_residual"] = _residual_summary(outer_residual, angles_valid)

    if disk_rows is None or disk_r_hat is None:
        out["disk_theta_source"] = "unavailable"
        out["inner_residual_available"] = False
        return out

    if theta_scalar is not None:
        theta_disk = np.full(phi.shape, float(theta_scalar), dtype=float)
        out["disk_theta_source"] = "global_param"
    elif local_disk:
        theta_disk = np.einsum("ij,ij->i", mesh.tilts_in_view()[disk_rows], disk_r_hat)
        out["disk_theta_source"] = "local_disk_rows"
    else:
        if disk_weights is None:
            out["disk_theta_source"] = "missing_disk_weights"
            out["inner_residual_available"] = False
            return out
        weight_sum = float(np.sum(disk_weights))
        if weight_sum <= 0.0:
            out["disk_theta_source"] = "degenerate_disk_weights"
            out["inner_residual_available"] = False
            return out
        theta_disk_scalar = float(
            np.sum(
                np.asarray(disk_weights, dtype=float)
                * np.einsum("ij,ij->i", mesh.tilts_in_view()[disk_rows], disk_r_hat)
            )
            / weight_sum
        )
        theta_disk = np.full(phi.shape, theta_disk_scalar, dtype=float)
        out["disk_theta_source"] = "weighted_disk_mean"

    tilt_in_rad = (
        tilt_w0 * np.einsum("ij,ij->i", mesh.tilts_in_view()[tilt_rows0], r_dir)
        + tilt_w1 * np.einsum("ij,ij->i", mesh.tilts_in_view()[tilt_rows1], r_dir)
    )[valid]
    inner_target = theta_disk[valid] - phi_valid
    inner_residual = tilt_in_rad - inner_target
    out["inner_residual_available"] = True
    out["theta_disk_mean"] = float(np.mean(theta_disk[valid]))
    out["theta_disk_median"] = float(np.median(theta_disk[valid]))
    out["inner_residual"] = _residual_summary(inner_residual, angles_valid)
    return out


def constraint_gradients_array(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
) -> list[np.ndarray] | None:
    """Return dense constraint gradients for rim matching (shape only)."""
    row_constraints = constraint_gradients_rows_array(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
    )
    if not row_constraints:
        return None
    constraints: list[np.ndarray] = []
    for rows, vecs in row_constraints:
        g_arr = np.zeros_like(positions)
        np.add.at(g_arr, rows, vecs)
        constraints.append(g_arr)
    return constraints or None


def constraint_gradients_rows_array(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Return sparse-row constraint gradients for rim matching (shape only)."""
    _ = index_map
    data = _build_matching_data(mesh, global_params, positions)
    if data is None:
        return None

    rim_rows = data["rim_rows"]
    outer_rows = data["outer_rows"]
    outer_idx0 = data["outer_idx0"]
    outer_idx1 = data["outer_idx1"]
    outer_w0 = data["outer_w0"]
    outer_w1 = data["outer_w1"]
    disk_rows = data["disk_rows"]
    weight_sqrt = data["weight_sqrt"]
    inv_dr = data["inv_dr"]
    valid = data["valid"]
    normal = data["normal"]
    matching_mode = _resolve_matching_mode(global_params)

    constraints: list[tuple[np.ndarray, np.ndarray]] = []
    # Note: theta_scalar is available but does not change the gradient structure
    # for the asymmetric constraints:
    # 1. t_out . r = phi  => C = t_out.r - phi = 0
    #    Gradient is -grad(phi).
    # 2. t_in . r = theta_B - phi => C = t_in.r + phi - theta_B = 0
    #    Gradient is +grad(phi).

    rows_out_all: list[int] = []
    vecs_out_all: list[np.ndarray] = []
    rows_in_all: list[int] = []
    vecs_in_all: list[np.ndarray] = []

    for i, ok in enumerate(valid):
        if not ok or weight_sqrt[i] == 0.0:
            continue
        coeff = weight_sqrt[i] * inv_dr[i]

        out0 = outer_rows[outer_idx0[i]]
        out1 = outer_rows[outer_idx1[i]]
        rows_out = [int(rim_rows[i]), int(out0)]
        vecs_out = [coeff * normal, -coeff * outer_w0[i] * normal]
        if out1 != out0 or outer_w1[i] != 0.0:
            rows_out.append(int(out1))
            vecs_out.append(-coeff * outer_w1[i] * normal)

        if matching_mode == "ring_average_radial_v1":
            rows_out_all.extend(rows_out)
            vecs_out_all.extend(vecs_out)
        else:
            constraints.append(
                (
                    np.asarray(rows_out, dtype=int),
                    np.asarray(vecs_out, dtype=float),
                )
            )

        if disk_rows is not None:
            rows_in = [int(rim_rows[i]), int(out0)]
            vecs_in = [-coeff * normal, coeff * outer_w0[i] * normal]
            if out1 != out0 or outer_w1[i] != 0.0:
                rows_in.append(int(out1))
                vecs_in.append(coeff * outer_w1[i] * normal)
            if matching_mode == "ring_average_radial_v1":
                rows_in_all.extend(rows_in)
                vecs_in_all.extend(vecs_in)
            else:
                constraints.append(
                    (
                        np.asarray(rows_in, dtype=int),
                        np.asarray(vecs_in, dtype=float),
                    )
                )

    if matching_mode == "ring_average_radial_v1":
        if rows_out_all:
            constraints.append(
                (
                    np.asarray(rows_out_all, dtype=int),
                    np.asarray(vecs_out_all, dtype=float),
                )
            )
        if rows_in_all:
            constraints.append(
                (
                    np.asarray(rows_in_all, dtype=int),
                    np.asarray(vecs_in_all, dtype=float),
                )
            )

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
            rows_in, vecs_in = in_part
            g_in = np.zeros_like(positions)
            np.add.at(g_in, rows_in, vecs_in)
        if out_part is not None:
            rows_out, vecs_out = out_part
            g_out = np.zeros_like(positions)
            np.add.at(g_out, rows_out, vecs_out)
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
    """Return sparse-row constraint gradients for rim matching (tilts)."""
    _ = index_map, tilts_in, tilts_out
    data = _build_matching_data(mesh, global_params, positions)
    if data is None:
        return None

    disk_rows = data["disk_rows"]
    disk_r_hat = data["disk_r_hat"]
    weight_sqrt = data["weight_sqrt"]
    valid = data["valid"]
    local_disk = data["local_disk"]
    disk_weights = data["disk_weights"]
    theta_scalar = data["theta_scalar"]
    matching_mode = _resolve_matching_mode(global_params)

    normals = mesh.vertex_normals(positions=positions)

    constraints: list[
        tuple[
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]
    ] = []

    agg_out_rows: list[int] = []
    agg_out_vecs: list[np.ndarray] = []
    agg_in_rows: list[int] = []
    agg_in_vecs: list[np.ndarray] = []

    for i, ok in enumerate(valid):
        if not ok or weight_sqrt[i] == 0.0:
            continue
        coeff = weight_sqrt[i]

        target = _tilt_target_rows_weights_and_direction(
            data=data,
            positions=positions,
            normals=normals,
            i=i,
            matching_mode=matching_mode,
        )
        if target is None:
            continue
        target_rows, target_weights, r_dir = target
        out_part = (
            np.asarray(target_rows, dtype=int),
            np.asarray([coeff * float(w) * r_dir for w in target_weights], dtype=float),
        )

        if disk_rows is None or disk_r_hat is None:
            if matching_mode == "ring_average_radial_v1":
                agg_out_rows.extend(out_part[0].tolist())
                agg_out_vecs.extend(out_part[1])
            else:
                constraints.append((None, out_part))
            continue

        rows_in = [int(row) for row in target_rows]
        vecs_in = [coeff * float(w) * r_dir for w in target_weights]

        if theta_scalar is None:
            if local_disk:
                rows_in.append(int(disk_rows[i]))
                vecs_in.append(-coeff * disk_r_hat[i])
            else:
                if disk_weights is None:
                    constraints.append((None, out_part))
                    continue
                weight_sum = float(np.sum(disk_weights))
                if weight_sum > 0.0:
                    factors = (disk_weights / weight_sum)[:, None] * disk_r_hat
                    for row_idx, factor in zip(disk_rows, factors):
                        rows_in.append(int(row_idx))
                        vecs_in.append(-coeff * factor)

        in_part = (
            np.asarray(rows_in, dtype=int),
            np.asarray(vecs_in, dtype=float),
        )
        if matching_mode == "ring_average_radial_v1":
            agg_out_rows.extend(out_part[0].tolist())
            agg_out_vecs.extend(out_part[1])
            agg_in_rows.extend(in_part[0].tolist())
            agg_in_vecs.extend(in_part[1])
        else:
            constraints.append((None, out_part))
            constraints.append((in_part, None))

    if matching_mode == "ring_average_radial_v1":
        if agg_out_rows:
            constraints.append(
                (
                    None,
                    (
                        np.asarray(agg_out_rows, dtype=int),
                        np.asarray(agg_out_vecs, dtype=float),
                    ),
                )
            )
        if agg_in_rows:
            constraints.append(
                (
                    (
                        np.asarray(agg_in_rows, dtype=int),
                        np.asarray(agg_in_vecs, dtype=float),
                    ),
                    None,
                )
            )

    return constraints or None


def enforce_tilt_constraint(mesh: Mesh, global_params=None, **_kwargs) -> None:
    """Project rim tilt components to satisfy the matching condition."""
    positions = mesh.positions_view()
    data = _build_matching_data(mesh, global_params, positions)
    if data is None:
        return

    disk_rows = data["disk_rows"]
    disk_r_hat = data["disk_r_hat"]
    phi = data["phi"]
    valid = data["valid"]
    local_disk = data["local_disk"]
    disk_weights = data["disk_weights"]
    weight_sqrt = data["weight_sqrt"]
    theta_scalar = data["theta_scalar"]
    matching_mode = _resolve_matching_mode(global_params)

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

    if matching_mode == "ring_average_radial_v1":
        out_num = 0.0
        out_den = 0.0
        in_num = 0.0
        in_den = 0.0
        out_updates: list[tuple[int, np.ndarray, float]] = []
        in_updates: list[tuple[int, np.ndarray, float]] = []

        for i, ok in enumerate(valid):
            if not ok:
                continue
            target = _tilt_target_rows_weights_and_direction(
                data=data,
                positions=positions,
                normals=normals,
                i=i,
                matching_mode=matching_mode,
            )
            if target is None:
                continue
            target_rows, target_weights, r_dir = target

            coeff = float(weight_sqrt[i])
            target_out = float(phi[i])
            row_out = [int(row) for row in target_rows]
            fixed_out = [
                bool(
                    getattr(
                        mesh.vertices[int(mesh.vertex_ids[idx])],
                        "tilt_fixed_out",
                        False,
                    )
                )
                for idx in row_out
            ]
            if not any(fixed_out):
                t_out_rad = 0.0
                denom_out = 0.0
                for idx, w in zip(row_out, target_weights):
                    t_out_rad += float(w) * float(np.dot(tilts_out[idx], r_dir))
                    denom_out += float(w) * float(w)
                out_num += coeff * (target_out - t_out_rad)
                out_den += coeff
                out_updates.append(
                    (row_out, [float(w) for w in target_weights], r_dir, denom_out)
                )

            if theta_disk is None:
                continue
            if theta_scalar is not None:
                target_in = float(theta_disk - phi[i])
            elif local_disk:
                target_in = float(theta_disk[i] - phi[i])
            else:
                target_in = float(theta_disk - phi[i])

            row_in = [int(row) for row in target_rows]
            fixed_in = [
                bool(
                    getattr(
                        mesh.vertices[int(mesh.vertex_ids[idx])], "tilt_fixed_in", False
                    )
                )
                for idx in row_in
            ]
            if not any(fixed_in):
                t_in_rad = 0.0
                denom_in = 0.0
                for idx, w in zip(row_in, target_weights):
                    t_in_rad += float(w) * float(np.dot(tilts_in[idx], r_dir))
                    denom_in += float(w) * float(w)
                in_num += coeff * (target_in - t_in_rad)
                in_den += coeff
                in_updates.append(
                    (row_in, [float(w) for w in target_weights], r_dir, denom_in)
                )

        if out_den > 0.0:
            delta_out = float(out_num / out_den)
            for rows, weights, r_dir, denom in out_updates:
                if denom <= 1.0e-12:
                    continue
                for row, weight in zip(rows, weights):
                    tilts_out[row] = (
                        tilts_out[row] + (delta_out * weight / denom) * r_dir
                    )
        if in_den > 0.0:
            delta_in = float(in_num / in_den)
            for rows, weights, r_dir, denom in in_updates:
                if denom <= 1.0e-12:
                    continue
                for row, weight in zip(rows, weights):
                    tilts_in[row] = tilts_in[row] + (delta_in * weight / denom) * r_dir

        mesh.set_tilts_in_from_array(tilts_in)
        mesh.set_tilts_out_from_array(tilts_out)
        return

    for i, ok in enumerate(valid):
        if not ok:
            continue
        target = _tilt_target_rows_weights_and_direction(
            data=data,
            positions=positions,
            normals=normals,
            i=i,
            matching_mode=matching_mode,
        )
        if target is None:
            continue
        target_rows, target_weights, r_dir = target

        # Constraint 1: t_out . r = phi
        target_out = phi[i]
        fixed_out = [
            bool(
                getattr(
                    mesh.vertices[int(mesh.vertex_ids[int(row)])],
                    "tilt_fixed_out",
                    False,
                )
            )
            for row in target_rows
        ]
        if not any(fixed_out):
            t_out_rad = 0.0
            denom_out = 0.0
            for row, weight in zip(target_rows, target_weights):
                t_out_rad += float(weight) * float(np.dot(tilts_out[int(row)], r_dir))
                denom_out += float(weight) * float(weight)
            if denom_out > 1.0e-12:
                delta_out = float(target_out - t_out_rad)
                for row, weight in zip(target_rows, target_weights):
                    idx = int(row)
                    tilts_out[idx] = (
                        tilts_out[idx] + (delta_out * float(weight) / denom_out) * r_dir
                    )

        if theta_disk is None:
            continue

        # Constraint 2: t_in . r = theta_B - phi
        if theta_scalar is not None:
            target_in = float(theta_disk - phi[i])
        elif local_disk:
            target_in = float(theta_disk[i] - phi[i])
        else:
            target_in = float(theta_disk - phi[i])

        fixed_in = [
            bool(
                getattr(
                    mesh.vertices[int(mesh.vertex_ids[int(row)])],
                    "tilt_fixed_in",
                    False,
                )
            )
            for row in target_rows
        ]
        if not any(fixed_in):
            t_in_rad = 0.0
            denom_in = 0.0
            for row, weight in zip(target_rows, target_weights):
                t_in_rad += float(weight) * float(np.dot(tilts_in[int(row)], r_dir))
                denom_in += float(weight) * float(weight)
            if denom_in > 1.0e-12:
                delta_in = float(target_in - t_in_rad)
                for row, weight in zip(target_rows, target_weights):
                    idx = int(row)
                    tilts_in[idx] = (
                        tilts_in[idx] + (delta_in * float(weight) / denom_in) * r_dir
                    )

    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)


__all__ = [
    "coarse_rim_family_diagnostics",
    "constraint_gradients_array",
    "constraint_gradients_rows_array",
    "constraint_gradients_tilt_array",
    "constraint_gradients_tilt_rows_array",
    "enforce_tilt_constraint",
    "matching_residual_diagnostics",
    "matching_ring_diagnostics",
]
