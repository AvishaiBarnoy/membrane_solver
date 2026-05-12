"""Geometric and row collection helpers for hard rim-matching constraints."""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh
from geometry.plane_ops import (
    fit_plane_normal,
    order_by_angle,
    orthonormal_basis_from_normal,
)
from modules.constraints.rim_slope_match_params import _uses_outer_shell_tilt_matching


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
    if _uses_outer_shell_tilt_matching(matching_mode):
        tilt_rows = np.asarray(data.get("tilt_rows", data["outer_rows"]), dtype=int)
        tilt_idx0 = np.asarray(data.get("tilt_idx0", data["outer_idx0"]), dtype=int)
        tilt_idx1 = np.asarray(data.get("tilt_idx1", data["outer_idx1"]), dtype=int)
        tilt_w0 = np.asarray(data.get("tilt_w0", data["outer_w0"]), dtype=float)
        tilt_w1 = np.asarray(data.get("tilt_w1", data["outer_w1"]), dtype=float)
        row0 = int(tilt_rows[tilt_idx0[i]])
        row1 = int(tilt_rows[tilt_idx1[i]])
        w0 = float(tilt_w0[i])
        w1 = float(tilt_w1[i])
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


def _disk_theta_rows_weights_and_direction(
    *,
    data: dict,
    i: int,
    theta_scalar_active: bool,
) -> tuple[list[int], list[float], list[np.ndarray]] | None:
    """Return disk-side rows and directions for the inner scalar-theta law."""
    disk_rows = data.get("disk_rows")
    disk_r_hat = data.get("disk_r_hat")
    if disk_rows is None or disk_r_hat is None:
        return None

    if theta_scalar_active:
        if data["local_disk"]:
            return [int(disk_rows[i])], [1.0], [np.asarray(disk_r_hat[i], dtype=float)]
        disk_weights = data.get("disk_weights")
        if disk_weights is None:
            return None
        disk_weights = np.asarray(disk_weights, dtype=float)
        weight_sum = float(np.sum(disk_weights))
        if weight_sum <= 0.0:
            return None
        rows = [int(row_idx) for row_idx in disk_rows]
        weights = [float(w / weight_sum) for w in disk_weights]
        dirs = [np.asarray(vec, dtype=float) for vec in disk_r_hat]
        return rows, weights, dirs

    rows: list[int] = []
    weights: list[float] = []
    dirs: list[np.ndarray] = []
    if data["local_disk"]:
        rows.append(int(disk_rows[i]))
        weights.append(1.0)
        dirs.append(np.asarray(disk_r_hat[i], dtype=float))
    else:
        disk_weights = data.get("disk_weights")
        if disk_weights is None:
            return None
        disk_weights = np.asarray(disk_weights, dtype=float)
        weight_sum = float(np.sum(disk_weights))
        if weight_sum <= 0.0:
            return None
        for row_idx, factor in zip(
            disk_rows, (disk_weights / weight_sum)[:, None] * disk_r_hat
        ):
            rows.append(int(row_idx))
            weights.append(1.0)
            dirs.append(np.asarray(factor, dtype=float))
    return rows, weights, dirs


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
    return fit_plane_normal(points)


def _orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return orthonormal_basis_from_normal(normal)


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
    return order_by_angle(positions, center=center, normal=normal)


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
