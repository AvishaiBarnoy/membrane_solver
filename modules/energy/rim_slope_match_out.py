"""Rim matching energy: enforce outer-leaflet tilt to match outer slope.

This module penalizes mismatch between the outer-leaflet radial tilt at a rim
and the small-slope geometric slope of the outer membrane just outside that rim.
It is intended to approximate the kinematic matching condition in
`docs/tex/1_disk_3d.pdf` (γ=0), where the proximal leaflet tilt equals the
outer membrane slope at the disk boundary.

If an optional disk-group is supplied, the module also enforces the complementary
relation on the distal leaflet:

    (t_in · r_hat) ≈ θ_disk - φ

where θ_disk is the radial tilt on the disk ring just inside the rim. When the
disk ring has the same vertex count as the rim, the module pairs vertices by
angle and enforces this condition per-vertex; otherwise it falls back to the
arc-length weighted mean tilt.

Energy (discrete, small-slope approximation)
--------------------------------------------
For each paired rim/outer vertex i:

    E = 1/2 * k * Σ_i w_i ( (t_out · r_hat)_i - φ_i )^2
      + 1/2 * k * Σ_i w_i ( (t_in · r_hat)_i - (θ_disk - φ_i) )^2

where
    φ_i = (h_out - h_rim) / (r_out - r_rim),
    h = (pos - center) · normal,
    r_hat is the in-plane radial unit vector,
    w_i is a local arc-length weight on the rim ring.

Notes
-----
- This is a *small-slope* coupling and only includes shape gradients along the
  plane normal (ignores dependence on radial distances).
- The rim and outer rings are paired by angle about `rim_slope_match_center`.
- The optional disk-group coupling uses an arc-length weighted *mean* radial
  tilt on the disk ring; it is intended for axisymmetric setups where the disk
  group is the ring just inside the rim. When the disk and rim rings match in
  vertex count, per-vertex pairing is used.

Parameters
----------
- `rim_slope_match_group`: rim vertex group name (string; required).
- `rim_slope_match_outer_group`: outer ring group name (string; required).
- `rim_slope_match_disk_group`: disk ring group name just inside the rim
  (string; optional).
- `rim_slope_match_strength`: k (float; default 0).
- `rim_slope_match_center`: 3D center point (default [0,0,0]).
- `rim_slope_match_normal`: plane normal (default [0,0,1] if fit fails).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh

logger = logging.getLogger("membrane_solver")
_WARNED_DISK_EQUALS_RIM = False

USES_TILT_LEAFLETS = True


def _resolve_group(param_resolver, key: str) -> str | None:
    raw = param_resolver.get(None, key)
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


def _resolve_strength(param_resolver) -> float:
    val = param_resolver.get(None, "rim_slope_match_strength")
    return float(val or 0.0)


def _resolve_center(param_resolver) -> np.ndarray:
    center = param_resolver.get(None, "rim_slope_match_center")
    if center is None:
        center = [0.0, 0.0, 0.0]
    return np.asarray(center, dtype=float).reshape(3)


def _resolve_normal(param_resolver) -> np.ndarray | None:
    raw = param_resolver.get(None, "rim_slope_match_normal")
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-15:
        return None
    return arr / norm


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
        v = mesh.vertices[int(vid)]
        opts = getattr(v, "options", None) or {}
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


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver, *, compute_gradient: bool = True
) -> (
    Tuple[float, Dict[int, np.ndarray]]
    | Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Dict-based wrapper returning (E, shape_grad[, tilt_in_grad, tilt_out_grad])."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_in_grad_arr = np.zeros_like(positions) if compute_gradient else None
    tilt_out_grad_arr = np.zeros_like(positions) if compute_gradient else None
    energy = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts_in=None,
        tilts_out=None,
        tilt_in_grad_arr=tilt_in_grad_arr,
        tilt_out_grad_arr=tilt_out_grad_arr,
    )
    if not compute_gradient:
        return float(energy), {}

    shape_grad = {
        int(vid): grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }
    tilt_out_grad = {
        int(vid): tilt_out_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if tilt_out_grad_arr is not None and np.any(tilt_out_grad_arr[row])
    }
    tilt_in_grad = (
        {
            int(vid): tilt_in_grad_arr[row].copy()
            for row, vid in enumerate(mesh.vertex_ids)
            if tilt_in_grad_arr is not None and np.any(tilt_in_grad_arr[row])
        }
        if tilt_in_grad_arr is not None
        else {}
    )
    return float(energy), shape_grad, tilt_in_grad, tilt_out_grad


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray | None,
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
    tilt_in_grad_arr: np.ndarray | None = None,
    tilt_out_grad_arr: np.ndarray | None = None,
) -> float:
    """Dense-array rim slope matching accumulation."""
    _ = global_params, index_map, tilts_in, tilt_in_grad_arr
    group = _resolve_group(param_resolver, "rim_slope_match_group")
    outer_group = _resolve_group(param_resolver, "rim_slope_match_outer_group")
    disk_group = _resolve_group(param_resolver, "rim_slope_match_disk_group")
    disk_group = _sanitize_disk_group(rim_group=group, disk_group=disk_group)
    if group is None or outer_group is None:
        return 0.0

    k_match = _resolve_strength(param_resolver)
    if k_match == 0.0:
        return 0.0

    rim_rows = _collect_group_rows(mesh, group)
    outer_rows = _collect_group_rows(mesh, outer_group)
    if rim_rows.size == 0 or outer_rows.size == 0:
        return 0.0
    if rim_rows.size != outer_rows.size:
        return 0.0

    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()
    else:
        tilts_out = np.asarray(tilts_out, dtype=float)
        if tilts_out.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")
    if disk_group is not None:
        if tilts_in is None:
            tilts_in = mesh.tilts_in_view()
        else:
            tilts_in = np.asarray(tilts_in, dtype=float)
            if tilts_in.shape != (len(mesh.vertex_ids), 3):
                raise ValueError("tilts_in must have shape (N_vertices, 3)")

    center = _resolve_center(param_resolver)
    normal = _resolve_normal(param_resolver)
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

    # Radial direction (in-plane).
    r_vec = rim_pos - center[None, :]
    r_vec = r_vec - np.einsum("ij,j->i", r_vec, normal)[:, None] * normal[None, :]
    r_len = np.linalg.norm(r_vec, axis=1)
    good = r_len > 1e-12
    if not np.any(good):
        return 0.0

    r_hat = np.zeros_like(r_vec)
    r_hat[good] = r_vec[good] / r_len[good][:, None]

    # Small-slope geometric slope along the normal.
    h_rim = np.einsum("ij,j->i", rim_pos - center[None, :], normal)
    h_out = np.einsum("ij,j->i", outer_pos - center[None, :], normal)
    r_out_vec = outer_pos - center[None, :]
    r_out_vec = (
        r_out_vec - np.einsum("ij,j->i", r_out_vec, normal)[:, None] * normal[None, :]
    )
    r_out = np.linalg.norm(r_out_vec, axis=1)
    dr = r_out - r_len
    dr_safe = np.where(np.abs(dr) > 1e-8, dr, np.nan)

    phi = (h_out - h_rim) / dr_safe
    valid = np.isfinite(phi) & good
    if not np.any(valid):
        return 0.0

    t_out = tilts_out[rim_rows]
    tilt_radial = np.einsum("ij,ij->i", t_out, r_hat)

    weights = _arc_length_weights(rim_pos, np.arange(len(rim_rows)))
    weights = np.where(valid, weights, 0.0)

    diff = np.zeros_like(tilt_radial)
    diff[valid] = tilt_radial[valid] - phi[valid]
    diff_in = np.zeros_like(diff)

    theta_disk = None
    disk_rows = None
    disk_weights = None
    disk_r_hat = None
    local_disk = False
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
                theta_disk = np.einsum(
                    "ij,ij->i", tilts_in[disk_rows], disk_r_hat
                ).astype(float, copy=False)
            else:
                disk_weights = _arc_length_weights(disk_pos, np.arange(len(disk_rows)))
                disk_weights = np.where(good_disk, disk_weights, 0.0)
                weight_sum = float(np.sum(disk_weights))
                if weight_sum > 0.0:
                    theta_disk = float(
                        np.sum(
                            disk_weights
                            * np.einsum("ij,ij->i", tilts_in[disk_rows], disk_r_hat)
                        )
                        / weight_sum
                    )

    if theta_disk is not None:
        tilt_in_rim = np.einsum("ij,ij->i", tilts_in[rim_rows], r_hat)
        if local_disk:
            diff_in[valid] = tilt_in_rim[valid] - (theta_disk[valid] - phi[valid])
        else:
            diff_in[valid] = tilt_in_rim[valid] - (theta_disk - phi[valid])

    energy = float(0.5 * k_match * np.sum(weights * diff * diff))
    if theta_disk is not None:
        energy += float(0.5 * k_match * np.sum(weights * diff_in * diff_in))

    if tilt_out_grad_arr is not None:
        tilt_out_grad_arr = np.asarray(tilt_out_grad_arr, dtype=float)
        if tilt_out_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_out_grad_arr must have shape (N_vertices, 3)")
        contrib = (k_match * weights * diff)[:, None] * r_hat
        np.add.at(tilt_out_grad_arr, rim_rows, contrib)

    if theta_disk is not None and tilt_in_grad_arr is not None:
        tilt_in_grad_arr = np.asarray(tilt_in_grad_arr, dtype=float)
        if tilt_in_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_in_grad_arr must have shape (N_vertices, 3)")
        contrib_in = (k_match * weights * diff_in)[:, None] * r_hat
        np.add.at(tilt_in_grad_arr, rim_rows, contrib_in)
        if (
            disk_rows is not None
            and disk_rows.size
            and disk_r_hat is not None
            and local_disk
        ):
            contrib_disk = (-k_match * weights * diff_in)[:, None] * disk_r_hat
            np.add.at(tilt_in_grad_arr, disk_rows, contrib_disk)
        elif (
            disk_rows is not None
            and disk_rows.size
            and disk_weights is not None
            and disk_r_hat is not None
        ):
            weight_sum = float(np.sum(disk_weights))
            if weight_sum > 0.0:
                coeff_disk = -k_match * float(np.sum(weights * diff_in))
                disk_factor = (disk_weights / weight_sum)[:, None] * disk_r_hat
                np.add.at(tilt_in_grad_arr, disk_rows, coeff_disk * disk_factor)

    # Shape gradient: only along the normal (small-slope approximation).
    #
    # During tilt-only relaxation, callers may pass grad_arr=None to skip
    # expensive (and unused) shape-gradient work.
    if grad_arr is not None:
        grad_coeff = k_match * weights * (diff - diff_in)
        grad_rim = np.zeros_like(rim_pos)
        grad_out = np.zeros_like(outer_pos)
        inv_dr = np.zeros_like(dr)
        inv_dr[valid] = 1.0 / dr[valid]
        grad_rim += (grad_coeff * inv_dr)[:, None] * normal[None, :]
        grad_out += (-grad_coeff * inv_dr)[:, None] * normal[None, :]
        np.add.at(grad_arr, rim_rows, grad_rim)
        np.add.at(grad_arr, outer_rows, grad_out)

    return energy


__all__ = ["compute_energy_and_gradient", "compute_energy_and_gradient_array"]
