"""Payload preparation for hard rim-matching constraints."""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh
from modules.constraints.local_interface_shells import build_local_interface_shell_data
from modules.constraints.rim_slope_match_params import (
    _resolve_center,
    _resolve_group,
    _resolve_matching_mode,
    _sanitize_disk_group,
    _use_curved_free_disk_shell2_tilt_continuation,
    _uses_scaffold_trace_lane,
)
from modules.constraints.rim_slope_match_utils import (
    _arc_length_params,
    _arc_length_weights,
    _collect_group_rows,
    _fit_plane_normal,
    _interp_ring_positions,
    _order_by_angle,
    _resolve_normal,
)


def _build_matching_data(mesh: Mesh, global_params, positions: np.ndarray):
    matching_mode = _resolve_matching_mode(global_params)
    group = _resolve_group(global_params, "rim_slope_match_group")
    outer_group = _resolve_group(global_params, "rim_slope_match_outer_group")
    disk_group = _resolve_group(global_params, "rim_slope_match_disk_group")
    if matching_mode != "physical_edge_staggered_v1":
        disk_group = _sanitize_disk_group(rim_group=group, disk_group=disk_group)
    theta_param = None
    if global_params is not None:
        theta_param = global_params.get("rim_slope_match_thetaB_param")
    if matching_mode == "physical_edge_staggered_v1":
        interface_group = disk_group or group
        if interface_group is None:
            return None
    elif group is None or outer_group is None:
        return None
    center = _resolve_center(global_params)
    raw_normal = _resolve_normal(global_params)
    normal_token = (
        None
        if raw_normal is None
        else tuple(np.asarray(raw_normal, dtype=float).reshape(3).tolist())
    )
    theta_token = None
    if (
        theta_param is not None
        and global_params is not None
        and not _uses_scaffold_trace_lane(global_params, matching_mode)
    ):
        theta_token = float(global_params.get(str(theta_param)) or 0.0)
    trace_layer_radius = (
        None
        if global_params is None
        else global_params.get("parity_trace_layer_radius")
    )
    cache_key = (
        int(mesh._version),
        int(mesh._vertex_ids_version),
        str(matching_mode),
        str(group),
        str(outer_group),
        None if disk_group is None else str(disk_group),
        None if theta_param is None else str(theta_param),
        theta_token,
        None if trace_layer_radius is None else float(trace_layer_radius),
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

    normal = raw_normal
    if matching_mode == "physical_edge_staggered_v1":
        try:
            local_shells = build_local_interface_shell_data(
                mesh, positions=positions, group=str(interface_group)
            )
        except AssertionError:
            return None
        scaffold_outer_shells = (
            0
            if global_params is None
            else int(global_params.get("parity_outer_shells") or 0)
        )
        if trace_layer_radius is None:
            rim_rows = np.asarray(local_shells.disk_rows, dtype=int)
            outer_rows = np.asarray(local_shells.rim_rows_for_disk, dtype=int)
            tilt_rows = outer_rows.copy()
            target_source = "first_outer_shell"
        else:
            rim_rows = np.asarray(local_shells.disk_rows, dtype=int)
            outer_rows = np.asarray(local_shells.rim_rows_for_disk, dtype=int)
            tilt_rows = outer_rows.copy()
            target_source = (
                "scaffold_trace_shell"
                if scaffold_outer_shells > 0
                else "explicit_trace_shell"
            )
        if rim_rows.size == 0 or outer_rows.size == 0:
            return None
        if normal is None:
            normal = _fit_plane_normal(positions[rim_rows])
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
        rim_pos = positions[rim_rows]
        outer_pos = positions[outer_rows]
        tilt_pos = positions[tilt_rows]
    else:
        target_source = "tagged_outer_shell"
        rim_rows = _collect_group_rows(mesh, group)
        outer_rows = _collect_group_rows(mesh, outer_group)
        if rim_rows.size == 0 or outer_rows.size == 0:
            return None
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
        tilt_rows = outer_rows.copy()
        tilt_pos = outer_pos
        if (
            matching_mode == "shared_rim_staggered_v1"
            and _use_curved_free_disk_shell2_tilt_continuation(global_params)
        ):
            try:
                local_shells = build_local_interface_shell_data(
                    mesh, positions=positions, group=str(group)
                )
            except AssertionError:
                local_shells = None
            if local_shells is not None:
                local_rim_rows = np.asarray(local_shells.rim_rows, dtype=int)
                local_outer_for_rim = np.asarray(
                    local_shells.outer_rows_for_rim, dtype=int
                )
                row_to_target = {
                    int(row): int(target)
                    for row, target in zip(local_rim_rows, local_outer_for_rim)
                }
                matched_outer = np.asarray(
                    [row_to_target.get(int(row), -1) for row in outer_rows],
                    dtype=int,
                )
                if matched_outer.size == outer_rows.size and np.all(matched_outer >= 0):
                    tilt_rows = matched_outer
                    tilt_pos = positions[tilt_rows]
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
    tilt_idx0 = np.arange(len(rim_rows), dtype=int)
    tilt_idx1 = np.arange(len(rim_rows), dtype=int)
    tilt_w0 = np.ones(len(rim_rows), dtype=float)
    tilt_w1 = np.zeros(len(rim_rows), dtype=float)
    if rim_rows.size != tilt_rows.size:
        s_rim, total_rim = _arc_length_params(rim_pos)
        if total_rim <= 0.0:
            return None
        interp_tilt = _interp_ring_positions(tilt_pos, s_rim)
        if interp_tilt is None:
            return None
        _tilt_pos_interp, tilt_idx0, tilt_idx1, tilt_w0, tilt_w1 = interp_tilt

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
    if matching_mode == "physical_edge_staggered_v1":
        disk_rows = rim_rows.copy()
        disk_r_hat = r_hat.copy()
        disk_weights = None
        local_disk = True
    elif disk_group is not None:
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
        "tilt_rows": tilt_rows,
        "tilt_idx0": tilt_idx0,
        "tilt_idx1": tilt_idx1,
        "tilt_w0": tilt_w0,
        "tilt_w1": tilt_w1,
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
        "matching_mode": matching_mode,
        "construction_mode": (
            "physical_edge_local_shell"
            if matching_mode == "physical_edge_staggered_v1"
            else "legacy_tagged_rim_shell"
        ),
        "target_source": target_source,
    }
    if mesh._geometry_cache_active(positions):
        cached = getattr(mesh, cache_attr, None)
        entries = cached.get("entries", {}) if isinstance(cached, dict) else {}
        entries[cache_key] = result
        if len(entries) > 16:
            entries = dict(list(entries.items())[-8:])
        setattr(mesh, cache_attr, {"entries": entries})
    return result
