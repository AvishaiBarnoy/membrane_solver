"""Hard rim matching constraint for outer-leaflet tilt vs slope.

This constraint enforces the (gamma=0) rim-matching conditions described in
`docs/tex/1_disk_3d.tex` using Lagrange-style gradient projection plus an
optional tilt-only projection hook:

    t_out · r_hat = phi
    t_in  · r_hat = theta_disk - phi

where
    phi = (h_out - h_rim) / (r_out - r_rim)
"""

from __future__ import annotations

import logging

import numpy as np

from geometry.entities import Mesh
from modules.constraints.rim_slope_match_diagnostics import (
    coarse_rim_family_diagnostics,
    matching_residual_diagnostics,
    matching_ring_diagnostics,
)
from modules.constraints.rim_slope_match_gradients import (
    constraint_gradients_array,
    constraint_gradients_joint_array,
    constraint_gradients_rows_array,
    constraint_gradients_tilt_array,
    constraint_gradients_tilt_rows_array,
)
from modules.constraints.rim_slope_match_params import (
    _resolve_center,
    _resolve_group,
    _resolve_matching_mode,
    _sanitize_disk_group,
    _use_curved_free_disk_shell2_tilt_continuation,
    _use_disk_theta_targeting,
    _uses_outer_shell_tilt_matching,
    _uses_scaffold_trace_lane,
)
from modules.constraints.rim_slope_match_payload import _build_matching_data
from modules.constraints.rim_slope_match_utils import (
    _arc_length_params,
    _arc_length_weights,
    _collect_group_rows,
    _disk_theta_rows_weights_and_direction,
    _fit_plane_normal,
    _interp_ring_positions,
    _order_by_angle,
    _orthonormal_basis,
    _resolve_normal,
    _tilt_target_rows_weights_and_direction,
)

logger = logging.getLogger("membrane_solver")


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

        if theta_scalar is not None and _use_disk_theta_targeting(
            global_params, matching_mode
        ):
            disk_target = _disk_theta_rows_weights_and_direction(
                data=data,
                i=i,
                theta_scalar_active=True,
            )
            if disk_target is None:
                continue
            rows_in, weights_in, dirs_in = disk_target
        else:
            rows_in = [int(row) for row in target_rows]
            weights_in = [float(w) for w in target_weights]
            dirs_in = [np.asarray(r_dir, dtype=float) for _ in target_rows]

        fixed_in = [
            bool(
                getattr(
                    mesh.vertices[int(mesh.vertex_ids[int(row)])],
                    "tilt_fixed_in",
                    False,
                )
            )
            for row in rows_in
        ]
        if not any(fixed_in):
            t_in_rad = 0.0
            denom_in = 0.0
            for row, weight, vec in zip(rows_in, weights_in, dirs_in):
                t_in_rad += float(weight) * float(np.dot(tilts_in[int(row)], vec))
                denom_in += float(weight) * float(weight)
            if denom_in > 1.0e-12:
                delta_in = float(target_in - t_in_rad)
                for row, weight, vec in zip(rows_in, weights_in, dirs_in):
                    idx = int(row)
                    tilts_in[idx] = (
                        tilts_in[idx] + (delta_in * float(weight) / denom_in) * vec
                    )

    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)


def enforce_constraint(mesh: Mesh, global_params=None, **_kwargs) -> None:
    """Project interface-shell heights onto the current rim law."""
    matching_mode = _resolve_matching_mode(global_params)
    context = str(_kwargs.get("context") or "").strip().lower()
    if matching_mode not in {
        "physical_edge_staggered_v1",
        "shared_rim_staggered_v1",
    }:
        return
    if matching_mode == "physical_edge_staggered_v1":
        scaffold_outer_shells = (
            0
            if global_params is None
            else int(global_params.get("parity_outer_shells") or 0)
        )
        trace_layer_radius = (
            None
            if global_params is None
            else global_params.get("parity_trace_layer_radius")
        )
        if scaffold_outer_shells <= 0 or trace_layer_radius is None:
            return
    elif not _use_curved_free_disk_shell2_tilt_continuation(global_params):
        return

    positions = mesh.positions_view()
    data = _build_matching_data(mesh, global_params, positions)
    if data is None:
        return

    disk_rows = data["disk_rows"]
    disk_r_hat = data["disk_r_hat"]
    valid = np.asarray(data["valid"], dtype=bool)
    local_disk = bool(data["local_disk"])
    disk_weights = data["disk_weights"]
    normal = np.asarray(data["normal"], dtype=float)
    outer_rows = np.asarray(data["outer_rows"], dtype=int)
    outer_idx0 = np.asarray(data["outer_idx0"], dtype=int)
    outer_idx1 = np.asarray(data["outer_idx1"], dtype=int)
    outer_w0 = np.asarray(data["outer_w0"], dtype=float)
    outer_w1 = np.asarray(data["outer_w1"], dtype=float)
    rim_rows = np.asarray(data["rim_rows"], dtype=int)
    inv_dr = np.asarray(data["inv_dr"], dtype=float)
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

    height_num = np.zeros(len(mesh.vertex_ids), dtype=float)
    height_den = np.zeros(len(mesh.vertex_ids), dtype=float)
    update_tilt_out = not (
        matching_mode == "shared_rim_staggered_v1" and context == "minimize"
    )
    tilt_num = np.zeros(len(mesh.vertex_ids), dtype=float)
    tilt_den = np.zeros(len(mesh.vertex_ids), dtype=float)

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
        t_out_rad = 0.0
        for row, weight in zip(target_rows, target_weights):
            t_out_rad += float(weight) * float(np.dot(tilts_out[int(row)], r_dir))

        if abs(float(inv_dr[i])) <= 1.0e-12:
            continue
        dr = 1.0 / float(inv_dr[i])
        current_rim_height = float(np.dot(positions[int(rim_rows[i])], normal))
        row0 = int(outer_rows[outer_idx0[i]])
        row1 = int(outer_rows[outer_idx1[i]])
        w0 = float(outer_w0[i])
        w1 = float(outer_w1[i])
        current_outer_height = 0.0
        height_weight = 0.0
        if abs(w0) > 1.0e-12:
            current_outer_height += w0 * float(np.dot(positions[row0], normal))
            height_weight += abs(w0)
        if abs(w1) > 1.0e-12:
            current_outer_height += w1 * float(np.dot(positions[row1], normal))
            height_weight += abs(w1)
        if height_weight <= 1.0e-12:
            continue
        current_outer_height /= height_weight
        phi_current = (current_outer_height - current_rim_height) / dr

        if matching_mode == "shared_rim_staggered_v1" and theta_scalar is not None:
            phi_target = 0.5 * float(theta_scalar)
            t_out_target = phi_target
        elif theta_disk is None:
            # Joint local proximal solve with equal weights on:
            # - staying near the current shell secant phi_current
            # - staying near the current outer radial tilt t_out_rad
            # - satisfying the existing outer-side relation t_out = phi
            phi_target = (2.0 * phi_current + float(t_out_rad)) / 3.0
            t_out_target = 0.5 * (phi_target + float(t_out_rad))
        else:
            t_in_rad = 0.0
            for row, weight in zip(target_rows, target_weights):
                t_in_rad += float(weight) * float(np.dot(tilts_in[int(row)], r_dir))
            theta_val = (
                float(theta_disk)
                if theta_scalar is not None or not local_disk
                else float(theta_disk[i])
            )
            continuity_target = theta_val - float(t_in_rad)
            # Joint local proximal solve with equal weights on:
            # - staying near the current shell secant phi_current
            # - staying near the current outer radial tilt t_out_rad
            # - satisfying t_out = phi
            # - satisfying t_in = theta_disk - phi
            phi_target = (
                2.0 * phi_current + float(t_out_rad) + 2.0 * continuity_target
            ) / 5.0
            t_out_target = 0.5 * (phi_target + float(t_out_rad))

        target_height = current_rim_height + phi_target * dr
        if abs(w0) > 1.0e-12:
            height_num[row0] += w0 * target_height
            height_den[row0] += abs(w0)
            if update_tilt_out:
                tilt_num[row0] += w0 * t_out_target
                tilt_den[row0] += abs(w0)
        if abs(w1) > 1.0e-12:
            height_num[row1] += w1 * target_height
            height_den[row1] += abs(w1)
            if update_tilt_out:
                tilt_num[row1] += w1 * t_out_target
                tilt_den[row1] += abs(w1)

    moved = False
    for row in np.flatnonzero(height_den > 1.0e-12):
        vid = int(mesh.vertex_ids[int(row)])
        if getattr(mesh.vertices[vid], "fixed", False):
            continue
        current_pos = np.asarray(mesh.vertices[vid].position, dtype=float)
        current_height = float(np.dot(current_pos, normal))
        target_height = float(height_num[int(row)] / height_den[int(row)])
        mesh.vertices[vid].position[:] = current_pos + (
            (target_height - current_height) * normal
        )
        moved = True

    for row in np.flatnonzero(tilt_den > 1.0e-12):
        vid = int(mesh.vertex_ids[int(row)])
        if getattr(mesh.vertices[vid], "tilt_fixed_out", False):
            continue
        normal_row = np.asarray(normals[int(row)], dtype=float)
        pos = np.asarray(mesh.vertices[vid].position, dtype=float)
        radius = float(np.linalg.norm(pos[:2]))
        if radius <= 1.0e-12:
            continue
        r_hat_row = np.array([pos[0] / radius, pos[1] / radius, 0.0], dtype=float)
        r_dir = r_hat_row - float(np.dot(r_hat_row, normal_row)) * normal_row
        r_norm = float(np.linalg.norm(r_dir))
        if r_norm <= 1.0e-12:
            continue
        r_dir = r_dir / r_norm
        current = np.asarray(mesh.vertices[vid].tilt_out, dtype=float)
        radial = float(np.dot(current, r_dir))
        target_tilt = float(tilt_num[int(row)] / tilt_den[int(row)])
        tangential = current - radial * r_dir
        mesh.vertices[vid].tilt_out = tangential + target_tilt * r_dir

    if moved:
        mesh.increment_version()
    if update_tilt_out:
        mesh.touch_tilts_out()


__all__ = [
    "_arc_length_params",
    "_arc_length_weights",
    "_build_matching_data",
    "_collect_group_rows",
    "_disk_theta_rows_weights_and_direction",
    "_fit_plane_normal",
    "_interp_ring_positions",
    "_order_by_angle",
    "_orthonormal_basis",
    "_resolve_center",
    "_resolve_group",
    "_resolve_matching_mode",
    "_resolve_normal",
    "_sanitize_disk_group",
    "_tilt_target_rows_weights_and_direction",
    "_use_curved_free_disk_shell2_tilt_continuation",
    "_use_disk_theta_targeting",
    "_uses_outer_shell_tilt_matching",
    "_uses_scaffold_trace_lane",
    "coarse_rim_family_diagnostics",
    "constraint_gradients_array",
    "constraint_gradients_joint_array",
    "constraint_gradients_rows_array",
    "constraint_gradients_tilt_array",
    "constraint_gradients_tilt_rows_array",
    "enforce_constraint",
    "enforce_tilt_constraint",
    "matching_residual_diagnostics",
    "matching_ring_diagnostics",
]
