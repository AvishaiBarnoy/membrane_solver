"""Gradient assembly for hard rim-matching constraints."""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh
from modules.constraints.rim_slope_match_params import (
    _resolve_matching_mode,
    _use_disk_theta_targeting,
)
from modules.constraints.rim_slope_match_payload import _build_matching_data
from modules.constraints.rim_slope_match_utils import (
    _disk_theta_rows_weights_and_direction,
    _tilt_target_rows_weights_and_direction,
)


def constraint_gradients_joint_array(
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
            tuple[np.ndarray, np.ndarray] | None,  # shape
            tuple[np.ndarray, np.ndarray] | None,  # tilt_in
            tuple[np.ndarray, np.ndarray] | None,  # tilt_out
        ]
    ]
    | None
):
    """Return paired shape/tilt gradients for joint KKT projection."""
    _ = tilts_in, tilts_out
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
    disk_r_hat = data["disk_r_hat"]
    weight_sqrt = data["weight_sqrt"]
    inv_dr = data["inv_dr"]
    valid = data["valid"]
    normal = data["normal"]
    matching_mode = _resolve_matching_mode(global_params)
    theta_scalar = data["theta_scalar"]

    normals = mesh.vertex_normals(positions=positions)

    if matching_mode != "physical_edge_staggered_v1" and not _use_disk_theta_targeting(
        global_params, matching_mode
    ):
        constraints: list[
            tuple[
                tuple[np.ndarray, np.ndarray] | None,
                tuple[np.ndarray, np.ndarray] | None,
                tuple[np.ndarray, np.ndarray] | None,
            ]
        ] = []
        agg_shape_outer_rows: list[int] = []
        agg_shape_outer_vecs: list[np.ndarray] = []
        agg_out_rows: list[int] = []
        agg_out_vecs: list[np.ndarray] = []
        agg_shape_inner_rows: list[int] = []
        agg_shape_inner_vecs: list[np.ndarray] = []
        agg_in_rows: list[int] = []
        agg_in_vecs: list[np.ndarray] = []

        for i, ok in enumerate(valid):
            if not ok or weight_sqrt[i] == 0.0:
                continue
            coeff_shape = weight_sqrt[i] * inv_dr[i]
            coeff_tilt = weight_sqrt[i]

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

            out0 = outer_rows[outer_idx0[i]]
            out1 = outer_rows[outer_idx1[i]]

            rows_shape = [int(rim_rows[i]), int(out0)]
            vecs_shape_outer = [
                coeff_shape * normal,
                -coeff_shape * outer_w0[i] * normal,
            ]
            vecs_shape_inner = [
                -coeff_shape * normal,
                coeff_shape * outer_w0[i] * normal,
            ]
            if out1 != out0 or outer_w1[i] != 0.0:
                rows_shape.append(int(out1))
                vecs_shape_outer.append(-coeff_shape * outer_w1[i] * normal)
                vecs_shape_inner.append(coeff_shape * outer_w1[i] * normal)

            rows_tilt = [int(row) for row in target_rows]
            vecs_tilt = [coeff_tilt * float(w) * r_dir for w in target_weights]
            out_part = (
                np.asarray(rows_tilt, dtype=int),
                np.asarray(vecs_tilt, dtype=float),
            )
            shape_part_outer = (
                np.asarray(rows_shape, dtype=int),
                np.asarray(vecs_shape_outer, dtype=float),
            )
            if matching_mode == "ring_average_radial_v1":
                agg_shape_outer_rows.extend(shape_part_outer[0].tolist())
                agg_shape_outer_vecs.extend(shape_part_outer[1])
                agg_out_rows.extend(out_part[0].tolist())
                agg_out_vecs.extend(out_part[1])
            else:
                constraints.append((shape_part_outer, None, out_part))

            if disk_rows is None or disk_r_hat is None:
                continue

            rows_in = [int(row) for row in target_rows]
            vecs_in = [coeff_tilt * float(w) * r_dir for w in target_weights]
            if theta_scalar is None:
                if data["local_disk"]:
                    rows_in.append(int(disk_rows[i]))
                    vecs_in.append(-coeff_tilt * disk_r_hat[i])
                elif data["disk_weights"] is not None:
                    disk_weights = np.asarray(data["disk_weights"], dtype=float)
                    weight_sum = float(np.sum(disk_weights))
                    if weight_sum > 0.0:
                        factors = (disk_weights / weight_sum)[:, None] * disk_r_hat
                        for row_idx, factor in zip(disk_rows, factors):
                            rows_in.append(int(row_idx))
                            vecs_in.append(-coeff_tilt * factor)

            shape_part_inner = (
                np.asarray(rows_shape, dtype=int),
                np.asarray(vecs_shape_inner, dtype=float),
            )
            in_part = (
                np.asarray(rows_in, dtype=int),
                np.asarray(vecs_in, dtype=float),
            )
            if matching_mode == "ring_average_radial_v1":
                agg_shape_inner_rows.extend(shape_part_inner[0].tolist())
                agg_shape_inner_vecs.extend(shape_part_inner[1])
                agg_in_rows.extend(in_part[0].tolist())
                agg_in_vecs.extend(in_part[1])
            else:
                constraints.append((shape_part_inner, in_part, None))

        if matching_mode == "ring_average_radial_v1":
            if agg_shape_outer_rows or agg_out_rows:
                constraints.append(
                    (
                        (
                            np.asarray(agg_shape_outer_rows, dtype=int),
                            np.asarray(agg_shape_outer_vecs, dtype=float),
                        ),
                        None,
                        (
                            np.asarray(agg_out_rows, dtype=int),
                            np.asarray(agg_out_vecs, dtype=float),
                        ),
                    )
                )
            if agg_shape_inner_rows or agg_in_rows:
                constraints.append(
                    (
                        (
                            np.asarray(agg_shape_inner_rows, dtype=int),
                            np.asarray(agg_shape_inner_vecs, dtype=float),
                        ),
                        (
                            np.asarray(agg_in_rows, dtype=int),
                            np.asarray(agg_in_vecs, dtype=float),
                        ),
                        None,
                    )
                )

        return constraints or None

    constraints: list[
        tuple[
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]
    ] = []

    for i, ok in enumerate(valid):
        if not ok or weight_sqrt[i] == 0.0:
            continue
        coeff_s = weight_sqrt[i] * inv_dr[i]
        coeff_t = weight_sqrt[i]

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

        out0 = outer_rows[outer_idx0[i]]
        out1 = outer_rows[outer_idx1[i]]

        rows_s = [int(rim_rows[i]), int(out0)]
        vecs_s = [coeff_s * normal, -coeff_s * outer_w0[i] * normal]
        if out1 != out0 or outer_w1[i] != 0.0:
            rows_s.append(int(out1))
            vecs_s.append(-coeff_s * outer_w1[i] * normal)

        shape_part = (np.asarray(rows_s, dtype=int), np.asarray(vecs_s, dtype=float))

        rows_tout = [int(row) for row in target_rows]
        vecs_tout = [coeff_t * float(w) * r_dir for w in target_weights]
        tout_part = (
            np.asarray(rows_tout, dtype=int),
            np.asarray(vecs_tout, dtype=float),
        )

        constraints.append((shape_part, None, tout_part))

        if disk_rows is not None and disk_r_hat is None:
            continue
        if disk_rows is not None:
            vecs_s_in = [-coeff_s * normal, coeff_s * outer_w0[i] * normal]
            if out1 != out0 or outer_w1[i] != 0.0:
                vecs_s_in.append(coeff_s * outer_w1[i] * normal)
            shape_part_in = (
                np.asarray(rows_s, dtype=int),
                np.asarray(vecs_s_in, dtype=float),
            )

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
                rows_disk, weights_disk, dirs_disk = disk_target
                rows_tin = list(rows_disk)
                vecs_tin = [
                    coeff_t * float(w) * np.asarray(vec, dtype=float)
                    for w, vec in zip(weights_disk, dirs_disk)
                ]
            else:
                rows_tin = [int(row) for row in target_rows]
                vecs_tin = [coeff_t * float(w) * r_dir for w in target_weights]
                disk_target = _disk_theta_rows_weights_and_direction(
                    data=data,
                    i=i,
                    theta_scalar_active=False,
                )
                if disk_target is not None:
                    rows_disk, _weights_disk, dirs_disk = disk_target
                    rows_tin.extend(rows_disk)
                    vecs_tin.extend(
                        [-coeff_t * np.asarray(vec, dtype=float) for vec in dirs_disk]
                    )

            tin_part = (
                np.asarray(rows_tin, dtype=int),
                np.asarray(vecs_tin, dtype=float),
            )
            constraints.append((shape_part_in, tin_part, None))

    return constraints or None


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

        if theta_scalar is not None and _use_disk_theta_targeting(
            global_params, matching_mode
        ):
            disk_target = _disk_theta_rows_weights_and_direction(
                data=data,
                i=i,
                theta_scalar_active=True,
            )
            if disk_target is None:
                if matching_mode == "ring_average_radial_v1":
                    agg_out_rows.extend(out_part[0].tolist())
                    agg_out_vecs.extend(out_part[1])
                else:
                    constraints.append((None, out_part))
                continue
            rows_disk, weights_disk, dirs_disk = disk_target
            rows_in = list(rows_disk)
            vecs_in = [
                coeff * float(w) * np.asarray(vec, dtype=float)
                for w, vec in zip(weights_disk, dirs_disk)
            ]
        else:
            rows_in = [int(row) for row in target_rows]
            vecs_in = [coeff * float(w) * r_dir for w in target_weights]
            disk_target = _disk_theta_rows_weights_and_direction(
                data=data,
                i=i,
                theta_scalar_active=False,
            )
            if disk_target is not None:
                rows_disk, _weights_disk, dirs_disk = disk_target
                rows_in.extend(rows_disk)
                vecs_in.extend(
                    [-coeff * np.asarray(vec, dtype=float) for vec in dirs_disk]
                )

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
