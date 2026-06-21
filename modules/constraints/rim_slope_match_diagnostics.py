"""Diagnostic reporters for hard rim-matching constraints."""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh
from modules.constraints.local_interface_shells import build_local_interface_shell_data
from modules.constraints.rim_slope_match_params import (
    _resolve_center,
    _resolve_group,
    _resolve_matching_mode,
    _sanitize_disk_group,
    _use_disk_theta_targeting,
)
from modules.constraints.rim_slope_match_payload import _build_matching_data
from modules.constraints.rim_slope_match_utils import (
    _collect_group_rows,
    _fit_plane_normal,
    _orthonormal_basis,
    _resolve_normal,
)


def matching_ring_diagnostics(mesh: Mesh, global_params, positions: np.ndarray) -> dict:
    """Report tagged ring candidates seen by the rim-matching constraint path."""
    matching_mode = _resolve_matching_mode(global_params)
    group = _resolve_group(global_params, "rim_slope_match_group")
    outer_group = _resolve_group(global_params, "rim_slope_match_outer_group")
    disk_group = _resolve_group(global_params, "rim_slope_match_disk_group")
    if matching_mode != "physical_edge_staggered_v1":
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
        "matching_mode": matching_mode,
    }
    if matching_mode == "physical_edge_staggered_v1":
        interface_group = disk_group or group
        out["reason"] = "missing_interface_group"
        out["outer_source"] = "local_interface_shell"
        if interface_group is None:
            return out
        try:
            local_shells = build_local_interface_shell_data(
                mesh, positions=positions, group=str(interface_group)
            )
        except AssertionError as exc:
            out["reason"] = str(exc)
            return out
        out.update(
            {
                "available": True,
                "reason": "ok",
                "construction_mode": "physical_edge_local_shell",
                "rim_count": int(local_shells.disk_rows.size),
                "outer_count": int(local_shells.rim_rows.size),
                "disk_count": int(local_shells.disk_rows.size),
                "rim_radius": float(local_shells.disk_radius),
                "outer_radius": float(local_shells.rim_radius),
                "disk_radius": float(local_shells.disk_radius),
            }
        )
        return out
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
    out["construction_mode"] = "legacy_tagged_rim_shell"
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
    elif matching_mode == "physical_edge_staggered_v1":
        tilt_rows = np.asarray(data.get("tilt_rows", data["outer_rows"]), dtype=int)
        tilt_rows0 = tilt_rows[
            np.asarray(data.get("tilt_idx0", data["outer_idx0"]), dtype=int)
        ]
        tilt_rows1 = tilt_rows[
            np.asarray(data.get("tilt_idx1", data["outer_idx1"]), dtype=int)
        ]
        tilt_w0 = np.asarray(data.get("tilt_w0", data["outer_w0"]), dtype=float)
        tilt_w1 = np.asarray(data.get("tilt_w1", data["outer_w1"]), dtype=float)
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
            "target_source": str(data.get("target_source", "unknown")),
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

    if theta_scalar is not None and _use_disk_theta_targeting(
        global_params, matching_mode
    ):
        if local_disk:
            tilt_in_rad = np.einsum(
                "ij,ij->i", mesh.tilts_in_view()[disk_rows], disk_r_hat
            )[valid]
        else:
            weight_sum = (
                float(np.sum(disk_weights)) if disk_weights is not None else 0.0
            )
            if weight_sum <= 0.0:
                out["disk_theta_source"] = "degenerate_disk_weights"
                out["inner_residual_available"] = False
                return out
            tilt_in_scalar = float(
                np.sum(
                    np.asarray(disk_weights, dtype=float)
                    * np.einsum("ij,ij->i", mesh.tilts_in_view()[disk_rows], disk_r_hat)
                )
                / weight_sum
            )
            tilt_in_rad = np.full(np.count_nonzero(valid), tilt_in_scalar, dtype=float)
    else:
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
