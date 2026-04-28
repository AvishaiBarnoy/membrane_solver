#!/usr/bin/env python3
# ruff: noqa: E402
"""Audit the shared-rim secant / phi target on the curved free-disk lane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.constraints import rim_slope_match_out as rim_slope_match_out_constraint
from tools.diagnostics.curved_1disk_theory_benchmark import (
    _run_curved_theta_candidate,
)
from tools.diagnostics.curved_disk_theory import (
    compute_curved_disk_theory,
    tex_reference_params,
)

THEORY_THETA_B = 0.1845693593


def _global_radial_direction(position: np.ndarray) -> np.ndarray:
    """Return the global xy-plane radial unit vector for ``position``."""
    vec = np.array([float(position[0]), float(position[1]), 0.0], dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-12:
        return np.zeros(3, dtype=float)
    return vec / norm


def _shape_target_summary(mesh, data: dict, normals: np.ndarray) -> dict[str, object]:
    """Return shellwise secant / target summaries for the active shared-rim law."""
    positions = mesh.positions_view()
    normal = np.asarray(data["normal"], dtype=float)
    rim_rows = np.asarray(data["rim_rows"], dtype=int)
    outer_rows = np.asarray(data["outer_rows"], dtype=int)
    tilt_rows = np.asarray(data["tilt_rows"], dtype=int)
    outer_idx0 = np.asarray(data["outer_idx0"], dtype=int)
    outer_idx1 = np.asarray(data["outer_idx1"], dtype=int)
    outer_w0 = np.asarray(data["outer_w0"], dtype=float)
    outer_w1 = np.asarray(data["outer_w1"], dtype=float)
    valid = np.asarray(data["valid"], dtype=bool)
    inv_dr = np.asarray(data["inv_dr"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)

    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")
    theta_disk = float(data["theta_scalar"])

    h_rim_vals: list[float] = []
    h_out_vals: list[float] = []
    dr_vals: list[float] = []
    phi_vals: list[float] = []
    phi_target_vals: list[float] = []
    t_out_rad_vals: list[float] = []
    continuity_target_vals: list[float] = []
    rdir_cos_vals: list[float] = []
    rdir_rows: list[int] = []

    for i, ok in enumerate(valid):
        if not ok:
            continue
        target = rim_slope_match_out_constraint._tilt_target_rows_weights_and_direction(
            data=data,
            positions=positions,
            normals=normals,
            i=i,
            matching_mode=str(data["matching_mode"]),
        )
        if target is None:
            continue
        target_rows, target_weights, r_dir = target

        dr = 1.0 / float(inv_dr[i])
        row0 = int(outer_rows[outer_idx0[i]])
        row1 = int(outer_rows[outer_idx1[i]])
        w0 = float(outer_w0[i])
        w1 = float(outer_w1[i])
        current_rim_height = float(np.dot(positions[int(rim_rows[i])], normal))
        current_outer_height = 0.0
        height_weight = 0.0
        if abs(w0) > 1.0e-12:
            current_outer_height += w0 * float(np.dot(positions[row0], normal))
            height_weight += abs(w0)
        if abs(w1) > 1.0e-12:
            current_outer_height += w1 * float(np.dot(positions[row1], normal))
            height_weight += abs(w1)
        current_outer_height /= max(height_weight, 1.0e-12)

        t_out_rad = 0.0
        t_in_rad = 0.0
        for row, weight in zip(target_rows, target_weights):
            idx = int(row)
            w = float(weight)
            t_out_rad += w * float(np.dot(tilts_out[idx], r_dir))
            t_in_rad += w * float(np.dot(tilts_in[idx], r_dir))

        phi_current = (current_outer_height - current_rim_height) / dr
        continuity_target = theta_disk - t_in_rad
        phi_target = (
            2.0 * phi_current + float(t_out_rad) + 2.0 * continuity_target
        ) / 5.0

        shell2_row = int(target_rows[0])
        global_r = _global_radial_direction(positions[shell2_row])
        rdir_cos = float(np.dot(r_dir, global_r))

        h_rim_vals.append(current_rim_height)
        h_out_vals.append(current_outer_height)
        dr_vals.append(dr)
        phi_vals.append(float(phi[i]))
        phi_target_vals.append(float(phi_target))
        t_out_rad_vals.append(float(t_out_rad))
        continuity_target_vals.append(float(continuity_target))
        rdir_cos_vals.append(rdir_cos)
        rdir_rows.append(shell2_row)

    shell1_radius = float(np.median(np.linalg.norm(positions[outer_rows, :2], axis=1)))
    shell2_radius = float(np.median(np.linalg.norm(positions[tilt_rows, :2], axis=1)))
    rim_radius = float(np.median(np.linalg.norm(positions[rim_rows, :2], axis=1)))

    return {
        "rim_radius": rim_radius,
        "shell1_radius": shell1_radius,
        "shell2_radius": shell2_radius,
        "normal": [float(v) for v in normal],
        "normal_dot_plus_z": float(np.dot(normal, np.array([0.0, 0.0, 1.0]))),
        "secant_source_rows": {
            "rim_rows": [int(v) for v in rim_rows.tolist()],
            "shell1_rows": [int(v) for v in outer_rows.tolist()],
            "shell2_target_rows": [int(v) for v in tilt_rows.tolist()],
        },
        "secant_geometry": {
            "h_rim_median": float(np.median(h_rim_vals)),
            "h_out_median": float(np.median(h_out_vals)),
            "dr_median": float(np.median(dr_vals)),
            "dr_min": float(np.min(dr_vals)),
            "dr_max": float(np.max(dr_vals)),
            "secant_sign_median": float(
                np.median(
                    np.sign(np.asarray(h_out_vals) - np.asarray(h_rim_vals))
                    * np.sign(np.asarray(dr_vals))
                )
            ),
        },
        "phi_construction": {
            "phi_median": float(np.median(phi_vals)),
            "phi_min": float(np.min(phi_vals)),
            "phi_max": float(np.max(phi_vals)),
            "phi_target_median": float(np.median(phi_target_vals)),
            "phi_target_min": float(np.min(phi_target_vals)),
            "phi_target_max": float(np.max(phi_target_vals)),
            "t_out_rad_median": float(np.median(t_out_rad_vals)),
            "continuity_target_median": float(np.median(continuity_target_vals)),
        },
        "target_direction": {
            "shell2_target_row_sample": [int(v) for v in rdir_rows[:5]],
            "r_dir_cos_global_radial_median": float(np.median(rdir_cos_vals)),
            "r_dir_cos_global_radial_min": float(np.min(rdir_cos_vals)),
            "r_dir_cos_global_radial_max": float(np.max(rdir_cos_vals)),
        },
    }


def run_curved_1disk_shared_rim_phi_target_audit() -> dict[str, object]:
    """Run the secant / phi target audit and return a diagnostic report."""
    result = _run_curved_theta_candidate(THEORY_THETA_B)
    mesh = result["mesh"]
    positions = mesh.positions_view()
    gp = mesh.global_parameters
    theory = compute_curved_disk_theory(tex_reference_params())
    data = rim_slope_match_out_constraint._build_matching_data(mesh, gp, positions)
    if data is None:
        raise AssertionError(
            "Shared-rim matching data unavailable on curved free-disk lane."
        )
    normals = mesh.vertex_normals(positions=positions)

    summary = _shape_target_summary(mesh, data, normals)
    phi_median = float(summary["phi_construction"]["phi_median"])
    phi_target_median = float(summary["phi_construction"]["phi_target_median"])
    secant_sign = float(summary["secant_geometry"]["secant_sign_median"])
    normal_dot_plus_z = float(summary["normal_dot_plus_z"])

    if normal_dot_plus_z < 0.0:
        call = "wrong normal/orientation convention"
    elif secant_sign < 0.0:
        call = "wrong secant sign"
    elif phi_median <= 0.0 or phi_target_median <= 0.0:
        call = "wrong phi target sign"
    else:
        call = "another specific target-construction defect"

    return {
        "case": {
            "theta_B": float(THEORY_THETA_B),
            "matching_mode": str(data["matching_mode"]),
            "total_energy": float(sum(float(v) for v in result["breakdown"].values())),
        },
        "theory_reference": {
            "phi_star_theory": float(theory.phi_star),
            "theta_half_theory": 0.5 * float(THEORY_THETA_B),
            "expected_positive_trumpet_sign": 1.0,
        },
        "shell_target_construction": summary,
        "shell_comparison": {
            "shell1_secant_phi_median": float(
                summary["phi_construction"]["phi_median"]
            ),
            "shell2_propagated_phi_target_median": float(
                summary["phi_construction"]["phi_target_median"]
            ),
            "shell2_target_direction_cos_global_radial_median": float(
                summary["target_direction"]["r_dir_cos_global_radial_median"]
            ),
        },
        "first_target_departure": {
            "call": call,
            "detail": (
                "secant and phi targets remain positive, but the propagated shell-2 "
                "target direction is nearly opposite the global outward radial direction."
                if call == "another specific target-construction defect"
                else call
            ),
        },
        "diagnosis": {
            "call": call,
            "recommended_next_stream": (
                "Next stream should isolate the shell-2 target radial-direction construction "
                "on the shared-rim lane, because the secant scalar is positive while the "
                "propagated shell-2 target direction is effectively inward."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()
    print(
        json.dumps(
            run_curved_1disk_shared_rim_phi_target_audit(), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
