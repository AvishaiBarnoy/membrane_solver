#!/usr/bin/env python3
# ruff: noqa: E402
"""Audit shell-2 outer-leaflet continuation on the curved free-disk shared-rim lane."""

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
from modules.energy.rim_slope_match_out import _tilt_match_rows_and_directions
from tools.diagnostics.curved_1disk_first_two_shell_ingredient_audit import (
    THEORY_THETA_B,
    _active_group_labels,
    _aggregate_row_records,
    _leaflet_runtime_payload,
    _row_shell_radius_map,
    _select_target_shells,
)
from tools.diagnostics.curved_1disk_theory_benchmark import (
    _configure_shape_relax,
    _refine_once,
    load_free_disk_curved_bilayer_mesh,
)
from tools.diagnostics.free_disk_profile_protocol import (
    configure_free_disk_curved_bilayer_stage2,
    measure_free_disk_curved_bilayer_near_rim,
)


def _run_case(
    *, theta_b: float, exclude_shared_rim_outer_rows: bool
) -> dict[str, object]:
    mesh = _refine_once(load_free_disk_curved_bilayer_mesh(None))
    configure_free_disk_curved_bilayer_stage2(mesh, theta_b=float(theta_b), z_bump=None)
    mesh.global_parameters.set(
        "tilt_out_exclude_shared_rim_outer_rows", bool(exclude_shared_rim_outer_rows)
    )
    minim = _configure_shape_relax(mesh, theta_b=float(theta_b))
    minim.minimize(n_steps=60)
    return {
        "mesh": mesh,
        "near_rim": measure_free_disk_curved_bilayer_near_rim(
            mesh, theta_b=float(theta_b)
        ),
        "breakdown": minim.compute_energy_breakdown(),
    }


def _row_component_table(mesh, rows: list[int]) -> list[dict[str, object]]:
    positions = mesh.positions_view()
    normals = mesh.vertex_normals(positions=positions)
    center = np.asarray(
        mesh.global_parameters.get("rim_slope_match_center") or [0.0, 0.0, 0.0],
        dtype=float,
    )
    normal = np.asarray(
        mesh.global_parameters.get("rim_slope_match_normal") or [0.0, 0.0, 1.0],
        dtype=float,
    )
    normal = normal / max(np.linalg.norm(normal), 1.0e-12)
    row_shell = _row_shell_radius_map(mesh)
    out = []
    for row in rows:
        row = int(row)
        pos = positions[row]
        n_row = normals[row]
        r_vec = pos - center
        r_vec = r_vec - np.dot(r_vec, normal) * normal
        r_hat = r_vec / max(np.linalg.norm(r_vec), 1.0e-12)
        t_hat = np.cross(n_row, r_hat)
        t_hat = t_hat / max(np.linalg.norm(t_hat), 1.0e-12)
        tilt_in = np.asarray(mesh.tilts_in_view()[row], dtype=float)
        tilt_out = np.asarray(mesh.tilts_out_view()[row], dtype=float)
        out.append(
            {
                "row": row,
                "shell_radius": float(row_shell[row]),
                "group_labels": _active_group_labels(mesh, row),
                "tilt_in": [float(v) for v in tilt_in],
                "tilt_out": [float(v) for v in tilt_out],
                "theta_in_radial": float(np.dot(tilt_in, r_hat)),
                "theta_out_radial": float(np.dot(tilt_out, r_hat)),
                "theta_in_tangential": float(np.dot(tilt_in, t_hat)),
                "theta_out_tangential": float(np.dot(tilt_out, t_hat)),
                "theta_in_normal": float(np.dot(tilt_in, n_row)),
                "theta_out_normal": float(np.dot(tilt_out, n_row)),
                "tilt_out_norm": float(np.linalg.norm(tilt_out)),
                "normal": [float(v) for v in n_row],
                "r_hat": [float(v) for v in r_hat],
                "t_hat": [float(v) for v in t_hat],
            }
        )
    return out


def _outer_shell_rows(mesh) -> tuple[list[int], list[int]]:
    positions = mesh.positions_view()
    data = rim_slope_match_out_constraint._build_matching_data(
        mesh, mesh.global_parameters, positions
    )
    if data is not None and data.get("tilt_rows") is not None:
        return (
            sorted({int(v) for v in np.asarray(data["outer_rows"], dtype=int)}),
            sorted({int(v) for v in np.asarray(data["tilt_rows"], dtype=int)}),
        )

    payload_in = _leaflet_runtime_payload(mesh, leaflet="in")
    payload_out = _leaflet_runtime_payload(mesh, leaflet="out")
    rows_in = _aggregate_row_records(mesh, payload_in)
    rows_out = _aggregate_row_records(mesh, payload_out)
    target_shells = _select_target_shells(rows_in)
    shell1, shell2 = [float(v) for v in target_shells]
    shell1_rows = sorted(
        int(rec["row"])
        for rec in rows_out.values()
        if float(rec["shell_radius"]) == shell1
    )
    shell2_rows = sorted(
        int(rec["row"])
        for rec in rows_out.values()
        if float(rec["shell_radius"]) == shell2
    )
    return shell1_rows, shell2_rows


def _continuation_summary(
    mesh, shell1_rows: list[int], shell2_rows: list[int]
) -> dict[str, object]:
    row_shell = _row_shell_radius_map(mesh)
    positions = mesh.positions_view()
    normals = mesh.vertex_normals(positions=positions)
    outer_rows = np.asarray(shell1_rows, dtype=int)
    rim_rows = outer_rows.copy()
    outer_idx0 = np.arange(len(rim_rows), dtype=int)
    outer_idx1 = np.arange(len(rim_rows), dtype=int)
    outer_w0 = np.ones(len(rim_rows), dtype=float)
    outer_w1 = np.zeros(len(rim_rows), dtype=float)
    center = np.asarray(
        mesh.global_parameters.get("rim_slope_match_center") or [0.0, 0.0, 0.0],
        dtype=float,
    )
    normal = np.asarray(
        mesh.global_parameters.get("rim_slope_match_normal") or [0.0, 0.0, 1.0],
        dtype=float,
    )
    normal = normal / max(np.linalg.norm(normal), 1.0e-12)
    rim_pos = positions[rim_rows]
    r_vec = rim_pos - center[None, :]
    r_vec = r_vec - np.einsum("ij,j->i", r_vec, normal)[:, None] * normal[None, :]
    r_hat = r_vec / np.maximum(np.linalg.norm(r_vec, axis=1)[:, None], 1.0e-12)
    tilt_rows0, tilt_rows1, tilt_w0, tilt_w1, r_dir, good_dir, mode = (
        _tilt_match_rows_and_directions(
            matching_mode="shared_rim_staggered_v1",
            rim_rows=rim_rows,
            outer_rows=outer_rows,
            outer_idx0=outer_idx0,
            outer_idx1=outer_idx1,
            outer_w0=outer_w0,
            outer_w1=outer_w1,
            r_hat=r_hat,
            normals=normals,
        )
    )
    shell2_set = set(int(v) for v in shell2_rows)
    out_rows = []
    for idx, row in enumerate(shell1_rows):
        out_rows.append(
            {
                "shell1_row": int(row),
                "shell1_group_labels": _active_group_labels(mesh, int(row)),
                "matched_tilt_row0": int(tilt_rows0[idx]),
                "matched_tilt_row1": int(tilt_rows1[idx]),
                "matched_row0_shell": float(row_shell[int(tilt_rows0[idx])]),
                "matched_row1_shell": float(row_shell[int(tilt_rows1[idx])]),
                "shell2_neighbor_present": bool(
                    any(
                        int(v) in shell2_set for v in (tilt_rows0[idx], tilt_rows1[idx])
                    )
                ),
                "tilt_weight0": float(tilt_w0[idx]),
                "tilt_weight1": float(tilt_w1[idx]),
                "r_dir": [float(v) for v in r_dir[idx]],
                "good_dir": bool(good_dir[idx]),
                "mode": str(mode),
            }
        )
    return {"shell1_to_active_tilt_rows": out_rows}


def _stage_summary(
    shell1_rows: list[dict[str, object]], shell2_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    def med(rows, key):
        return float(np.median(np.asarray([float(r[key]) for r in rows], dtype=float)))

    stages = [
        {
            "stage": "theta_out_radial",
            "shell1_abs_median": abs(med(shell1_rows, "theta_out_radial")),
            "shell2_abs_median": abs(med(shell2_rows, "theta_out_radial")),
        },
        {
            "stage": "theta_out_tangential",
            "shell1_abs_median": abs(med(shell1_rows, "theta_out_tangential")),
            "shell2_abs_median": abs(med(shell2_rows, "theta_out_tangential")),
        },
        {
            "stage": "theta_out_normal",
            "shell1_abs_median": abs(med(shell1_rows, "theta_out_normal")),
            "shell2_abs_median": abs(med(shell2_rows, "theta_out_normal")),
        },
        {
            "stage": "tilt_out_norm",
            "shell1_abs_median": abs(med(shell1_rows, "tilt_out_norm")),
            "shell2_abs_median": abs(med(shell2_rows, "tilt_out_norm")),
        },
    ]
    for stage in stages:
        stage["ratio_shell2_over_shell1"] = float(
            stage["shell2_abs_median"] / max(stage["shell1_abs_median"], 1.0e-12)
        )
    first = "theta_out_radial"
    if (
        stages[0]["ratio_shell2_over_shell1"] > 0.5
        and stages[1]["ratio_shell2_over_shell1"] > 1.5
    ):
        first = "theta_out_tangential"
    return stages, first


def run_curved_1disk_shell2_tiltout_audit() -> dict[str, object]:
    baseline = _run_case(theta_b=THEORY_THETA_B, exclude_shared_rim_outer_rows=True)
    mesh = baseline["mesh"]
    shell1_rows, shell2_rows = _outer_shell_rows(mesh)
    shell1_table = _row_component_table(mesh, shell1_rows)
    shell2_table = _row_component_table(mesh, shell2_rows)
    stages, first_departure = _stage_summary(shell1_table, shell2_table)

    toggle = _run_case(theta_b=THEORY_THETA_B, exclude_shared_rim_outer_rows=False)
    toggle_mesh = toggle["mesh"]
    _, toggle_shell2_rows = _outer_shell_rows(toggle_mesh)
    toggle_shell2_table = _row_component_table(toggle_mesh, toggle_shell2_rows)

    shell2_baseline_rad = float(
        np.median([float(r["theta_out_radial"]) for r in shell2_table])
    )
    shell2_toggle_rad = float(
        np.median([float(r["theta_out_radial"]) for r in toggle_shell2_table])
    )

    diagnosis = "shell-2 outer tilt field departure"
    if abs(shell2_toggle_rad - shell2_baseline_rad) > 1.0e-3:
        diagnosis = "shared-rim outer-row exclusion branch"

    return {
        "case": {
            "theta_B": float(THEORY_THETA_B),
            "transport_model": str(
                mesh.global_parameters.get("tilt_transport_model") or "ambient_v1"
            ),
            "rim_slope_match_mode": str(
                mesh.global_parameters.get("rim_slope_match_mode") or ""
            ),
        },
        "shell_selection": {
            "shell1_radius": float(shell1_table[0]["shell_radius"]),
            "shell2_radius": float(shell2_table[0]["shell_radius"]),
            "shell1_row_count": int(len(shell1_table)),
            "shell2_row_count": int(len(shell2_table)),
        },
        "rim_reference": baseline["near_rim"],
        "shell1_rows": shell1_table,
        "shell2_rows": shell2_table,
        "continuation_ladder": stages,
        "transport_and_stencil_audit": _continuation_summary(
            mesh, shell1_rows, shell2_rows
        ),
        "toggle_comparison": {
            "tilt_out_exclude_shared_rim_outer_rows_true": shell2_baseline_rad,
            "tilt_out_exclude_shared_rim_outer_rows_false": shell2_toggle_rad,
        },
        "first_material_departure": {
            "call": first_departure,
            "shell_radius": float(shell2_table[0]["shell_radius"]),
        },
        "diagnosis": {
            "call": diagnosis,
            "recommended_next_stream": (
                "If no lane-local continuation toggle changes shell-2 tilt_out, the next stream should "
                "inspect the outer-leaflet tilt relaxation sources on shell 2 rather than div_eval assembly."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()
    print(json.dumps(run_curved_1disk_shell2_tiltout_audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
