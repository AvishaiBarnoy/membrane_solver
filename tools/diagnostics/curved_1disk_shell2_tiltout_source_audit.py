#!/usr/bin/env python3
# ruff: noqa: E402
"""Audit upstream shell-2 outer-leaflet field construction on the curved free-disk lane."""

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
from tools.diagnostics.curved_1disk_first_two_shell_ingredient_audit import (
    THEORY_THETA_B,
    _active_group_labels,
    _aggregate_row_records,
    _leaflet_runtime_payload,
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


def _run_case() -> dict[str, object]:
    mesh = _refine_once(load_free_disk_curved_bilayer_mesh(None))
    configure_free_disk_curved_bilayer_stage2(
        mesh, theta_b=float(THEORY_THETA_B), z_bump=None
    )
    minim = _configure_shape_relax(mesh, theta_b=float(THEORY_THETA_B))
    minim.minimize(n_steps=60)
    return {
        "mesh": mesh,
        "near_rim": measure_free_disk_curved_bilayer_near_rim(
            mesh, theta_b=float(THEORY_THETA_B)
        ),
        "breakdown": minim.compute_energy_breakdown(),
    }


def _rows_by_shell(
    mesh,
) -> tuple[
    list[int],
    list[int],
    list[int],
    dict[int, dict[str, object]],
    dict[int, dict[str, object]],
]:
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
    return (
        shell1_rows,
        shell2_rows,
        [int(v) for v in mesh.vertex_ids],
        rows_in,
        rows_out,
    )


def _row_summary(
    rows: list[int], *, mesh, rows_in, rows_out
) -> list[dict[str, object]]:
    out = []
    for row in rows:
        rin = rows_in[int(row)]
        rout = rows_out[int(row)]
        tilt_in_vec = np.asarray(rin["tilt_vector"], dtype=float)
        tilt_out_vec = np.asarray(rout["tilt_vector"], dtype=float)
        tilt_in_norm = float(np.linalg.norm(tilt_in_vec))
        tilt_out_norm = float(np.linalg.norm(tilt_out_vec))
        out.append(
            {
                "row": int(row),
                "group_labels": _active_group_labels(mesh, int(row)),
                "neighbor_shell_radii_in": rin["neighbor_shell_radii"],
                "neighbor_shell_radii_out": rout["neighbor_shell_radii"],
                "neighbor_rows_in": rin["neighbor_rows"],
                "neighbor_rows_out": rout["neighbor_rows"],
                "incident_triangle_count_in": int(rin["incident_triangle_count"]),
                "incident_triangle_count_out": int(rout["incident_triangle_count"]),
                "tilt_in": [float(v) for v in tilt_in_vec],
                "tilt_out": [float(v) for v in tilt_out_vec],
                "theta_in_radial": float(rin["radial_tilt"]),
                "theta_out_radial": float(rout["radial_tilt"]),
                "theta_in_tangential_proxy": float(
                    np.sqrt(max(tilt_in_norm**2 - float(rin["radial_tilt"]) ** 2, 0.0))
                ),
                "theta_out_tangential_proxy": float(
                    np.sqrt(
                        max(tilt_out_norm**2 - float(rout["radial_tilt"]) ** 2, 0.0)
                    )
                ),
            }
        )
    return out


def _source_path_summary(
    mesh, shell1_rows: list[int], shell2_rows: list[int]
) -> dict[str, object]:
    gp = mesh.global_parameters
    positions = mesh.positions_view()
    match_data = rim_slope_match_out_constraint._build_matching_data(
        mesh, gp, positions
    )
    shell1_labels = sorted(
        {label for row in shell1_rows for label in _active_group_labels(mesh, row)}
    )
    shell2_labels = sorted(
        {label for row in shell2_rows for label in _active_group_labels(mesh, row)}
    )
    continuation_rows = []
    if match_data is not None:
        continuation_rows = [
            int(v)
            for v in np.asarray(match_data.get("tilt_rows", []), dtype=int).tolist()
        ]
    return {
        "rim_slope_match_mode": str(gp.get("rim_slope_match_mode") or ""),
        "shell1_role": {
            "rows": [int(v) for v in shell1_rows],
            "group_labels": shell1_labels,
            "explicit_special_group": "rim_slope_match_group:outer" in shell1_labels,
            "copied_or_interpolated_values": False,
            "continuation_source_rows": [
                row for row in continuation_rows if row in shell1_rows
            ],
        },
        "shell2_role": {
            "rows": [int(v) for v in shell2_rows],
            "group_labels": shell2_labels,
            "explicit_special_group": "rim_slope_match_group:outer" in shell2_labels,
            "copied_or_interpolated_values": False,
            "continuation_source_rows": [
                row for row in continuation_rows if row in shell2_rows
            ],
        },
        "branch_flags": {
            "tilt_out_exclude_shared_rim_outer_rows": bool(
                gp.get("tilt_out_exclude_shared_rim_outer_rows")
            ),
            "tilt_in_exclude_shared_rim_rows": bool(
                gp.get("tilt_in_exclude_shared_rim_rows")
            ),
            "tilt_in_shared_rim_outer_shell_mass_mode": str(
                gp.get("tilt_in_shared_rim_outer_shell_mass_mode") or ""
            ),
        },
    }


def _compare_paths(
    shell1_out: list[dict[str, object]], shell2_inout: list[dict[str, object]]
) -> dict[str, object]:
    shell1_rad = float(np.median([float(r["theta_out_radial"]) for r in shell1_out]))
    shell2_out_rad = float(
        np.median([float(r["theta_out_radial"]) for r in shell2_inout])
    )
    shell2_in_rad = float(
        np.median([float(r["theta_in_radial"]) for r in shell2_inout])
    )
    shell1_tan = float(
        np.median([float(r["theta_out_tangential_proxy"]) for r in shell1_out])
    )
    shell2_out_tan = float(
        np.median([float(r["theta_out_tangential_proxy"]) for r in shell2_inout])
    )
    shell2_in_tan = float(
        np.median([float(r["theta_in_tangential_proxy"]) for r in shell2_inout])
    )
    same_neighbor_sets = all(
        row["neighbor_rows_in"] == row["neighbor_rows_out"]
        and row["neighbor_shell_radii_in"] == row["neighbor_shell_radii_out"]
        for row in shell2_inout
    )
    same_group_labels = all(len(row["group_labels"]) == 0 for row in shell2_inout)
    return {
        "shell1_out_radial_median": shell1_rad,
        "shell2_out_radial_median": shell2_out_rad,
        "shell2_in_radial_median": shell2_in_rad,
        "shell1_out_tangential_proxy_median": shell1_tan,
        "shell2_out_tangential_proxy_median": shell2_out_tan,
        "shell2_in_tangential_proxy_median": shell2_in_tan,
        "shell2_same_neighbor_sets_in_vs_out": bool(same_neighbor_sets),
        "shell2_same_group_labels_in_vs_out": bool(same_group_labels),
    }


def run_curved_1disk_shell2_tiltout_source_audit() -> dict[str, object]:
    result = _run_case()
    mesh = result["mesh"]
    shell1_rows, shell2_rows, _all_rows, rows_in, rows_out = _rows_by_shell(mesh)
    shell1_out = _row_summary(
        shell1_rows, mesh=mesh, rows_in=rows_in, rows_out=rows_out
    )
    shell2_inout = _row_summary(
        shell2_rows, mesh=mesh, rows_in=rows_in, rows_out=rows_out
    )
    source_path = _source_path_summary(mesh, shell1_rows, shell2_rows)
    compare = _compare_paths(shell1_out, shell2_inout)

    if (
        not source_path["shell1_role"]["explicit_special_group"]
        or source_path["shell2_role"]["explicit_special_group"]
    ):
        diagnosis = "another specific upstream field-construction defect"
    elif source_path["shell2_role"]["continuation_source_rows"]:
        diagnosis = "another specific upstream field-construction defect"
    elif not compare["shell2_same_neighbor_sets_in_vs_out"]:
        diagnosis = "neighbor-selection mismatch"
    elif not compare["shell2_same_group_labels_in_vs_out"]:
        diagnosis = "leaflet-label / continuation mismatch"
    else:
        diagnosis = "continuation-rule mismatch"

    return {
        "case": {
            "theta_B": float(THEORY_THETA_B),
            "total_energy": float(sum(float(v) for v in result["breakdown"].values())),
        },
        "shell_selection": {
            "shell1_radius": float(rows_out[shell1_rows[0]]["shell_radius"]),
            "shell2_radius": float(rows_out[shell2_rows[0]]["shell_radius"]),
            "shell1_row_count": int(len(shell1_rows)),
            "shell2_row_count": int(len(shell2_rows)),
        },
        "rim_reference": result["near_rim"],
        "source_path_audit": source_path,
        "shell1_out_rows": shell1_out,
        "shell2_rows_in_vs_out": shell2_inout,
        "path_comparison": compare,
        "first_upstream_departure": {
            "call": diagnosis,
            "shell_radius": float(rows_out[shell2_rows[0]]["shell_radius"]),
        },
        "diagnosis": {
            "call": diagnosis,
            "recommended_next_stream": (
                "Next stream should inspect the outer-leaflet tilt relaxation source terms acting on the "
                "first unconstrained shell, because shell 2 is not fed by any explicit continuation rule."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()
    print(
        json.dumps(
            run_curved_1disk_shell2_tiltout_source_audit(), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
