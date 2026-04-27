#!/usr/bin/env python3
# ruff: noqa: E402
"""Audit first-two-shell ``div_eval`` assembly on the curved free-disk lane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.tilt_operators import _resolve_transport_model
from modules.energy import bending_tilt_leaflet as _btl
from tools.diagnostics.curved_1disk_first_two_shell_ingredient_audit import (
    THEORY_THETA_B,
    _aggregate_row_records,
    _leaflet_runtime_payload,
    _select_target_shells,
)
from tools.diagnostics.curved_1disk_theory_benchmark import _run_curved_theta_candidate


def _shell_rows(
    records: dict[int, dict[str, object]], shell: float
) -> list[dict[str, object]]:
    """Return sorted row records for one shell."""
    rows = [
        rec for rec in records.values() if float(rec["shell_radius"]) == float(shell)
    ]
    return sorted(rows, key=lambda item: int(item["row"]))


def _median(rows: list[dict[str, object]], key: str) -> float:
    """Return median value for a per-row scalar key."""
    vals = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.median(vals)) if vals.size else 0.0


def run_curved_1disk_first_two_shell_diveval_audit() -> dict[str, object]:
    """Run the first-two-shell ``div_eval`` audit and return a report."""
    result = _run_curved_theta_candidate(THEORY_THETA_B)
    mesh = result["mesh"]
    gp = mesh.global_parameters
    payload_in = _leaflet_runtime_payload(mesh, leaflet="in")
    payload_out = _leaflet_runtime_payload(mesh, leaflet="out")
    rows_in = _aggregate_row_records(mesh, payload_in)
    rows_out = _aggregate_row_records(mesh, payload_out)
    target_shells = _select_target_shells(rows_in)

    shells: list[dict[str, object]] = []
    first_culprit = "combined local expression"
    first_shell = None
    for shell in target_shells:
        in_rows = _shell_rows(rows_in, shell)
        out_rows = _shell_rows(rows_out, shell)
        shell_row = {
            "shell_radius": float(shell),
            "transport_model": str(
                _resolve_transport_model(gp.get("tilt_transport_model", "ambient_v1"))
            ),
            "inner_lane_sign_fix_active": bool(
                _btl._use_curved_free_disk_shared_rim_inner_div_sign_fix(gp)
            ),
            "in": {
                "row_count": int(len(in_rows)),
                "div_sign": float(
                    _btl._resolved_div_sign(gp, cache_tag="in", default=-1.0)
                ),
                "div_raw_median": _median(in_rows, "div_raw_median"),
                "div_signed_median": _median(in_rows, "div_signed_median"),
                "div_term_median": _median(in_rows, "div_term_median"),
                "div_eval_median": _median(in_rows, "div_eval_median"),
                "base_term_median": _median(in_rows, "base_term_vertex"),
                "radial_tilt_median": _median(in_rows, "radial_tilt"),
                "rows": [
                    {
                        "row": int(row["row"]),
                        "group_labels": row["group_labels"],
                        "neighbor_rows": row["neighbor_rows"],
                        "neighbor_shell_radii": row["neighbor_shell_radii"],
                        "div_raw_values": [float(v) for v in row["div_raw_values"]],
                        "div_signed_values": [
                            float(v) for v in row["div_signed_values"]
                        ],
                        "div_term_values": [float(v) for v in row["div_term_values"]],
                        "div_eval_values": [float(v) for v in row["div_eval_values"]],
                        "base_term_vertex": float(row["base_term_vertex"]),
                    }
                    for row in in_rows
                ],
            },
            "out": {
                "row_count": int(len(out_rows)),
                "div_sign": float(
                    _btl._resolved_div_sign(gp, cache_tag="out", default=1.0)
                ),
                "div_raw_median": _median(out_rows, "div_raw_median"),
                "div_signed_median": _median(out_rows, "div_signed_median"),
                "div_term_median": _median(out_rows, "div_term_median"),
                "div_eval_median": _median(out_rows, "div_eval_median"),
                "base_term_median": _median(out_rows, "base_term_vertex"),
                "radial_tilt_median": _median(out_rows, "radial_tilt"),
                "rows": [
                    {
                        "row": int(row["row"]),
                        "group_labels": row["group_labels"],
                        "neighbor_rows": row["neighbor_rows"],
                        "neighbor_shell_radii": row["neighbor_shell_radii"],
                        "div_raw_values": [float(v) for v in row["div_raw_values"]],
                        "div_signed_values": [
                            float(v) for v in row["div_signed_values"]
                        ],
                        "div_term_values": [float(v) for v in row["div_term_values"]],
                        "div_eval_values": [float(v) for v in row["div_eval_values"]],
                        "base_term_vertex": float(row["base_term_vertex"]),
                    }
                    for row in out_rows
                ],
            },
        }
        shell_row["subexpression_deltas"] = {
            "div_raw_sign_matches": bool(
                np.sign(shell_row["in"]["div_raw_median"])
                == np.sign(shell_row["out"]["div_raw_median"])
            ),
            "div_signed_sign_matches": bool(
                np.sign(shell_row["in"]["div_signed_median"])
                == np.sign(shell_row["out"]["div_signed_median"])
            ),
            "div_term_sign_matches": bool(
                np.sign(shell_row["in"]["div_term_median"])
                == np.sign(shell_row["out"]["div_term_median"])
            ),
            "div_eval_sign_matches": bool(
                np.sign(shell_row["in"]["div_eval_median"])
                == np.sign(shell_row["out"]["div_eval_median"])
            ),
        }
        if first_shell is None:
            if (
                shell_row["subexpression_deltas"]["div_raw_sign_matches"]
                and not shell_row["subexpression_deltas"]["div_signed_sign_matches"]
            ):
                first_culprit = "sign convention application"
                first_shell = float(shell)
            elif (
                shell_row["subexpression_deltas"]["div_signed_sign_matches"]
                and not shell_row["subexpression_deltas"]["div_term_sign_matches"]
            ):
                first_culprit = "boundary-conditioned div_term branch"
                first_shell = float(shell)
            elif (
                shell_row["subexpression_deltas"]["div_term_sign_matches"]
                and not shell_row["subexpression_deltas"]["div_eval_sign_matches"]
            ):
                first_culprit = "post-div_term div_eval branch"
                first_shell = float(shell)
        shells.append(shell_row)

    return {
        "case": {"theta_B": float(THEORY_THETA_B)},
        "lane_signature": {
            "rim_slope_match_mode": str(gp.get("rim_slope_match_mode") or ""),
            "bending_tilt_base_term_boundary_group_in": str(
                gp.get("bending_tilt_base_term_boundary_group_in") or ""
            ),
            "bending_tilt_base_term_boundary_group_out": str(
                gp.get("bending_tilt_base_term_boundary_group_out") or ""
            ),
            "tilt_thetaB_group_in": str(gp.get("tilt_thetaB_group_in") or ""),
            "rim_slope_match_group": str(gp.get("rim_slope_match_group") or ""),
            "rim_slope_match_outer_group": str(
                gp.get("rim_slope_match_outer_group") or ""
            ),
        },
        "target_shell_radii": [float(v) for v in target_shells],
        "shells": shells,
        "first_offending_subexpression": {
            "call": str(first_culprit),
            "shell_radius": None if first_shell is None else float(first_shell),
        },
    }


def main() -> None:
    """Run the audit and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()
    report = run_curved_1disk_first_two_shell_diveval_audit()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
