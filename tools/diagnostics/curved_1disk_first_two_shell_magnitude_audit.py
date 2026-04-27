#!/usr/bin/env python3
# ruff: noqa: E402
"""Audit first-two-shell magnitude-side divergence on the curved free-disk lane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.curved_1disk_first_two_shell_ingredient_audit import (
    OUTER_RADIUS,
    THEORY_THETA_B,
    _aggregate_row_records,
    _leaflet_runtime_payload,
    _select_target_shells,
)
from tools.diagnostics.curved_1disk_theory_benchmark import _run_curved_theta_candidate
from tools.diagnostics.free_disk_profile_protocol import (
    measure_free_disk_curved_bilayer_near_rim,
)


def _median_abs(rows: list[dict[str, object]], key: str) -> float:
    def _resolve_value(row: dict[str, object]) -> float:
        if key in row:
            return abs(float(row[key]))
        if key == "prefactor_median":
            return abs(
                float(row.get("kappa_median", 0.0))
                * float(row.get("effective_over_vor_ratio", 0.0))
            )
        return 0.0

    vals = np.asarray([_resolve_value(r) for r in rows], dtype=float)
    return float(np.median(vals)) if vals.size else 0.0


def _median_of_list_abs(rows: list[dict[str, object]], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        if key in row:
            vals.extend(abs(float(v)) for v in row[key])
            continue
        if key == "corner_div_contrib_values":
            tilt = np.asarray(row.get("tilt_vector", []), dtype=float)
            weights = row.get("triangle_weight_vectors", [])
            vals.extend(
                abs(float(np.dot(tilt, np.asarray(weight, dtype=float))))
                for weight in weights
            )
    arr = np.asarray(vals, dtype=float)
    return float(np.median(arr)) if arr.size else 0.0


def _triangle_records(
    leaflet_payload: dict[str, object], target_shells: list[float]
) -> dict[str, list[dict[str, object]]]:
    """Return trianglewise contributions touching the target shells."""
    out = {str(float(shell)): [] for shell in target_shells}
    required = {
        "tri_rows",
        "outer_mask",
        "row_shell_radius",
        "g0",
        "g1",
        "g2",
        "tilt_vectors",
        "div_raw",
        "div_signed",
        "div_term",
        "div_eval",
        "base_tri",
        "kappa_tri",
        "va_eff",
        "energy_vertex",
    }
    if not required.issubset(leaflet_payload):
        return out
    tri_rows = np.asarray(leaflet_payload["tri_rows"], dtype=np.int32)
    outer_mask = np.asarray(leaflet_payload["outer_mask"], dtype=bool)
    row_shell_radius = np.asarray(leaflet_payload["row_shell_radius"], dtype=float)
    g0 = np.asarray(leaflet_payload["g0"], dtype=float)
    g1 = np.asarray(leaflet_payload["g1"], dtype=float)
    g2 = np.asarray(leaflet_payload["g2"], dtype=float)
    tilt_vectors = np.asarray(leaflet_payload["tilt_vectors"], dtype=float)
    div_raw = np.asarray(leaflet_payload["div_raw"], dtype=float)
    div_signed = np.asarray(leaflet_payload["div_signed"], dtype=float)
    div_term = np.asarray(leaflet_payload["div_term"], dtype=float)
    div_eval = np.asarray(leaflet_payload["div_eval"], dtype=float)
    base_tri = np.asarray(leaflet_payload["base_tri"], dtype=float)
    kappa_tri = np.asarray(leaflet_payload["kappa_tri"], dtype=float)
    va_eff = np.asarray(leaflet_payload["va_eff"], dtype=float)
    energy_vertex = np.asarray(leaflet_payload["energy_vertex"], dtype=float)

    for tri_idx in np.flatnonzero(outer_mask):
        rows = tri_rows[tri_idx]
        tri_shells = sorted(
            float(v)
            for v in {float(row_shell_radius[int(row)]) for row in rows}
            if float(v) in target_shells
        )
        if not tri_shells:
            continue
        gradients = (g0[tri_idx], g1[tri_idx], g2[tri_idx])
        rec = {
            "triangle_index": int(tri_idx),
            "rows": [int(v) for v in rows],
            "row_shell_radii": [float(row_shell_radius[int(v)]) for v in rows],
            "div_raw": float(div_raw[tri_idx]),
            "div_signed": float(div_signed[tri_idx]),
            "div_term": float(div_term[tri_idx]),
            "div_eval": float(div_eval[tri_idx]),
            "corners": [],
        }
        for corner, row in enumerate(rows):
            grad = np.asarray(gradients[corner], dtype=float)
            tilt = np.asarray(tilt_vectors[int(row)], dtype=float)
            rec["corners"].append(
                {
                    "row": int(row),
                    "row_shell_radius": float(row_shell_radius[int(row)]),
                    "tilt_vector": [float(v) for v in tilt],
                    "tilt_norm": float(np.linalg.norm(tilt)),
                    "gradient": [float(v) for v in grad],
                    "gradient_norm": float(np.linalg.norm(grad)),
                    "corner_div_contrib": float(np.dot(tilt, grad)),
                    "base_corner": float(base_tri[tri_idx, corner]),
                    "prefactor": float(
                        kappa_tri[tri_idx, corner] * va_eff[tri_idx, corner]
                    ),
                    "term_corner": float(base_tri[tri_idx, corner] + div_eval[tri_idx]),
                    "local_contribution": float(energy_vertex[tri_idx, corner]),
                }
            )
        for shell in tri_shells:
            out[str(float(shell))].append(rec)
    return out


def _shell_stage_summary(
    shell: float,
    *,
    in_rows: list[dict[str, object]],
    out_rows: list[dict[str, object]],
    near_rim: dict[str, float],
) -> dict[str, object]:
    """Build ordered magnitude-stage comparison for one shell."""
    stages = [
        {
            "stage": "radial_tilt_input",
            "in_abs_median": _median_abs(in_rows, "radial_tilt"),
            "out_abs_median": _median_abs(out_rows, "radial_tilt"),
        },
        {
            "stage": "corner_divergence_stencil_input",
            "in_abs_median": _median_of_list_abs(in_rows, "corner_div_contrib_values"),
            "out_abs_median": _median_of_list_abs(
                out_rows, "corner_div_contrib_values"
            ),
        },
        {
            "stage": "div_raw",
            "in_abs_median": _median_abs(in_rows, "div_raw_median"),
            "out_abs_median": _median_abs(out_rows, "div_raw_median"),
        },
        {
            "stage": "div_eval",
            "in_abs_median": _median_abs(in_rows, "div_eval_median"),
            "out_abs_median": _median_abs(out_rows, "div_eval_median"),
        },
        {
            "stage": "geometric_prefactor",
            "in_abs_median": _median_abs(in_rows, "prefactor_median"),
            "out_abs_median": _median_abs(out_rows, "prefactor_median"),
        },
        {
            "stage": "combined_term",
            "in_abs_median": _median_abs(in_rows, "term_median"),
            "out_abs_median": _median_abs(out_rows, "term_median"),
        },
        {
            "stage": "local_contribution",
            "in_abs_median": _median_abs(in_rows, "local_contribution_sum"),
            "out_abs_median": _median_abs(out_rows, "local_contribution_sum"),
        },
    ]
    for stage in stages:
        stage["ratio_in_over_out"] = float(
            stage["in_abs_median"] / max(stage["out_abs_median"], 1.0e-12)
        )
        stage["material_magnitude_departure"] = bool(
            stage["ratio_in_over_out"] > 1.5 or stage["ratio_in_over_out"] < (1.0 / 1.5)
        )
    return {
        "shell_radius": float(shell),
        "rim_reference": {
            "theta_outer_in": float(near_rim["theta_outer_in"]),
            "theta_outer_out": float(near_rim["theta_outer_out"]),
            "phi": float(near_rim["phi"]),
            "theta_B_half": 0.5 * float(near_rim["theta_b"]),
        },
        "stages": stages,
        "row_count": {"in": int(len(in_rows)), "out": int(len(out_rows))},
    }


def _first_material_departure(shellwise: list[dict[str, object]]) -> dict[str, object]:
    """Return earliest materially divergent magnitude stage."""
    for shell in shellwise:
        for stage in shell["stages"]:
            if stage["material_magnitude_departure"]:
                return {
                    "call": str(stage["stage"]),
                    "shell_radius": float(shell["shell_radius"]),
                    "ratio_in_over_out": float(stage["ratio_in_over_out"]),
                }
    return {
        "call": "combined local expression",
        "shell_radius": None,
        "ratio_in_over_out": 1.0,
    }


def run_curved_1disk_first_two_shell_magnitude_audit() -> dict[str, object]:
    """Run the first-two-shell magnitude audit and return a report."""
    result = _run_curved_theta_candidate(THEORY_THETA_B)
    mesh = result["mesh"]
    near_rim = measure_free_disk_curved_bilayer_near_rim(mesh, theta_b=THEORY_THETA_B)

    payload_in = _leaflet_runtime_payload(mesh, leaflet="in")
    payload_out = _leaflet_runtime_payload(mesh, leaflet="out")
    row_records_in = _aggregate_row_records(mesh, payload_in)
    row_records_out = _aggregate_row_records(mesh, payload_out)
    target_shells = _select_target_shells(row_records_in)
    rowwise = {float(shell): {"in": [], "out": []} for shell in target_shells}
    for rec in row_records_in.values():
        shell = float(rec["shell_radius"])
        if shell in rowwise:
            rowwise[shell]["in"].append(rec)
    for rec in row_records_out.values():
        shell = float(rec["shell_radius"])
        if shell in rowwise:
            rowwise[shell]["out"].append(rec)

    shellwise = [
        _shell_stage_summary(
            float(shell),
            in_rows=rowwise[float(shell)]["in"],
            out_rows=rowwise[float(shell)]["out"],
            near_rim=near_rim,
        )
        for shell in target_shells
    ]
    triangles = {
        "in": _triangle_records(payload_in, target_shells),
        "out": _triangle_records(payload_out, target_shells),
    }
    first_departure = _first_material_departure(shellwise)

    diagnosis_call = "local tilt / raw stencil magnitude departure"
    if first_departure["call"] == "geometric_prefactor":
        diagnosis_call = "geometric prefactor magnitude departure"
    elif first_departure["call"] in {"combined_term", "local_contribution"}:
        diagnosis_call = "downstream combined local expression magnitude departure"

    return {
        "case": {
            "theta_B": float(THEORY_THETA_B),
            "outer_radius": float(OUTER_RADIUS),
            "total_energy": float(sum(float(v) for v in result["breakdown"].values())),
        },
        "shell_selection": {
            "target_shell_radii": [float(v) for v in target_shells],
            "selection_rule": "first two outer shells with nonzero inner-leaflet outer-membrane contribution",
        },
        "rim_continuation_reference": {
            key: float(near_rim[key])
            for key in ("theta_b", "theta_outer_in", "theta_outer_out", "phi", "ring_r")
        },
        "shellwise_comparison": shellwise,
        "rowwise_ingredient_audit": {
            str(float(shell)): {
                "in": sorted(
                    rowwise[float(shell)]["in"], key=lambda item: int(item["row"])
                ),
                "out": sorted(
                    rowwise[float(shell)]["out"], key=lambda item: int(item["row"])
                ),
            }
            for shell in target_shells
        },
        "trianglewise_ingredient_audit": {
            str(float(shell)): {
                "in": triangles["in"][str(float(shell))],
                "out": triangles["out"][str(float(shell))],
            }
            for shell in target_shells
        },
        "first_material_magnitude_departure": first_departure,
        "diagnosis": {
            "call": diagnosis_call,
            "recommended_next_stream": (
                "If this remains unfixed, the next stream should isolate the first-two-shell "
                "outer leaflet field continuation that feeds the raw divergence stencil, not "
                "the already-correct sign/base-term path."
            ),
        },
    }


def main() -> None:
    """Run the audit and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.parse_args()
    report = run_curved_1disk_first_two_shell_magnitude_audit()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
