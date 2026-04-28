#!/usr/bin/env python3
"""Aggregate diagnosis for the curved one-disk free-membrane miss.

This module intentionally does not change runtime physics.  It turns the
existing curved one-disk benchmark and focused sub-audits into a compact report
that ranks likely root causes for the miss against ``docs/1_disk_3d.tex``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from tools.diagnostics.curved_1disk_theory_benchmark import (
    OUTER_LOG_WINDOW,
    run_curved_1disk_theory_benchmark,
)
from tools.diagnostics.free_disk_profile_protocol import (
    run_free_disk_curved_bilayer_refinement_sweep,
)

CONTROL_VOLUME_THETA = 0.14


def _safe_ratio(numer: float, denom: float) -> float:
    """Return ``numer / denom`` with a finite small-denominator fallback."""
    if abs(float(denom)) <= 1.0e-12:
        return float("inf") if float(numer) else 0.0
    return float(numer) / float(denom)


def _expected_energy_at_selected_theta(
    benchmark: dict[str, object],
) -> dict[str, float]:
    """Return TeX-theory energy split evaluated at selected numeric thetaB."""
    theory = benchmark["theory"]
    theta_opt = float(theory["theta_B_opt"])
    theta_num = float(benchmark["theta_B_selected"])
    theta_sq_ratio = (theta_num / max(abs(theta_opt), 1.0e-12)) ** 2
    theta_ratio = theta_num / max(theta_opt, 1.0e-12)
    inner = float(theory["F_in_el"]) * theta_sq_ratio
    outer = float(theory["F_out_el"]) * theta_sq_ratio
    contact = float(theory["F_cont"]) * theta_ratio
    total = inner + outer + contact
    return {
        "theta_B": theta_num,
        "inner_elastic": inner,
        "outer_elastic": outer,
        "contact": contact,
        "total": total,
    }


def _shape_propagation_evidence(
    benchmark: dict[str, object],
) -> dict[str, object]:
    """Summarize whether curvature/height propagates past the first free shell."""
    radius = float(benchmark["theory"]["radius"])
    shell_rows = list(benchmark.get("shell_rows") or [])
    outer_rows = [row for row in shell_rows if float(row["radius"]) > radius + 1.0e-6]
    nonzero_rows = [
        row for row in outer_rows if abs(float(row.get("z", 0.0))) > 1.0e-12
    ]
    first_nonzero = min((float(row["radius"]) for row in nonzero_rows), default=None)
    last_nonzero = max((float(row["radius"]) for row in nonzero_rows), default=None)
    log_fit = benchmark["fits"]["outer_height_log"]
    log_window_lo = float(OUTER_LOG_WINDOW[0]) * radius
    log_window_hi = float(OUTER_LOG_WINDOW[1]) * radius
    log_window_rows = [
        row
        for row in outer_rows
        if log_window_lo <= float(row["radius"]) <= log_window_hi
    ]
    max_abs_z_log_window = max(
        (abs(float(row.get("z", 0.0))) for row in log_window_rows), default=0.0
    )
    return {
        "outer_shell_count": int(len(outer_rows)),
        "nonzero_outer_z_shell_count": int(len(nonzero_rows)),
        "first_nonzero_outer_z_radius": first_nonzero,
        "last_nonzero_outer_z_radius": last_nonzero,
        "log_window": [float(OUTER_LOG_WINDOW[0]), float(OUTER_LOG_WINDOW[1])],
        "max_abs_z_in_log_window": float(max_abs_z_log_window),
        "slope_fit": float(log_fit["slope_fit"]),
        "slope_ratio": float(log_fit["slope_ratio"]),
        "rel_rmse": float(log_fit["rel_rmse"]),
        "call": (
            "height confined to local support shell"
            if nonzero_rows and max_abs_z_log_window <= 1.0e-12
            else "height propagation inconclusive"
        ),
    }


def _selected_energy_control_case(
    benchmark: dict[str, object],
    energy_control_audit: dict[str, object] | None,
) -> dict[str, object] | None:
    """Return the energy/control audit case nearest the selected benchmark theta."""
    if not energy_control_audit:
        return None
    cases = list(energy_control_audit.get("cases") or [])
    if not cases:
        return None
    theta_selected = float(benchmark["theta_B_selected"])
    return min(cases, key=lambda case: abs(float(case["theta_B"]) - theta_selected))


def _energy_evidence(
    benchmark: dict[str, object],
    energy_control_audit: dict[str, object] | None = None,
) -> dict[str, object]:
    """Summarize energy mismatch at the selected numeric thetaB."""
    expected = _expected_energy_at_selected_theta(benchmark)
    audit_case = _selected_energy_control_case(benchmark, energy_control_audit)
    using_reconciled_audit = (
        audit_case is not None
        and abs(float(audit_case["theta_B"]) - float(benchmark["theta_B_selected"]))
        <= 1.0e-9
    )
    energies = (
        audit_case["numeric_energy_split"]
        if using_reconciled_audit
        else benchmark["energies"]
    )
    ratios = {
        "inner_elastic_numeric_over_tex_at_selected_theta": _safe_ratio(
            float(energies["inner_elastic_numeric"]), expected["inner_elastic"]
        ),
        "outer_elastic_numeric_over_tex_at_selected_theta": _safe_ratio(
            float(energies["outer_elastic_numeric"]), expected["outer_elastic"]
        ),
        "contact_numeric_over_tex_at_selected_theta": _safe_ratio(
            float(energies["contact_numeric"]), expected["contact"]
        ),
        "total_numeric_minus_tex_at_selected_theta": float(energies["total_numeric"])
        - expected["total"],
    }
    legacy = None
    reconciliation = None
    attribution = None
    if using_reconciled_audit:
        legacy = audit_case.get("legacy_numeric_energy_split")
        reconciliation = audit_case.get("runtime_energy_reconciliation")
        attribution = audit_case.get("shell_attribution_coverage")
    return {
        "source": (
            "energy_control_runtime_reconciled"
            if using_reconciled_audit
            else "benchmark_legacy_split"
        ),
        "tex_at_selected_theta": expected,
        "numeric_at_selected_theta": {
            "inner_elastic": float(energies["inner_elastic_numeric"]),
            "outer_elastic": float(energies["outer_elastic_numeric"]),
            "contact": float(energies["contact_numeric"]),
            "total": float(energies["total_numeric"]),
        },
        "legacy_numeric_at_selected_theta": None
        if legacy is None
        else {
            "inner_elastic": float(legacy["inner_elastic_numeric"]),
            "outer_elastic": float(legacy["outer_elastic_numeric"]),
            "contact": float(legacy["contact_numeric"]),
            "total": float(legacy["total_numeric"]),
        },
        "runtime_energy_reconciliation": reconciliation,
        "shell_attribution_coverage": attribution,
        "ratios": ratios,
        "call": (
            "outer elastic overgrowth dominates selected-theta energy"
            if ratios["outer_elastic_numeric_over_tex_at_selected_theta"] > 5.0
            else "legacy split mismatch was diagnostic attribution, not runtime energy"
            if using_reconciled_audit
            and attribution is not None
            and abs(float(attribution.get("unattributed_fraction", 1.0))) <= 1.0e-9
            else "energy split mismatch needs deeper attribution"
        ),
    }


def _control_volume_evidence(
    rows: Sequence[dict[str, float]] | None,
    energy_control_audit: dict[str, object] | None = None,
) -> dict[str, object]:
    """Summarize shared-rim control-volume evidence from refinement sweep rows."""
    if energy_control_audit:
        selected_case = (energy_control_audit.get("cases") or [{}])[0]
        control = selected_case.get("control_volume") or {}
        ratios = control.get("ratios") or {}
        concentration = selected_case.get("shell_concentration") or {}
        return {
            "available": True,
            "theta_B": float(selected_case.get("theta_B", CONTROL_VOLUME_THETA)),
            "outer_control_over_annulus": float(
                ratios.get("outer_control_over_gap_annulus", 0.0)
            ),
            "rim_control_over_annulus": float(
                ratios.get("rim_control_over_gap_annulus", 0.0)
            ),
            "outer_control_over_adjacent_shell": float(
                ratios.get("outer_control_over_adjacent_shell", 0.0)
            ),
            "rim_control_over_adjacent_shell": float(
                ratios.get("rim_control_over_adjacent_shell", 0.0)
            ),
            "support_fraction_of_outer_shell_elastic": float(
                concentration.get("support_fraction_of_outer_shell_elastic", 0.0)
            ),
            "first_two_fraction_of_outer_shell_elastic": float(
                concentration.get("first_two_fraction_of_outer_shell_elastic", 0.0)
            ),
            "call": str(control.get("call") or "energy/control audit supplied"),
        }
    if not rows:
        return {
            "available": False,
            "call": "not run",
            "theta_B": CONTROL_VOLUME_THETA,
        }
    row = dict(rows[0])
    outer_ratio = _safe_ratio(
        float(row.get("outer_control_area", 0.0)),
        float(row.get("outer_annulus_area", 0.0)),
    )
    rim_ratio = _safe_ratio(
        float(row.get("rim_control_area", 0.0)),
        float(row.get("rim_annulus_area", 0.0)),
    )
    return {
        "available": True,
        "theta_B": float(row.get("theta_b", CONTROL_VOLUME_THETA)),
        "outer_control_area": float(row.get("outer_control_area", 0.0)),
        "outer_annulus_area": float(row.get("outer_annulus_area", 0.0)),
        "outer_control_over_annulus": outer_ratio,
        "rim_control_area": float(row.get("rim_control_area", 0.0)),
        "rim_annulus_area": float(row.get("rim_annulus_area", 0.0)),
        "rim_control_over_annulus": rim_ratio,
        "outer_shell_area": float(row.get("outer_shell_area", 0.0)),
        "rim_shell_area": float(row.get("rim_shell_area", 0.0)),
        "call": (
            "shared-rim support rows carry oversized shell control volumes"
            if outer_ratio > 4.0 or rim_ratio > 2.0
            else "shared-rim control volumes are not the dominant evidence"
        ),
    }


def _target_direction_evidence(
    phi_target_audit: dict[str, object] | None,
    shell2_audit: dict[str, object] | None,
) -> dict[str, object]:
    """Summarize target-direction and shell-2 continuation evidence."""
    out: dict[str, object] = {
        "available": bool(phi_target_audit or shell2_audit),
        "phi_target_call": None,
        "shell2_departure_call": None,
        "target_direction_cos_global_radial": None,
        "call": "not run",
    }
    if phi_target_audit:
        diagnosis = phi_target_audit.get("diagnosis") or {}
        target = (phi_target_audit.get("shell_target_construction") or {}).get(
            "target_direction"
        ) or {}
        out["phi_target_call"] = diagnosis.get("call")
        out["target_direction_cos_global_radial"] = target.get(
            "r_dir_cos_global_radial_median"
        )
    if shell2_audit:
        departure = shell2_audit.get("first_material_departure") or {}
        out["shell2_departure_call"] = departure.get("call")
        out["shell2_departure_shell_radius"] = departure.get("shell_radius")

    phi_call = str(out.get("phi_target_call") or "")
    shell_call = str(out.get("shell2_departure_call") or "")
    rdir = out.get("target_direction_cos_global_radial")
    if rdir is not None and float(rdir) < -0.5:
        call = "target radial direction points inward"
    elif (
        rdir is not None
        and float(rdir) > 0.5
        and phi_call == "target direction outward"
    ):
        call = "target direction fixed; inspect energy/profile residuals"
    elif phi_call and phi_call != "another specific target-construction defect":
        call = phi_call
    elif shell_call == "no shell-2 tilt-out departure":
        call = "target direction fixed; inspect energy/profile residuals"
    elif shell_call:
        call = f"shell-2 continuation departure: {shell_call}"
    else:
        call = "target-direction evidence unavailable"
    out["call"] = call
    return out


def _failure_explanations(
    benchmark: dict[str, object],
    *,
    shape_evidence: dict[str, object],
    energy_evidence: dict[str, object],
) -> dict[str, object]:
    """Map each benchmark failure group to a direct interpretation."""
    theory = benchmark["theory"]
    fits = benchmark["fits"]
    near_rim = benchmark["near_rim"]
    energies = energy_evidence
    theta_num = float(benchmark["theta_B_selected"])
    theta_theory = float(theory["theta_B_opt"])
    outer_ratio = energies["ratios"]["outer_elastic_numeric_over_tex_at_selected_theta"]
    energy_source = str(energies.get("source") or "benchmark_legacy_split")
    return {
        "theta_B_opt": {
            "numeric": theta_num,
            "tex_target": theta_theory,
            "ratio_numeric_over_target": _safe_ratio(theta_num, theta_theory),
            "interpretation": (
                "the local scan still selects a low-theta branch; after runtime "
                "energy reconciliation the severe legacy split is gone, so the "
                "remaining miss points back to profile propagation and moderate "
                "elastic overgrowth"
                if energy_source == "energy_control_runtime_reconciled"
                else "the local scan selects a low-theta branch because the realized "
                "outer elastic cost is far above the TeX quadratic cost"
            ),
        },
        "outer_height_log_fit": {
            "slope_fit": float(fits["outer_height_log"]["slope_fit"]),
            "slope_ratio": float(fits["outer_height_log"]["slope_ratio"]),
            "shape_propagation_call": shape_evidence["call"],
            "interpretation": (
                "outer height is flat through the log-fit window, so the "
                "tensionless trumpet is not propagating beyond the local support ring"
            ),
        },
        "outer_k1_fit": {
            "lambda_fit": float(fits["outer_k1"]["lambda_fit"]),
            "lambda_theory": float(theory["lambda_theory"]),
            "lambda_ratio": float(fits["outer_k1"]["lambda_ratio"]),
            "leaflet_mismatch_median": float(
                fits["outer_k1"]["leaflet_mismatch_median"]
            ),
            "interpretation": (
                "outer leaflets match each other in the far window, but the decay "
                "length is too long and the near shell changes sign before the K1 tail"
            ),
        },
        "inner_i1_fit": {
            "lambda_fit": float(fits["inner_i1"]["lambda_fit"]),
            "lambda_theory": float(theory["lambda_theory"]),
            "lambda_ratio": float(fits["inner_i1"]["lambda_ratio"]),
            "rel_rmse": float(fits["inner_i1"]["rel_rmse"]),
            "interpretation": (
                "the disk-side profile is not the TeX I1 mode; it is distorted by "
                "the current shared-rim/local-shell continuation"
            ),
        },
        "energy_split": {
            "source": energy_source,
            "outer_elastic_numeric_over_tex_at_selected_theta": outer_ratio,
            "contact_numeric_over_tex_at_selected_theta": energies["ratios"][
                "contact_numeric_over_tex_at_selected_theta"
            ],
            "shell_unattributed_outer_fraction": None
            if energies.get("shell_attribution_coverage") is None
            else float(energies["shell_attribution_coverage"]["unattributed_fraction"]),
            "theta_out_over_half_theta_B": float(
                near_rim["theta_out_over_half_theta_B"]
            ),
            "interpretation": (
                "Gate A reconciles the diagnostic split to runtime module totals: "
                "contact work is consistent with the selected thetaB, shell-local "
                "outer attribution closes, and the remaining elastic overgrowth is "
                "moderate rather than the old severe outer/inner split"
                if energy_source == "energy_control_runtime_reconciled"
                else "contact work is consistent with the selected thetaB, but selected "
                "thetaB is too small and outer elastic overgrowth dominates the split"
            ),
        },
    }


def _rank_candidate_causes(
    *,
    shape_evidence: dict[str, object],
    target_evidence: dict[str, object],
    energy_evidence: dict[str, object],
    control_evidence: dict[str, object],
    energy_control_audit: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Return ranked root-cause candidates with compact evidence."""
    outer_energy_ratio = float(
        energy_evidence["ratios"]["outer_elastic_numeric_over_tex_at_selected_theta"]
    )
    shape_blocked = shape_evidence["call"] == "height confined to local support shell"
    control_oversized = control_evidence.get("available") and (
        float(control_evidence["outer_control_over_annulus"]) > 4.0
        or float(control_evidence["rim_control_over_annulus"]) > 2.0
    )
    support_localized = control_evidence.get("available") and (
        float(control_evidence.get("support_fraction_of_outer_shell_elastic", 0.0))
        > 0.5
    )
    target_call = str(target_evidence.get("call") or "")
    target_suspicious = target_call not in {
        "not run",
        "target-direction evidence unavailable",
        "target direction fixed; inspect energy/profile residuals",
    }
    target_fixed = (
        target_call == "target direction fixed; inspect energy/profile residuals"
    )
    energy_control_causes = []
    if energy_control_audit is not None:
        energy_control_causes = list(energy_control_audit.get("root_causes_ranked", []))
    energy_control_top = energy_control_causes[0] if energy_control_causes else None
    energy_control_top_score = (
        int(energy_control_top.get("rank_score", 0))
        if isinstance(energy_control_top, dict)
        else 0
    )

    audit_conclusion = (
        "The deeper audit now reconciles shell-local outer attribution to runtime "
        "module totals; the previous large unattributed outer fraction was a "
        "diagnostic split bug, not evidence for a runtime energy-ownership defect."
    )
    audit_stream = (
        "treat energy ownership as reconciled for now and prioritize the highest "
        "remaining audit/shape cause before changing runtime physics"
    )
    if isinstance(energy_control_top, dict):
        audit_conclusion = (
            f"The deeper audit ranks {energy_control_top.get('cause')} highest "
            "after runtime-module reconciliation; use its evidence before any "
            "runtime energy/sign change."
        )
        audit_stream = str(energy_control_top.get("recommended_stream") or audit_stream)

    candidates = [
        {
            "cause": "reconciled energy/control audit residuals",
            "rank_score": energy_control_top_score if energy_control_top_score else 35,
            "evidence": {
                "energy_control_audit": {
                    "available": energy_control_audit is not None,
                    "root_causes_ranked": energy_control_causes,
                },
            },
            "conclusion": audit_conclusion,
            "smallest_next_fix_stream": audit_stream,
        },
        {
            "cause": "curvature generation does not propagate",
            "rank_score": 100 if shape_blocked else 30,
            "evidence": shape_evidence,
            "conclusion": (
                "The first active support shell moves, but the log-window shells stay flat."
            ),
            "smallest_next_fix_stream": (
                "isolate shape-side constraints/projection under fixed thetaB"
            ),
        },
        {
            "cause": "excess shared-rim/local-shell elastic cost",
            "rank_score": int(min(95.0, 20.0 + 2.0 * outer_energy_ratio))
            + (20 if control_oversized else 0)
            + (15 if support_localized else 0),
            "evidence": {
                "energy": energy_evidence,
                "control_volume": control_evidence,
                "energy_control_audit": {
                    "available": energy_control_audit is not None,
                    "root_causes_ranked": []
                    if energy_control_audit is None
                    else energy_control_audit.get("root_causes_ranked", []),
                },
            },
            "conclusion": (
                "At selected thetaB the outer elastic term is much larger than "
                "the TeX value, while contact work itself is not the limiting term."
            ),
            "smallest_next_fix_stream": (
                "audit/control shared-rim support-volume ownership before changing physics"
            ),
        },
        {
            "cause": "wrong rim/shell target direction or shell-2 continuation",
            "rank_score": 80 if target_suspicious else (10 if target_fixed else 25),
            "evidence": target_evidence,
            "conclusion": (
                "The dedicated target-direction audit no longer reports the "
                "inward radial target; remaining failures are energy/profile residuals."
                if target_fixed
                else (
                    "Dedicated target-direction audits should drive the next runtime "
                    "fix if they report inward radial targets or shell-2 departure."
                )
            ),
            "smallest_next_fix_stream": (
                "inspect energy/profile residuals before another target-direction fix"
                if target_fixed
                else "fix shell-2 target construction only after a Feature Contract"
            ),
        },
    ]
    return sorted(candidates, key=lambda row: int(row["rank_score"]), reverse=True)


def run_curved_1disk_miss_diagnosis(
    *,
    benchmark_report: dict[str, object] | None = None,
    phi_target_audit: dict[str, object] | None = None,
    shell2_audit: dict[str, object] | None = None,
    control_volume_rows: Sequence[dict[str, float]] | None = None,
    energy_control_audit: dict[str, object] | None = None,
    include_target_audits: bool = True,
    include_control_volume: bool = True,
    include_energy_control_audit: bool = True,
) -> dict[str, object]:
    """Run or aggregate diagnostics into one ranked miss report."""
    benchmark = benchmark_report or run_curved_1disk_theory_benchmark()
    if include_target_audits and phi_target_audit is None:
        from tools.diagnostics.curved_1disk_shared_rim_phi_target_audit import (
            run_curved_1disk_shared_rim_phi_target_audit,
        )

        phi_target_audit = run_curved_1disk_shared_rim_phi_target_audit()
    if include_target_audits and shell2_audit is None:
        from tools.diagnostics.curved_1disk_shell2_tiltout_audit import (
            run_curved_1disk_shell2_tiltout_audit,
        )

        shell2_audit = run_curved_1disk_shell2_tiltout_audit()
    if include_control_volume and control_volume_rows is None:
        control_volume_rows = run_free_disk_curved_bilayer_refinement_sweep(
            [CONTROL_VOLUME_THETA],
            refine_steps=0,
            shape_steps=10,
        )
    if include_energy_control_audit and energy_control_audit is None:
        from tools.diagnostics.curved_1disk_energy_control_volume_audit import (
            SELECTED_THETA_B_AFTER_SHARED_RIM_FIX,
            THEORY_THETA_B,
            run_curved_1disk_energy_control_volume_audit,
        )

        theta_values = (
            float(benchmark["theta_B_selected"])
            if benchmark_report is None
            else SELECTED_THETA_B_AFTER_SHARED_RIM_FIX,
            THEORY_THETA_B,
        )
        energy_control_audit = run_curved_1disk_energy_control_volume_audit(
            theta_values
        )

    shape = _shape_propagation_evidence(benchmark)
    control = _control_volume_evidence(control_volume_rows, energy_control_audit)
    target = _target_direction_evidence(phi_target_audit, shell2_audit)
    energy = _energy_evidence(benchmark, energy_control_audit)
    failures = _failure_explanations(
        benchmark,
        shape_evidence=shape,
        energy_evidence=energy,
    )
    candidates = _rank_candidate_causes(
        shape_evidence=shape,
        target_evidence=target,
        energy_evidence=energy,
        control_evidence=control,
        energy_control_audit=energy_control_audit,
    )
    return {
        "title": "Curved 1-disk free-membrane miss diagnosis",
        "scope": {
            "diagnosis_only": True,
            "runtime_physics_changed": False,
            "reference": "docs/1_disk_3d.tex",
        },
        "benchmark_summary": {
            "lock_passed": bool(benchmark["benchmark_lock_passed"]),
            "lock_failures": list(benchmark["benchmark_lock_failures"]),
            "theta_B_selected": float(benchmark["theta_B_selected"]),
            "theta_B_tex": float(benchmark["theory"]["theta_B_opt"]),
            "total_numeric": float(benchmark["energies"]["total_numeric"]),
            "total_tex": float(benchmark["theory"]["F_tot"]),
        },
        "failure_explanations": failures,
        "energy_control_volume_audit": energy_control_audit,
        "candidate_causes_ranked": candidates,
        "recommended_next_pr": {
            "feature_contract_required": True,
            "reason": (
                "Any fix will change constraints, energy accounting, or "
                "theory-facing numerical behavior."
            ),
            "recommended_stream": str(candidates[0]["smallest_next_fix_stream"]),
        },
    }


def main() -> None:
    """Run the aggregate diagnosis and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-target-audits",
        action="store_true",
        help="Only aggregate benchmark/control-volume evidence.",
    )
    parser.add_argument(
        "--skip-control-volume",
        action="store_true",
        help="Skip the shared-rim control-volume refinement sweep.",
    )
    args = parser.parse_args()
    report = run_curved_1disk_miss_diagnosis(
        include_target_audits=not args.skip_target_audits,
        include_control_volume=not args.skip_control_volume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
