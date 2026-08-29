"""Markdown serialization for the thetaB cadence audit."""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.diagnostics.parity_acceptance_triage import _as_float
from tools.diagnostics.utils import format_float as _fmt


def _ranked_hypotheses(report: dict[str, Any]) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    scan_rows = report.get("thetaB_scan_sensitivity_matrix", {}).get("rows", [])
    trap_count = sum(
        1 for row in scan_rows if row.get("classification") == "local_thetaB_scan_trap"
    )
    if trap_count:
        hypotheses.append(
            {
                "label": "local_thetaB_scan_trap",
                "score": int(trap_count),
                "evidence": f"{trap_count} scan cells prefer a wider-grid theta over the local base +/- delta sample.",
            }
        )

    theta_summaries = report.get("fixed_theta_relaxation_matrix", {}).get(
        "theta_summaries", []
    )
    under_relaxed_count = sum(
        1 for row in theta_summaries if row.get("classification") == "under_relaxed"
    )
    if under_relaxed_count:
        hypotheses.append(
            {
                "label": "scan_under_relaxed",
                "score": int(under_relaxed_count),
                "evidence": f"{under_relaxed_count} fixed-theta traces still gain elastic response as the tilt relaxation budget increases.",
            }
        )
    cancellation_count = sum(
        1
        for row in theta_summaries
        if row.get("classification") == "outer_canceled_by_inner"
    )
    if cancellation_count:
        hypotheses.append(
            {
                "label": "coupled_relaxation_cancellation",
                "score": int(cancellation_count),
                "evidence": f"{cancellation_count} fixed-theta traces show outer response peaking before later relaxation passes.",
            }
        )

    line_rows = report.get("line_search_interaction", [])
    if line_rows:
        theta_values = np.asarray(
            [_as_float(row.get("thetaB_value")) for row in line_rows], dtype=float
        )
        if theta_values.size and (
            float(np.max(theta_values) - np.min(theta_values)) > 0.015
        ):
            hypotheses.append(
                {
                    "label": "line_search_reduced_relaxation_interference",
                    "score": 1,
                    "evidence": "Varying line_search_reduced_tilt_inner_steps changes the converged thetaB materially.",
                }
            )

    state_path = _classify_state_path_report(report)
    if bool(state_path.get("fresh_anchor_mismatch")):
        hypotheses.append(
            {
                "label": "fresh_anchor_mismatch",
                "score": 2,
                "evidence": "Fresh fixed-theta replays recover more outer elastic response than optimized-anchor replays at matched theta and budget.",
            }
        )
    if bool(state_path.get("projection_erases_outer_field")):
        hypotheses.append(
            {
                "label": "projection_erases_outer_field",
                "score": 2,
                "evidence": "Outer shell magnitude drops materially while projection remains active and shell participation stays available.",
            }
        )
    if bool(state_path.get("outer_rows_not_free")):
        hypotheses.append(
            {
                "label": "outer_rows_not_free",
                "score": 2,
                "evidence": "Outer shell rows exist but remain fixed during leaflet relaxation.",
            }
        )
    if bool(state_path.get("outer_area_suppressed")):
        hypotheses.append(
            {
                "label": "outer_area_suppressed",
                "score": 2,
                "evidence": "Outer shell rows exist but their effective outer vertex area is zero.",
            }
        )
    if bool(state_path.get("gradient_stall")):
        hypotheses.append(
            {
                "label": "gradient_stall",
                "score": 1,
                "evidence": "Leaflet relaxation stops with high residual gradient while outer response remains small.",
            }
        )

    if not hypotheses:
        hypotheses.append(
            {
                "label": "true_reduced_energy_prefers_low_theta",
                "score": 0,
                "evidence": "No cadence probe produced a stronger alternative to the low-theta optimized state.",
            }
        )
    hypotheses.sort(key=lambda row: int(row.get("score", 0)), reverse=True)
    return hypotheses


def _classify_state_path_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("state_path_comparison_matrix", {}).get("rows", [])
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("requested_thetaB_value")),
            int(row.get("relax_steps", 0)),
        )
        by_key.setdefault(key, {})[str(row.get("warm_start_policy"))] = row

    fresh_anchor_mismatch = False
    projection_erases_outer_field = False
    outer_rows_not_free = False
    outer_area_suppressed = False
    gradient_stall = False
    for policy_rows in by_key.values():
        fresh = policy_rows.get("fresh_fixture")
        anchor = policy_rows.get("anchor_optimized")
        if fresh and anchor:
            fresh_outer = _as_float(
                fresh.get("energy_breakdown", {}).get("tilt_out")
            ) + _as_float(fresh.get("energy_breakdown", {}).get("bending_tilt_out"))
            anchor_outer = _as_float(
                anchor.get("energy_breakdown", {}).get("tilt_out")
            ) + _as_float(anchor.get("energy_breakdown", {}).get("bending_tilt_out"))
            if fresh_outer > max(anchor_outer * 3.0, anchor_outer + 0.01):
                fresh_anchor_mismatch = True
        for row in policy_rows.values():
            relax = row.get("leaflet_relaxation_stats", {})
            part = row.get("outer_participation", {})
            shell = row.get("outer_shell_field", {})
            if (
                int(part.get("outer_shell_row_count", 0)) > 0
                and int(part.get("outer_shell_free_count", 0)) == 0
            ):
                outer_rows_not_free = True
            if (
                int(part.get("outer_shell_row_count", 0)) > 0
                and int(relax.get("active_outer_area_rows", 0)) == 0
            ):
                outer_area_suppressed = True
            if (
                _as_float(relax.get("initial_gradient_norm")) > 1.0e-6
                and _as_float(relax.get("final_gradient_norm"))
                > 0.9 * _as_float(relax.get("initial_gradient_norm"))
                and str(relax.get("stop_reason"))
                in {"line_search_rejected", "completed_max_iters"}
            ):
                gradient_stall = True
            norm_ref = _as_float(relax.get("tilt_projection_norm_ref_outer_far"))
            norm_loss = _as_float(relax.get("tilt_projection_norm_loss_outer_far"))
            if norm_ref > 0.0 and norm_loss > 0.5 * norm_ref:
                projection_erases_outer_field = True
            if (
                int(shell.get("count", 0)) > 0
                and _as_float(shell.get("tilt_out_norm_mean")) < 1.0e-9
                and int(relax.get("outer_shell_row_count", 0)) > 0
            ):
                projection_erases_outer_field = projection_erases_outer_field or (
                    _as_float(relax.get("projection_apply_count")) > 0
                )
    return {
        "fresh_anchor_mismatch": bool(fresh_anchor_mismatch),
        "projection_erases_outer_field": bool(projection_erases_outer_field),
        "outer_rows_not_free": bool(outer_rows_not_free),
        "outer_area_suppressed": bool(outer_area_suppressed),
        "gradient_stall": bool(gradient_stall),
    }


def _classify_report(report: dict[str, Any]) -> dict[str, Any]:
    scan_rows = report.get("thetaB_scan_sensitivity_matrix", {}).get("rows", [])
    theta_summaries = report.get("fixed_theta_relaxation_matrix", {}).get(
        "theta_summaries", []
    )
    line_rows = report.get("line_search_interaction", [])
    classifications = {
        "local_thetaB_scan_trap": any(
            row.get("classification") == "local_thetaB_scan_trap" for row in scan_rows
        ),
        "scan_under_relaxed": any(
            row.get("classification") == "under_relaxed" for row in theta_summaries
        ),
        "coupled_relaxation_cancellation": any(
            row.get("classification") == "outer_canceled_by_inner"
            for row in theta_summaries
        ),
        "line_search_reduced_relaxation_interference": False,
    }
    if line_rows:
        theta_values = np.asarray(
            [_as_float(row.get("thetaB_value")) for row in line_rows], dtype=float
        )
        if theta_values.size:
            classifications["line_search_reduced_relaxation_interference"] = (
                float(np.max(theta_values) - np.min(theta_values)) > 0.015
            )
    classifications.update(_classify_state_path_report(report))
    if not any(classifications.values()):
        classifications["true_reduced_energy_prefers_low_theta"] = True
    return classifications


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# thetaB Cadence / Relaxation Audit",
        "",
        f"- Mode: `{report.get('meta', {}).get('mode', 'unknown')}`",
        f"- Protocol: `{' '.join(str(x) for x in report.get('meta', {}).get('protocol', []))}`",
        "",
        "## Optimized Replay",
        "",
        "| Variant | thetaB | tex ratio | elastic total | tilt_out | bending_tilt_out |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("optimized_trace_replay", []):
        breakdown = row.get("energy_breakdown", {})
        lines.append(
            f"| `{row.get('label')}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('tex_total_ratio'))} | {_fmt(row.get('elastic_total_from_breakdown'))} | {_fmt(breakdown.get('tilt_out'))} | {_fmt(breakdown.get('bending_tilt_out'))} |"
        )

    lines.extend(
        [
            "",
            "## Top Hypotheses",
            "",
        ]
    )
    for row in report.get("ranked_hypotheses", []):
        lines.append(f"- `{row.get('label')}`: {row.get('evidence')}")

    lines.extend(
        [
            "",
            "## Classification",
            "",
        ]
    )
    for key, value in (report.get("classification") or {}).items():
        lines.append(f"- `{key}`: `{bool(value)}`")

    fixed_summaries = report.get("fixed_theta_relaxation_matrix", {}).get(
        "theta_summaries", []
    )
    if fixed_summaries:
        lines.extend(
            [
                "",
                "## Fixed Theta Summary",
                "",
                "| thetaB | classification | high-budget elastic | high-budget tilt_out | high-budget bending_tilt_out |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in fixed_summaries:
            budget_rows = row.get("budget_rows", [])
            last = budget_rows[-1] if budget_rows else {}
            lines.append(
                f"| `{row.get('theta_label')}` | `{row.get('classification')}` | {_fmt(last.get('elastic_total_from_breakdown'))} | {_fmt(last.get('tilt_out'))} | {_fmt(last.get('bending_tilt_out'))} |"
            )

    state_rows = report.get("state_path_comparison_matrix", {}).get("rows", [])
    if state_rows:
        lines.extend(
            [
                "",
                "## State Path Summary",
                "",
                "| policy | thetaB | steps | elastic total | tilt_out | bending_tilt_out | shell tilt_out mean | stop reason | final grad |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for row in state_rows[:16]:
            shell = row.get("outer_shell_field", {})
            relax = row.get("leaflet_relaxation_stats", {})
            breakdown = row.get("energy_breakdown", {})
            lines.append(
                f"| `{row.get('warm_start_policy')}` | {_fmt(row.get('requested_thetaB_value'))} | {int(row.get('relax_steps', 0))} | {_fmt(row.get('elastic_total_from_breakdown'))} | {_fmt(breakdown.get('tilt_out'))} | {_fmt(breakdown.get('bending_tilt_out'))} | {_fmt(shell.get('tilt_out_norm_mean'))} | `{relax.get('stop_reason', 'n/a')}` | {_fmt(relax.get('final_gradient_norm'))} |"
            )

    candidate_rows = report.get("thetaB_candidate_state_delta", [])
    if candidate_rows:
        lines.extend(
            [
                "",
                "## Candidate Delta",
                "",
                "| thetaB | elastic total | shell tilt_out delta | shell radial delta | stop reason |",
                "| ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in candidate_rows:
            delta = row.get("candidate_delta", {})
            relax = row.get("leaflet_relaxation_stats", {})
            lines.append(
                f"| {_fmt(row.get('requested_thetaB_value'))} | {_fmt(row.get('elastic_total_from_breakdown'))} | {_fmt(delta.get('tilt_out_norm_mean_delta'))} | {_fmt(delta.get('tilt_out_radial_mean_delta'))} | `{relax.get('stop_reason', 'n/a')}` |"
            )

    solver_rows = report.get("relaxation_solver_path", [])
    if solver_rows:
        lines.extend(
            [
                "",
                "## Solver Path",
                "",
                "| variant | thetaB | elastic total | tilt_out | bending_tilt_out | grad out shell before | grad out shell after | update out shell | precond out shell | stop reason |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in solver_rows:
            relax = row.get("leaflet_relaxation_stats", {})
            before = relax.get("gradient_norms_before_constraints", {}).get("out", {})
            after = relax.get("gradient_norms_after_constraints", {}).get("out", {})
            updates = relax.get("accepted_update_norms_out", {})
            precond = relax.get("preconditioner_mean_inv_out", {})
            breakdown = row.get("energy_breakdown", {})
            lines.append(
                f"| `{row.get('solver_path_label')}` | {_fmt(row.get('requested_thetaB_value'))} | {_fmt(row.get('elastic_total_from_breakdown'))} | {_fmt(breakdown.get('tilt_out'))} | {_fmt(breakdown.get('bending_tilt_out'))} | {_fmt(before.get('outer_shell'), 6)} | {_fmt(after.get('outer_shell'), 6)} | {_fmt(updates.get('outer_shell'), 6)} | {_fmt(precond.get('outer_shell'))} | `{relax.get('stop_reason', 'n/a')}` |"
            )

    lane_rows = report.get("full_physics_lane_matrix", [])
    if lane_rows:
        lines.extend(
            [
                "",
                "## Full-Physics Lane Matrix",
                "",
                "| lane | intent | ref mode | thetaB | tex ratio | contact ratio | elastic ratio | shell grad | shell update | shell active rows | shell base mean | stop reason |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in lane_rows:
            ratio = row.get("tex_ratio_summary", {})
            combined = row.get("combined_shell_summary", {})
            update = row.get("first_shell_update_summary", {})
            coupling = row.get("bending_coupling_summary", {})
            part = row.get("outer_participation", {})
            relax = row.get("leaflet_relaxation_stats", {})
            lines.append(
                f"| `{row.get('label')}` | `{row.get('model_intent')}` | `{row.get('reference_mode')}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('tex_total_ratio'))} | {_fmt(ratio.get('contact_ratio'))} | {_fmt(ratio.get('elastic_ratio'))} | {_fmt(combined.get('norm'), 6)} | {_fmt(update.get('norm'), 6)} | {int(part.get('outer_shell_free_count', 0))} | {_fmt(coupling.get('base_term_outer_shell_mean'), 6)} | `{relax.get('stop_reason', 'n/a')}` |"
            )

    trace_conv = report.get("full_physics_trace_convergence", {})
    trace_rows = trace_conv.get("rows", [])
    if trace_rows:
        lines.extend(
            [
                "",
                "## Full-Physics Trace Convergence",
                "",
                "| variant | eps | geometry | thetaB | direct_t_out | direct_phi | shell grad | shell update | contact ratio | elastic ratio | classification |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in trace_rows:
            lines.append(
                f"| `{row.get('label')}` | {_fmt(row.get('epsilon'))} | `{row.get('geometry')}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('direct_t_out'))} | {_fmt(row.get('direct_phi'))} | {_fmt(row.get('shell_grad_norm'), 6)} | {_fmt(row.get('shell_update_norm'), 6)} | {_fmt(row.get('contact_ratio'))} | {_fmt(row.get('elastic_ratio'))} | `{row.get('classification')}` |"
            )
        lines.append("")
        lines.append(
            f"- Trace convergence summary: `{trace_conv.get('summary', {}).get('classification', 'n/a')}`"
        )

    scaffold_probe = report.get("full_physics_scaffold_collapse_probe", {})
    scaffold_rows = scaffold_probe.get("rows", [])
    if scaffold_rows:
        lines.extend(
            [
                "",
                "## Full-Physics Scaffold Collapse Probe",
                "",
                "| geometry | variant | thetaB | direct_t_out | trace t_out | support t_out | shell grad | shell update | gd descent | gd best dE | gd trace update | cg fallback | projector | scans | stop reason | classification |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
            ]
        )
        for row in scaffold_rows:
            fields = row.get("role_field_summary", {})
            trace_field = fields.get("trace", {})
            support_field = fields.get("support", {})
            gd_probe = row.get("gd_line_search_probe", {})
            best = gd_probe.get("best_sample", {})
            lines.append(
                f"| `{row.get('geometry')}` | `{row.get('variant')}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('direct_t_out'))} | {_fmt(trace_field.get('tilt_out_radial_mean'))} | {_fmt(support_field.get('tilt_out_radial_mean'))} | {_fmt(row.get('shell_grad_norm'), 6)} | {_fmt(row.get('shell_update_norm'), 6)} | `{bool(gd_probe.get('has_gd_descent_step'))}` | {_fmt(best.get('tilt_dependent_delta'), 6)} | {_fmt(best.get('trace_update_radial_mean'), 6)} | {int(row.get('cg_fallback_accepted_count', 0))} | `{row.get('projector_mode')}` | {int(row.get('thetaB_scan_count', 0))} | `{row.get('stop_reason', 'n/a')}` | `{row.get('classification')}` |"
            )
        lines.append("")
        lines.append(
            f"- Scaffold collapse summary: `{scaffold_probe.get('summary', {}).get('classification', 'n/a')}`"
        )

    support_probe = report.get("full_physics_scaffold_support_ownership_probe", {})
    support_rows = support_probe.get("rows", [])
    if support_rows:
        lines.extend(
            [
                "",
                "## Full-Physics Scaffold Support Ownership",
                "",
                "| geometry | variant | thetaB | direct_t_out | trace t_out | support t_out | trace phi | support phi | trace grad | support grad | trace update | support update | cont dE | stop reason | classification |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in support_rows:
            fields = row.get("role_field_summary", {})
            trace_field = fields.get("trace", {})
            support_field = fields.get("support", {})
            grads = row.get("role_gradient_summary", {})
            updates = row.get("role_update_summary", {})
            continuation = row.get("support_continuation_probe", {}).get(
                "best_sample", {}
            )
            lines.append(
                f"| `{row.get('geometry')}` | `{row.get('variant')}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('direct_t_out'))} | {_fmt(trace_field.get('tilt_out_radial_mean'))} | {_fmt(support_field.get('tilt_out_radial_mean'))} | {_fmt(trace_field.get('phi_to_inner'))} | {_fmt(support_field.get('phi_to_inner'))} | {_fmt(grads.get('trace', {}).get('norm'), 6)} | {_fmt(grads.get('support', {}).get('norm'), 6)} | {_fmt(updates.get('trace', {}).get('norm'), 6)} | {_fmt(updates.get('support', {}).get('norm'), 6)} | {_fmt(continuation.get('tilt_dependent_delta'), 6)} | `{row.get('stop_reason', 'n/a')}` | `{row.get('classification')}` |"
            )
        lines.append("")
        lines.append(
            f"- Support ownership summary: `{support_probe.get('summary', {}).get('classification', 'n/a')}`"
        )

    landscape = report.get("trace_continuation_landscape_probe", {})
    landscape_rows = landscape.get("rows", [])
    if landscape_rows:
        lines.extend(
            [
                "",
                "## Trace Continuation Landscape",
                "",
                "| variant | geometry | mode | alpha | thetaB | trace t_out | support t_out | dE tilt-dependent | dE tilt_out | dE bending_tilt_out | dominant positive |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in landscape_rows:
            theta = row.get("thetaB_value")
            for sample in row.get("samples", []):
                terms = sample.get("term_deltas", {})
                lines.append(
                    f"| `{row.get('variant')}` | `{row.get('geometry')}` | `{sample.get('mode')}` | {_fmt(sample.get('alpha'))} | {_fmt(theta)} | {_fmt(sample.get('trace_radial_before'))} | {_fmt(sample.get('support_radial_before'))} | {_fmt(sample.get('tilt_dependent_delta'), 6)} | {_fmt(terms.get('tilt_out'), 6)} | {_fmt(terms.get('bending_tilt_out'), 6)} | `{sample.get('dominant_positive_term')}` |"
                )
        lines.append("")
        lines.append(
            f"- Landscape summary: dominant suppressing term `{landscape.get('summary', {}).get('most_common_suppressing_term', 'n/a')}`"
        )

    btl_interface = report.get("bending_tilt_out_scaffold_interface_audit", {})
    btl_interface_rows = btl_interface.get("rows", [])
    if btl_interface_rows:
        lines.extend(
            [
                "",
                "## Bending Tilt Out Scaffold Interface",
                "",
                "| variant | geometry | reference | role | thetaB | direct t_out | triangles | area | total | base | divergence | cross | base mean | div mean |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in btl_interface_rows:
            for decomposition in row.get("decompositions", []):
                reference = decomposition.get("reference_mode")
                for role, role_data in decomposition.get("roles", {}).items():
                    lines.append(
                        f"| `{row.get('variant')}` | `{row.get('geometry')}` | `{reference}` | `{role}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('direct_t_out'))} | {int(role_data.get('triangle_count', 0))} | {_fmt(role_data.get('area'), 6)} | {_fmt(role_data.get('total_energy'), 6)} | {_fmt(role_data.get('base_energy'), 6)} | {_fmt(role_data.get('divergence_energy'), 6)} | {_fmt(role_data.get('cross_energy'), 6)} | {_fmt(role_data.get('base_term_mean'), 6)} | {_fmt(role_data.get('divergence_mean'), 6)} |"
                    )
        lines.append("")
        lines.append(
            f"- Interface decomposition summary: largest current-geometry cross role `{btl_interface.get('summary', {}).get('largest_current_geometry_cross_role', 'n/a')}`"
        )

    conditioning = report.get("bending_tilt_out_divergence_conditioning_audit", {})
    conditioning_rows = conditioning.get("rows", [])
    if conditioning_rows:
        lines.extend(
            [
                "",
                "## Bending Tilt Out Divergence Conditioning",
                "",
                "| variant | geometry | role | thetaB | direct t_out | triangles | area mean | min edge | aspect mean | basis max | div abs mean | div max | trace corner | support corner | disk corner |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in conditioning_rows:
            for role, role_data in row.get("conditioning", {}).get("roles", {}).items():
                corner_roles = role_data.get("corner_components_by_row_role", {})
                lines.append(
                    f"| `{row.get('variant')}` | `{row.get('geometry')}` | `{role}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('direct_t_out'))} | {int(role_data.get('triangle_count', 0))} | {_fmt(role_data.get('area', {}).get('mean'), 6)} | {_fmt(role_data.get('min_edge', {}).get('min'), 6)} | {_fmt(role_data.get('aspect', {}).get('mean'), 6)} | {_fmt(role_data.get('basis_norm', {}).get('max_abs'), 6)} | {_fmt(role_data.get('divergence', {}).get('abs_mean'), 6)} | {_fmt(role_data.get('divergence', {}).get('max_abs'), 6)} | {_fmt(corner_roles.get('trace', {}).get('mean'), 6)} | {_fmt(corner_roles.get('support', {}).get('mean'), 6)} | {_fmt(corner_roles.get('disk', {}).get('mean'), 6)} |"
                )
        lines.append("")
        lines.append(
            f"- Divergence conditioning summary: max divergence `{conditioning.get('summary', {}).get('max_divergence_role', 'n/a')}`, max basis `{conditioning.get('summary', {}).get('max_basis_norm_role', 'n/a')}`"
        )

    spacing = report.get("scaffold_geometry_spacing_probe", {})
    spacing_rows = spacing.get("rows", [])
    if spacing_rows:
        lines.extend(
            [
                "",
                "## Scaffold Geometry Spacing Probe",
                "",
                "| label | geometry | div mode | inner shape mode | mesh op | shape fallback | shells | d | thetaB | direct t_out | bending_tilt_out | trace grad radial | trace update radial | shape descent z | fallback accepted | fallback count | fallback dz | dominant down | dominant up | projection 0.15-> | gap to phi | high-tilt replay | high-geom replay | trace div abs | stop |",
                "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in spacing_rows:
            breakdown = row.get("energy_breakdown", {})
            gradient_probe = row.get("gradient_update_probe", {})
            trace_gradient = (
                gradient_probe.get("combined_gradient", {})
                .get("trace", {})
                .get("radial_mean")
            )
            trace_update = (
                gradient_probe.get("first_update", {})
                .get("trace", {})
                .get("radial_mean")
            )
            high_seed = row.get("high_trace_seed_replay", {})
            projection = row.get("high_trace_constraint_projection", {})
            shape_gradient = row.get("shape_gradient_probe", {})
            shape_trace = shape_gradient.get("roles", {}).get("trace", {})
            module_probe = row.get("shape_gradient_module_probe", {})
            down = module_probe.get("dominant_trace_downward_module", {})
            up = module_probe.get("dominant_trace_upward_module", {})
            geometry_seed = row.get("high_trace_geometry_seed_probe", {})
            fallback = row.get("shape_scaffold_rejected_step_fallback_stats", {})
            lines.append(
                f"| `{row.get('label')}` | `{row.get('geometry')}` | `{row.get('interface_divergence_mode', 'p1_triangle')}` | `{row.get('inner_scaffold_shape_stencil_mode', 'off')}` | `{row.get('scaffold_mesh_operation_mode', 'project')}` | `{row.get('shape_scaffold_rejected_step_fallback', 'off')}` | {int(row.get('outer_shells', 0))} | {_fmt(row.get('outer_shells_d'), 4)} | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('direct_t_out'))} | {_fmt(breakdown.get('bending_tilt_out'))} | {_fmt(trace_gradient, 6)} | {_fmt(trace_update, 6)} | {_fmt(shape_trace.get('descent_z_mean'), 6)} | `{fallback.get('accepted', '')}` | {int(fallback.get('accepted_count', 0) or 0)} | {_fmt(fallback.get('trace_dz_mean'), 6)} | `{down.get('module', '')}:{_fmt(down.get('trace_descent_z_mean'), 4)}` | `{up.get('module', '')}:{_fmt(up.get('trace_descent_z_mean'), 4)}` | {_fmt(projection.get('after_trace_t_out'))} | {_fmt(projection.get('after_gap_to_phi'), 6)} | {_fmt(high_seed.get('relaxed_direct_t_out'))} | {_fmt(geometry_seed.get('relaxed_direct_t_out'))} | {_fmt(row.get('trace_div_abs_mean'), 6)} | `{row.get('stop_reason', 'n/a')}` |"
            )
        lines.append("")
        lines.append(
            f"- Spacing summary: best trace divergence `{spacing.get('summary', {}).get('best_trace_divergence_label', 'n/a')}`, best direct t_out `{spacing.get('summary', {}).get('best_direct_t_out_label', 'n/a')}`, best high-tilt replay `{spacing.get('summary', {}).get('best_high_seed_label', 'n/a')}`, best high-geometry replay `{spacing.get('summary', {}).get('best_geometry_seed_label', 'n/a')}`"
        )
        stage_rows = [
            (row.get("label"), replay_row)
            for row in spacing_rows
            for replay_row in (
                row.get("high_trace_stage_replay_probe", {}).get("rows", [])
            )
        ]
        if stage_rows:
            lines.extend(
                [
                    "",
                    "### High-Geometry Stage Replay",
                    "",
                    "| label | iter | stage | thetaB | trace t_out | trace t_in | trace phi | energy | down | up | step ok | step size |",
                    "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |",
                ]
            )
            for label, replay in stage_rows:
                down = replay.get("dominant_down", {})
                up = replay.get("dominant_up", {})
                lines.append(
                    f"| `{label}` | {int(replay.get('iteration', 0))} | `{replay.get('stage')}` | {_fmt(replay.get('thetaB_value'))} | {_fmt(replay.get('trace_t_out'))} | {_fmt(replay.get('trace_t_in'))} | {_fmt(replay.get('trace_phi'))} | {_fmt(replay.get('energy'))} | `{down.get('module', '')}:{_fmt(down.get('trace_descent_z_mean'), 4)}` | `{up.get('module', '')}:{_fmt(up.get('trace_descent_z_mean'), 4)}` | `{replay.get('shape_step_success', '')}` | {_fmt(replay.get('shape_step_size_out'))} |"
                )
        branch_rows = [
            (row.get("label"), branch_row)
            for row in spacing_rows
            for branch_row in (row.get("branch_access_probe", {}).get("rows", []))
        ]
        if branch_rows:
            lines.extend(
                [
                    "",
                    "### Branch Access Probe",
                    "",
                    "| label | state | thetaB | trace t_out | trace phi | tilt descent radial | first update radial | shape descent z | shape up | shape down |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
                ]
            )
            for label, branch in branch_rows:
                tilt_descent = branch.get("tilt_descent_trace", {})
                first_update = branch.get("first_update_trace", {})
                shape_trace = branch.get("shape_trace", {})
                up = branch.get("shape_dominant_up", {})
                down = branch.get("shape_dominant_down", {})
                lines.append(
                    f"| `{label}` | `{branch.get('label')}` | {_fmt(branch.get('thetaB_value'))} | {_fmt(branch.get('trace_t_out'))} | {_fmt(branch.get('trace_phi'))} | {_fmt(tilt_descent.get('radial_mean'), 6)} | {_fmt(first_update.get('radial_mean'), 6)} | {_fmt(shape_trace.get('descent_z_mean'), 6)} | `{up.get('module', '')}:{_fmt(up.get('trace_descent_z_mean'), 4)}` | `{down.get('module', '')}:{_fmt(down.get('trace_descent_z_mean'), 4)}` |"
                )
        trace_z_trial_rows = [
            (row.get("label"), trial)
            for row in spacing_rows
            for trial in (
                row.get("trace_z_fallback_trial_decomposition_probe", {}).get(
                    "samples", []
                )
            )
        ]
        if trace_z_trial_rows:
            lines.extend(
                [
                    "",
                    "### Trace-Z Fallback Trial Decomposition",
                    "",
                    "| label | alpha | constraints | dE | dominant positive | BT in role | BT out role | trace dz kept | trace phi | support phi |",
                    "| --- | ---: | --- | ---: | --- | --- | --- | ---: | ---: | ---: |",
                ]
            )
            for label, trial in trace_z_trial_rows:
                dominant = trial.get("dominant_positive_delta", {})
                bt_roles = trial.get("bending_tilt_role_deltas", {})
                bt_in = bt_roles.get("bending_tilt_in", {})
                bt_out = bt_roles.get("bending_tilt_out", {})
                lines.append(
                    f"| `{label}` | {_fmt(trial.get('alpha'), 6)} | `{trial.get('constraint_context')}` | {_fmt(trial.get('energy_delta'), 6)} | `{dominant.get('module', '')}:{_fmt(dominant.get('delta'), 6)}` | `{bt_in.get('dominant_positive_role', '')}:{_fmt(bt_in.get('dominant_positive_delta'), 6)}` | `{bt_out.get('dominant_positive_role', '')}:{_fmt(bt_out.get('dominant_positive_delta'), 6)}` | {_fmt(trial.get('trace_dz_preserved_ratio'), 6)} | {_fmt(trial.get('trace_phi_after'), 6)} | {_fmt(trial.get('support_phi_after'), 6)} |"
                )

    assembly_rows = report.get("outer_energy_gradient_assembly", [])
    if assembly_rows:
        lines.extend(
            [
                "",
                "## Gradient Assembly",
                "",
                "| thetaB | tilt_out shell grad | bending_tilt_out shell grad | combined shell grad | cosine | kept shell tris | full shell tris | shell base term mean | shell weight mean |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in assembly_rows:
            tilt_mod = row.get("tilt_out_module", {})
            btl_mod = row.get("bending_tilt_out_module", {})
            tri_counts = btl_mod.get("triangle_counts", {})
            combined = row.get("combined_outer_shell_gradient", {})
            lines.append(
                f"| {_fmt(row.get('requested_thetaB_value'))} | {_fmt(tilt_mod.get('tilt_grad_norm_by_region', {}).get('outer_shell'), 6)} | {_fmt(btl_mod.get('tilt_grad_norm_by_region', {}).get('outer_shell'), 6)} | {_fmt(combined.get('norm'), 6)} | {_fmt(combined.get('cosine'))} | {int(tri_counts.get('kept_touching_outer_shell', 0))} | {int(tri_counts.get('full_touching_outer_shell', 0))} | {_fmt(btl_mod.get('base_term_outer_shell_mean'), 6)} | {_fmt(tilt_mod.get('active_row_weight_mean_outer_shell'))} |"
            )

    bridge_rows = report.get("runtime_gradient_bridge", [])
    if bridge_rows:
        lines.extend(
            [
                "",
                "## Runtime Gradient Bridge",
                "",
                "| thetaB | tilt shell grad | bending shell grad | combined shell grad | tilt/bending cosine | runtime shell grad before | runtime shell grad after | first shell update | direct/runtime cosine | before/after cosine | after/update cosine |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in bridge_rows:
            compare = row.get("shell_vector_comparison", {})
            direct_summary = row.get("direct_module_outer_gradient", {})
            direct = direct_summary.get("tilt_grad_norm_by_region", {})
            before = row.get("runtime_aggregated_gradient_before_constraints", {}).get(
                "tilt_grad_norm_by_region", {}
            )
            after = row.get("runtime_aggregated_gradient_after_constraints", {}).get(
                "tilt_grad_norm_by_region", {}
            )
            update = row.get("accepted_update", {}).get("tilt_grad_norm_by_region", {})
            lines.append(
                f"| {_fmt(row.get('requested_thetaB_value'))} | {_fmt(direct_summary.get('tilt_out_shell_norm'), 6)} | {_fmt(direct_summary.get('bending_tilt_out_shell_norm'), 6)} | {_fmt(direct.get('outer_shell'), 6)} | {_fmt(direct_summary.get('tilt_vs_bending_cosine'))} | {_fmt(before.get('outer_shell'), 6)} | {_fmt(after.get('outer_shell'), 6)} | {_fmt(update.get('outer_shell'), 6)} | {_fmt(compare.get('direct_vs_runtime_before_cosine'))} | {_fmt(compare.get('runtime_before_vs_after_cosine'))} | {_fmt(compare.get('runtime_after_vs_update_cosine'))} |"
            )

    reference_rows = report.get("base_term_reference_sweep", [])
    coupling_rows = report.get("outer_coupling_sign_sweep", [])
    if coupling_rows:
        lines.extend(
            [
                "",
                "## Outer Coupling Sign Sweep",
                "",
                "| variant | thetaB | tilt shell grad | bending shell grad | combined shell grad | tilt/bending cosine | base/div cosine | combined radial | combined tangential | shell base mean | shell div mean | shell descent radial |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in coupling_rows:
            tilt_shell = row.get("tilt_shell_summary", {})
            bending_shell = row.get("bending_shell_summary", {})
            combined_shell = row.get("combined_shell_summary", {})
            descent_shell = row.get("descent_shell_update_summary", {})
            coupling = row.get("bending_coupling_summary", {})
            lines.append(
                f"| `{row.get('variant_label')}` | {_fmt(row.get('requested_thetaB_value'))} | {_fmt(tilt_shell.get('norm'), 6)} | {_fmt(bending_shell.get('norm'), 6)} | {_fmt(combined_shell.get('norm'), 6)} | {_fmt(row.get('tilt_vs_bending_cosine'))} | {_fmt(row.get('base_vs_divergence_cosine'))} | {_fmt(combined_shell.get('radial_norm'), 6)} | {_fmt(combined_shell.get('tangential_norm'), 6)} | {_fmt(coupling.get('base_term_outer_shell_mean'), 6)} | {_fmt(coupling.get('div_eval_outer_shell_mean'), 6)} | {_fmt(descent_shell.get('radial_norm'), 6)} |"
            )

    if reference_rows:
        lines.extend(
            [
                "",
                "## Base-Term Reference Sweep",
                "",
                "| variant | thetaB | shell base mean | tilt shell grad | bending shell grad | combined shell grad | cosine | first shell update | tilt_out | bending_tilt_out | contact ratio | elastic ratio | total ratio |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in reference_rows:
            ratio = row.get("tex_ratio_summary", {})
            breakdown = row.get("energy_breakdown", {})
            combined = row.get("combined_outer_shell_gradient", {})
            lines.append(
                f"| `{row.get('variant_label')}` | {_fmt(row.get('requested_thetaB_value'))} | {_fmt(row.get('outer_shell_base_term_mean'), 6)} | {_fmt(row.get('tilt_out_shell_gradient'), 6)} | {_fmt(row.get('bending_tilt_out_shell_gradient'), 6)} | {_fmt(combined.get('norm'), 6)} | {_fmt(combined.get('cosine'))} | {_fmt(row.get('first_accepted_shell_update_norm'), 6)} | {_fmt(breakdown.get('tilt_out'))} | {_fmt(breakdown.get('bending_tilt_out'))} | {_fmt(ratio.get('contact_ratio'))} | {_fmt(ratio.get('elastic_ratio'))} | {_fmt(ratio.get('total_ratio'))} |"
            )

    return "\n".join(lines).rstrip() + "\n"
