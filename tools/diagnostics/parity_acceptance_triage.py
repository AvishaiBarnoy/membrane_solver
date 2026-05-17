#!/usr/bin/env python3
"""Triage known theory-parity acceptance failures without changing solver state."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tools.diagnostics.scaffold_energy_imbalance_audit import (  # noqa: E402
    _base_term_summary_for_fixture,
)
from tools.reproduce_theory_parity import (  # noqa: E402
    DEFAULT_PROTOCOL,
    _build_context,
    _collect_report_from_context,
    _run_protocol_with_parity_activation,
)
from tools.theory_parity_interface_profiles import build_profiled_fixture  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "reproduce_theory_parity.py"
BASE_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
DEFAULT_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml"
)
GHOST_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_ghost_eps005.yaml"
)
DEFAULT_BASELINE = (
    ROOT / "tests" / "fixtures" / "theory_parity_physical_edge_default_baseline.yaml"
)
GHOST_BAD_BRANCH_BASELINE = {
    "direct_t_out": 0.0,
    "direct_phi": 0.0,
    "free_inner_vs_free_outer_director_gap": 0.008275776127628786,
    "thetaB_value": 0.21000000000000005,
}


def _get_path(dct: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = dct
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _run_report(
    mesh_path: Path, protocol: tuple[str, ...] = DEFAULT_PROTOCOL
) -> dict[str, Any]:
    ctx = _build_context(mesh_path)
    _run_protocol_with_parity_activation(ctx, protocol=protocol)
    return _collect_report_from_context(ctx=ctx, mesh_path=mesh_path, protocol=protocol)


def _write_temp_fixture(doc: dict[str, Any], directory: Path, label: str) -> Path:
    path = directory / f"{label}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def _build_physical_edge_profile_fixture(profile: str, lane: str) -> dict[str, Any]:
    base_doc = yaml.safe_load(BASE_FIXTURE.read_text(encoding="utf-8")) or {}
    doc = build_profiled_fixture(base_doc=base_doc, profile=profile, lane=lane)
    gp = dict(doc.get("global_parameters") or {})
    gp["rim_slope_match_mode"] = "physical_edge_staggered_v1"
    gp["tilt_solver"] = "cg"
    gp["tilt_cg_max_iters"] = 120
    gp["tilt_mass_mode_in"] = "consistent"
    doc["global_parameters"] = gp
    constraints = [str(x) for x in (doc.get("constraint_modules") or [])]
    doc["constraint_modules"] = [
        x for x in constraints if x != "tilt_thetaB_boundary_in"
    ]
    return doc


def _assertion(
    *,
    case: str,
    metric_path: str,
    condition: str,
    actual: float | None = None,
    expected: float | None = None,
    baseline: float | None = None,
) -> dict[str, Any]:
    passed = None
    if actual is not None and expected is not None:
        if condition == ">":
            passed = bool(actual > expected)
        elif condition == "<":
            passed = bool(actual < expected)
        elif condition == "abs<":
            passed = bool(abs(actual) < expected)
    return {
        "case": case,
        "metric_path": metric_path,
        "condition": condition,
        "actual": actual,
        "expected": expected,
        "baseline": baseline,
        "passed": passed,
    }


def _interface_summary(report: dict[str, Any]) -> dict[str, Any]:
    diagnostics = report["metrics"]["diagnostics"]
    split = diagnostics.get("outer_split", {})
    traces = diagnostics.get("interface_traces_at_R", {})
    shell = diagnostics.get("interface_shell_at_R_plus_epsilon", {})
    primary = diagnostics.get("interface_primary_readout", {})
    directors = diagnostics.get("interface_directors", {})
    geometry = diagnostics.get("outer_shell_geometry", {})
    scans = diagnostics.get("thetaB_scan_trace", [])
    return {
        "delta_r": _as_float(geometry.get("delta_r")),
        "target_source": str(split.get("target_source", "")),
        "primary_source": str(primary.get("source", "")),
        "phi_mean": _as_float(split.get("phi_mean")),
        "t_in_mean": _as_float(split.get("t_in_mean")),
        "t_out_mean": _as_float(split.get("t_out_mean")),
        "trace_t_in": _as_float(traces.get("outer_t_in_trace_at_R_plus")),
        "trace_t_out": _as_float(traces.get("outer_t_out_trace_at_R_plus")),
        "direct_t_in": _as_float(shell.get("t_in_at_R_plus_epsilon")),
        "direct_t_out": _as_float(shell.get("t_out_at_R_plus_epsilon")),
        "direct_phi": _as_float(shell.get("phi_secant_at_R_plus_epsilon")),
        "disk_vs_free_inner_director_gap": _as_float(
            directors.get("disk_vs_free_inner_director_gap")
        ),
        "free_inner_vs_free_outer_director_gap": _as_float(
            directors.get("free_inner_vs_free_outer_director_gap")
        ),
        "thetaB_scan_count": len(scans) if isinstance(scans, list) else 0,
        "thetaB_scan_tail": scans[-5:] if isinstance(scans, list) else [],
    }


def _schema_only() -> dict[str, Any]:
    cases = [
        "ghost_shell_direct_interface",
        "generated_family_smoothness",
        "default_free_side_trace_continuation",
        "default_director_profile_parity",
    ]
    return {
        "meta": {"mode": "schema"},
        "cases": [{"case": case, "status": "not_run"} for case in cases],
        "assertions": [
            _assertion(
                case="ghost_shell_direct_interface",
                metric_path="diagnostics.interface_shell_at_R_plus_epsilon.phi_secant_at_R_plus_epsilon",
                condition=">",
            ),
            _assertion(
                case="generated_family_smoothness",
                metric_path="reports.default_lo.metrics.diagnostics.outer_split.t_out_mean",
                condition="abs<",
            ),
            _assertion(
                case="default_free_side_trace_continuation",
                metric_path="metrics.diagnostics.interface_traces_at_R.outer_t_in_trace_at_R_plus",
                condition=">",
            ),
            _assertion(
                case="default_director_profile_parity",
                metric_path="metrics.diagnostics.interface_directors.disk_vs_free_inner_director_gap",
                condition="<",
            ),
        ],
    }


def run_triage(*, mode: str = "run") -> dict[str, Any]:
    if mode == "schema":
        return _schema_only()

    baseline = yaml.safe_load(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    ghost = _run_report(GHOST_FIXTURE)
    default = _run_report(DEFAULT_FIXTURE)
    ghost_base_term = _base_term_summary_for_fixture(GHOST_FIXTURE, "ghost")
    default_base_term = _base_term_summary_for_fixture(DEFAULT_FIXTURE, "default")
    with tempfile.TemporaryDirectory(prefix="parity_triage_") as tmp:
        tmpdir = Path(tmp)
        family: dict[str, dict[str, Any]] = {}
        family_base_terms: dict[str, dict[str, Any]] = {}
        for label in ("default_lo", "default", "default_hi"):
            path = _write_temp_fixture(
                _build_physical_edge_profile_fixture(label, label), tmpdir, label
            )
            family[label] = _run_report(path)
            family_base_terms[label] = _base_term_summary_for_fixture(path, label)

    assertions: list[dict[str, Any]] = []
    shell = ghost["metrics"]["diagnostics"]["interface_shell_at_R_plus_epsilon"]
    assertions.append(
        _assertion(
            case="ghost_shell_direct_interface",
            metric_path="metrics.diagnostics.interface_shell_at_R_plus_epsilon.phi_secant_at_R_plus_epsilon",
            condition=">",
            actual=_as_float(shell.get("phi_secant_at_R_plus_epsilon")),
            expected=1.0e-4,
            baseline=GHOST_BAD_BRANCH_BASELINE["direct_phi"],
        )
    )
    assertions.append(
        _assertion(
            case="generated_family_smoothness",
            metric_path="reports.default_lo.metrics.diagnostics.outer_split.t_out_mean",
            condition="abs<",
            actual=_as_float(
                family["default_lo"]["metrics"]["diagnostics"]["outer_split"].get(
                    "t_out_mean"
                )
            ),
            expected=1.0e-6,
        )
    )
    baseline_traces = baseline["metrics"]["diagnostics"]["interface_traces_at_R"]
    current_traces = default["metrics"]["diagnostics"]["interface_traces_at_R"]
    baseline_t_in = _as_float(baseline_traces.get("outer_t_in_trace_at_R_plus"))
    assertions.append(
        _assertion(
            case="default_free_side_trace_continuation",
            metric_path="metrics.diagnostics.interface_traces_at_R.outer_t_in_trace_at_R_plus",
            condition=">",
            actual=_as_float(current_traces.get("outer_t_in_trace_at_R_plus")),
            expected=baseline_t_in + 0.02,
            baseline=baseline_t_in,
        )
    )
    baseline_directors = baseline["metrics"]["diagnostics"]["interface_directors"]
    current_directors = default["metrics"]["diagnostics"]["interface_directors"]
    baseline_gap = _as_float(baseline_directors.get("disk_vs_free_inner_director_gap"))
    assertions.append(
        _assertion(
            case="default_director_profile_parity",
            metric_path="metrics.diagnostics.interface_directors.disk_vs_free_inner_director_gap",
            condition="<",
            actual=_as_float(current_directors.get("disk_vs_free_inner_director_gap")),
            expected=baseline_gap - 0.02,
            baseline=baseline_gap,
        )
    )
    return {
        "meta": {"mode": "run", "protocol": list(DEFAULT_PROTOCOL)},
        "cases": [
            {
                "case": "ghost_shell_direct_interface",
                "thetaB_value": _as_float(ghost["metrics"].get("thetaB_value")),
                "tex_total_ratio": _as_float(
                    ghost["metrics"]["tex_benchmark"]["ratios"].get("total_ratio")
                ),
                "interface_summary": _interface_summary(ghost),
                "base_term_summary": ghost_base_term,
            },
            {
                "case": "generated_family_smoothness",
                "labels": {
                    label: {
                        "thetaB_value": _as_float(
                            report["metrics"].get("thetaB_value")
                        ),
                        "tex_total_ratio": _as_float(
                            report["metrics"]["tex_benchmark"]["ratios"].get(
                                "total_ratio"
                            )
                        ),
                        "interface_summary": _interface_summary(report),
                        "base_term_summary": family_base_terms.get(label, {}),
                    }
                    for label, report in family.items()
                },
            },
            {
                "case": "default_free_side_trace_continuation",
                "thetaB_value": _as_float(default["metrics"].get("thetaB_value")),
                "interface_summary": _interface_summary(default),
                "base_term_summary": default_base_term,
            },
            {
                "case": "default_director_profile_parity",
                "thetaB_value": _as_float(default["metrics"].get("thetaB_value")),
                "interface_summary": _interface_summary(default),
                "base_term_summary": default_base_term,
            },
        ],
        "assertions": assertions,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--mode", choices=("run", "schema"), default="run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    triage = run_triage(mode=str(args.mode))
    text = yaml.safe_dump(triage, sort_keys=False)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
