#!/usr/bin/env python3
"""Build fixed-lane theory-parity trend diagnostics against target ratios."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.reproduce_theory_parity import (
    DEFAULT_MESH,
    DEFAULT_OUT,
    DEFAULT_PROTOCOL,
    ROOT,
    _collect_report,
)

DEFAULT_TARGETS = ROOT / "tests" / "fixtures" / "theory_parity_targets.yaml"
DEFAULT_TREND_OUT = (
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "theory_parity_trend.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def compute_ratio_trend(
    *, report: dict[str, Any], targets: dict[str, Any]
) -> dict[str, Any]:
    """Compute ratio deltas against configured target tolerances."""
    rows: dict[str, Any] = {}
    target_ratios = targets["targets"]["ratios"]
    report_ratios = report["metrics"]["theory"]["ratios"]

    within_count = 0
    for name, cfg in target_ratios.items():
        expected = float(cfg["expected"])
        abs_tol = float(cfg["abs_tol"])
        actual = float(report_ratios[name])
        abs_delta = abs(actual - expected)
        within = abs_delta <= abs_tol
        if within:
            within_count += 1
        rows[str(name)] = {
            "actual": actual,
            "expected": expected,
            "abs_tol": abs_tol,
            "abs_delta": abs_delta,
            "within_tolerance": bool(within),
        }

    ratio_count = len(rows)
    return {
        "meta": {
            "fixture": report["meta"]["fixture"],
            "protocol": report["meta"]["protocol"],
            "format": "yaml",
        },
        "summary": {
            "ratio_count": ratio_count,
            "within_tolerance_count": within_count,
            "all_within_tolerance": within_count == ratio_count,
        },
        "ratios": rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument(
        "--protocol",
        nargs="+",
        default=list(DEFAULT_PROTOCOL),
        help="Command sequence used to compute fixed-lane parity report.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=DEFAULT_OUT,
        help="Write intermediate fixed-lane parity report to this YAML path.",
    )
    parser.add_argument(
        "--fixed-polish-steps",
        type=int,
        default=0,
        help="Forwarded to fixed-lane reproducer (default: 0).",
    )
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--out", type=Path, default=DEFAULT_TREND_OUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = tuple(str(x) for x in args.protocol)
    report = _collect_report(
        mesh_path=Path(args.mesh),
        protocol=protocol,
        fixed_polish_steps=int(args.fixed_polish_steps),
    )

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")

    targets = _load_yaml(Path(args.targets))
    trend = compute_ratio_trend(report=report, targets=targets)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(trend, sort_keys=False), encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
