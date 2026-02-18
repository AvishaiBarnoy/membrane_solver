#!/usr/bin/env python3
"""Audit fixed-lane parity stability across mesh refinement levels."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.executor import execute_command_line
from tools.audit_theory_equivalence import build_equivalence_audit
from tools.reproduce_theory_parity import (
    DEFAULT_MESH,
    DEFAULT_PROTOCOL,
    ROOT,
    _build_context,
    _collect_report_from_context,
)

DEFAULT_OUT = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "diagnostics"
    / "theory_mesh_convergence_audit.yaml"
)
RATIO_KEYS = ("theta_ratio", "elastic_ratio", "contact_ratio", "total_ratio")


def _finite(report: dict[str, Any]) -> bool:
    reduced = report["metrics"]["reduced_terms"]
    vals = [
        float(report["metrics"]["final_energy"]),
        float(reduced["elastic_measured"]),
        float(reduced["contact_measured"]),
        float(reduced["total_measured"]),
    ]
    vals.extend(float(report["metrics"]["theory"]["ratios"][k]) for k in RATIO_KEYS)
    return bool(np.all(np.isfinite(np.asarray(vals, dtype=float))))


def _run_level(
    *, mesh_path: Path, refine_level: int, protocol: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any], float]:
    ctx = _build_context(mesh_path)
    t0 = time.perf_counter()
    for _ in range(int(refine_level)):
        execute_command_line(ctx, "r")
    for cmd in protocol:
        execute_command_line(ctx, cmd)
    report = _collect_report_from_context(
        ctx=ctx, mesh_path=mesh_path, protocol=protocol
    )
    runtime = time.perf_counter() - t0
    mesh_meta = {
        "refine_level": int(refine_level),
        "vertex_count": int(len(ctx.mesh.vertices)),
        "triangle_count": int(len(ctx.mesh.facets)),
    }
    return report, mesh_meta, runtime


def build_mesh_convergence_audit(
    *,
    mesh_path: Path,
    refine_levels: list[int],
    protocol: tuple[str, ...],
    ratio_drift_max: float,
    energy_drift_max: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for lvl in sorted(int(x) for x in refine_levels):
        report, mesh_meta, runtime = _run_level(
            mesh_path=mesh_path, refine_level=lvl, protocol=protocol
        )
        eq = build_equivalence_audit(
            report=report,
            radii=(0.35, 7.0 / 15.0, 0.6),
            ratio_tolerance=1.0e-6,
            sum_tolerance=1.0e-9,
        )
        rows.append(
            {
                "level": int(lvl),
                "mesh": mesh_meta,
                "runtime_s": float(runtime),
                "ratios": {
                    k: float(report["metrics"]["theory"]["ratios"][k])
                    for k in RATIO_KEYS
                },
                "total_measured": float(
                    report["metrics"]["reduced_terms"]["total_measured"]
                ),
                "finite": bool(_finite(report)),
                "equivalence_all_pass": bool(eq["summary"]["all_pass"]),
            }
        )

    max_ratio_drift = 0.0
    max_energy_drift = 0.0
    for i in range(len(rows) - 1):
        a = rows[i]
        b = rows[i + 1]
        d_ratio = max(
            abs(float(a["ratios"][k]) - float(b["ratios"][k])) for k in RATIO_KEYS
        )
        d_energy = abs(float(a["total_measured"]) - float(b["total_measured"]))
        max_ratio_drift = max(max_ratio_drift, d_ratio)
        max_energy_drift = max(max_energy_drift, d_energy)

    checks = {
        "all_finite": bool(all(x["finite"] for x in rows)),
        "all_equivalence_pass": bool(all(x["equivalence_all_pass"] for x in rows)),
        "ratio_drift_within_max": bool(max_ratio_drift <= float(ratio_drift_max)),
        "energy_drift_within_max": bool(max_energy_drift <= float(energy_drift_max)),
    }
    return {
        "meta": {
            "fixture": str(mesh_path.relative_to(ROOT)),
            "protocol": list(protocol),
            "format": "yaml",
        },
        "levels": rows,
        "drift": {
            "max_adjacent_ratio_drift": float(max_ratio_drift),
            "max_adjacent_energy_drift": float(max_energy_drift),
            "ratio_drift_max_allowed": float(ratio_drift_max),
            "energy_drift_max_allowed": float(energy_drift_max),
        },
        "checks": checks,
        "summary": {
            "all_pass": bool(all(checks.values())),
            "check_count": len(checks),
            "pass_count": sum(1 for v in checks.values() if v),
        },
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--refine-levels", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--ratio-drift-max", type=float, default=0.20)
    p.add_argument("--energy-drift-max", type=float, default=0.40)
    p.add_argument(
        "--fail-on-check-fail",
        action="store_true",
        help="Exit non-zero when summary.all_pass is false.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    audit = build_mesh_convergence_audit(
        mesh_path=Path(args.mesh),
        refine_levels=[int(x) for x in args.refine_levels],
        protocol=tuple(DEFAULT_PROTOCOL),
        ratio_drift_max=float(args.ratio_drift_max),
        energy_drift_max=float(args.energy_drift_max),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(audit, sort_keys=False), encoding="utf-8")
    print(f"wrote: {out_path}")
    if bool(args.fail_on_check_fail):
        return 0 if bool(audit["summary"]["all_pass"]) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
