#!/usr/bin/env python3
"""Numeric equivalence audit for the fixed-lane theory parity report."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import special

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.reproduce_theory_parity import DEFAULT_OUT, _collect_report

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EQ_OUT = (
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "theory_equivalence_audit.yaml"
)
DEFAULT_RADII = (0.35, 7.0 / 15.0, 0.6)


def _finite(vals: list[float]) -> bool:
    return bool(np.all(np.isfinite(np.asarray(vals, dtype=float))))


def _theory_terms(
    *, radius: float, kappa: float, kappa_t: float, drive: float
) -> dict[str, float]:
    lam = float(np.sqrt(kappa_t / kappa))
    x = float(lam * radius)
    ratio_i = float(special.iv(0, x) / special.iv(1, x))
    ratio_k = float(special.kv(0, x) / special.kv(1, x))
    den = float(ratio_i + 0.5 * ratio_k)
    theta = float(drive / (np.sqrt(kappa * kappa_t) * den))
    fin = float(np.pi * kappa * radius * lam * ratio_i * theta * theta)
    fout = float(np.pi * kappa * radius * lam * 0.5 * ratio_k * theta * theta)
    elastic = float(fin + fout)
    contact = float(-2.0 * np.pi * radius * drive * theta)
    total = float(elastic + contact)
    return {
        "radius": float(radius),
        "thetaB_star": theta,
        "elastic_star": elastic,
        "contact_star": contact,
        "total_star": total,
    }


def _non_decreasing(vals: list[float]) -> bool:
    return all(vals[i] <= vals[i + 1] + 1e-15 for i in range(len(vals) - 1))


def _non_increasing(vals: list[float]) -> bool:
    return all(vals[i] >= vals[i + 1] - 1e-15 for i in range(len(vals) - 1))


def build_equivalence_audit(
    *,
    report: dict[str, Any],
    radii: tuple[float, ...],
    ratio_tolerance: float,
    sum_tolerance: float,
) -> dict[str, Any]:
    reduced = report["metrics"]["reduced_terms"]
    ratios = report["metrics"]["theory"]["ratios"]
    theory = report["metrics"]["theory"]

    theta_ratio = float(ratios["theta_ratio"])
    contact_ratio = float(ratios["contact_ratio"])
    elastic = float(reduced["elastic_measured"])
    contact = float(reduced["contact_measured"])
    total = float(reduced["total_measured"])
    final_energy = float(report["metrics"]["final_energy"])

    checks = {
        "theta_contact_ratio_close": {
            "pass": abs(theta_ratio - contact_ratio) <= float(ratio_tolerance),
            "value": abs(theta_ratio - contact_ratio),
            "tolerance": float(ratio_tolerance),
        },
        "sign_relations": {
            "pass": bool(elastic > 0.0 and contact < 0.0 and total < 0.0),
            "elastic_positive": bool(elastic > 0.0),
            "contact_negative": bool(contact < 0.0),
            "total_negative": bool(total < 0.0),
        },
        "energy_sum_consistency": {
            "pass": abs((elastic + contact) - total) <= float(sum_tolerance),
            "value": abs((elastic + contact) - total),
            "tolerance": float(sum_tolerance),
        },
        "finite_metrics": {
            "pass": _finite(
                [elastic, contact, total, final_energy]
                + [float(v) for v in ratios.values()]
            )
        },
    }

    kappa = float(theory["kappa"])
    kappa_t = float(theory["kappa_t"])
    drive = float(theory["drive"])
    sweep = [
        _theory_terms(radius=float(r), kappa=kappa, kappa_t=kappa_t, drive=drive)
        for r in radii
    ]
    theta_vals = [x["thetaB_star"] for x in sweep]
    elastic_vals = [x["elastic_star"] for x in sweep]
    total_vals = [x["total_star"] for x in sweep]
    checks["radius_sweep_monotonic"] = {
        "pass": bool(
            _non_decreasing(theta_vals)
            and _non_decreasing(elastic_vals)
            and _non_increasing(total_vals)
        ),
        "theta_star_non_decreasing": bool(_non_decreasing(theta_vals)),
        "elastic_star_non_decreasing": bool(_non_decreasing(elastic_vals)),
        "total_star_non_increasing": bool(_non_increasing(total_vals)),
    }

    all_pass = bool(all(bool(v["pass"]) for v in checks.values()))
    return {
        "meta": {
            "fixture": report["meta"]["fixture"],
            "protocol": report["meta"]["protocol"],
            "format": "yaml",
        },
        "checks": checks,
        "radius_sweep": sweep,
        "summary": {
            "all_pass": all_pass,
            "check_count": len(checks),
            "pass_count": sum(1 for v in checks.values() if v["pass"]),
        },
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, default=None)
    p.add_argument("--protocol", nargs="+", default=None)
    p.add_argument("--report-out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--out", type=Path, default=DEFAULT_EQ_OUT)
    p.add_argument("--radii", nargs="+", type=float, default=list(DEFAULT_RADII))
    p.add_argument("--ratio-tol", type=float, default=1.0e-6)
    p.add_argument("--sum-tol", type=float, default=1.0e-9)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mesh is None:
        report = yaml.safe_load(Path(args.report_out).read_text(encoding="utf-8"))
    else:
        protocol = tuple(str(x) for x in (args.protocol or []))
        report = _collect_report(mesh_path=Path(args.mesh), protocol=protocol)
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
        )

    audit = build_equivalence_audit(
        report=report,
        radii=tuple(float(r) for r in args.radii),
        ratio_tolerance=float(args.ratio_tol),
        sum_tolerance=float(args.sum_tol),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(audit, sort_keys=False), encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0 if bool(audit["summary"]["all_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
