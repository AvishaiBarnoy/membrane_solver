#!/usr/bin/env python3
"""Run controlled parameter sweeps and audit theory-consistent trend directions."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.reproduce_theory_parity import (
    DEFAULT_MESH,
    DEFAULT_PROTOCOL,
    ROOT,
    _collect_report,
)

DEFAULT_OUT = (
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "theory_scaling_audit.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _non_decreasing(vals: list[float], eps: float) -> bool:
    return all(vals[i] <= vals[i + 1] + eps for i in range(len(vals) - 1))


def _non_increasing(vals: list[float], eps: float) -> bool:
    return all(vals[i] >= vals[i + 1] - eps for i in range(len(vals) - 1))


def _run_with_params(
    *,
    base_geom: dict[str, Any],
    mesh_name: str,
    protocol: tuple[str, ...],
    overrides: dict[str, float],
) -> tuple[dict[str, Any], float]:
    geom = copy.deepcopy(base_geom)
    gp = geom.setdefault("global_parameters", {})
    gp.update(overrides)
    t0 = time.perf_counter()
    temp_root = ROOT / "benchmarks" / "outputs" / "diagnostics"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="theory_scaling_", dir=str(temp_root)
    ) as td:
        path = Path(td) / mesh_name
        path.write_text(yaml.safe_dump(geom, sort_keys=False), encoding="utf-8")
        report = _collect_report(mesh_path=path, protocol=protocol)
    dt = time.perf_counter() - t0
    return report, dt


def _extract_run(
    *, key: str, value: float, report: dict[str, Any], runtime_s: float
) -> dict[str, Any]:
    ratios = report["metrics"]["theory"]["ratios"]
    reduced = report["metrics"]["reduced_terms"]
    return {
        "parameter": key,
        "value": float(value),
        "runtime_s": float(runtime_s),
        "thetaB_value": float(report["metrics"]["thetaB_value"]),
        "abs_total_measured": abs(float(reduced["total_measured"])),
        "ratios": {k: float(v) for k, v in ratios.items()},
    }


def build_scaling_audit(
    *,
    mesh: Path,
    protocol: tuple[str, ...],
    drive_values: list[float],
    kt_scales: list[float],
    monotonic_eps: float,
) -> dict[str, Any]:
    base_geom = _load_yaml(mesh)
    mesh_name = mesh.name
    rows_drive: list[dict[str, Any]] = []
    rows_kt: list[dict[str, Any]] = []

    base_gp = base_geom.get("global_parameters", {})
    base_kt_in = float(base_gp.get("tilt_modulus_in") or 0.0)
    base_kt_out = float(base_gp.get("tilt_modulus_out") or 0.0)

    for x in drive_values:
        report, dt = _run_with_params(
            base_geom=base_geom,
            mesh_name=mesh_name,
            protocol=protocol,
            overrides={"tilt_thetaB_contact_strength_in": float(x)},
        )
        rows_drive.append(
            _extract_run(
                key="tilt_thetaB_contact_strength_in",
                value=float(x),
                report=report,
                runtime_s=dt,
            )
        )

    for s in kt_scales:
        report, dt = _run_with_params(
            base_geom=base_geom,
            mesh_name=mesh_name,
            protocol=protocol,
            overrides={
                "tilt_modulus_in": float(base_kt_in * s),
                "tilt_modulus_out": float(base_kt_out * s),
            },
        )
        rows_kt.append(
            _extract_run(
                key="tilt_modulus_scale",
                value=float(s),
                report=report,
                runtime_s=dt,
            )
        )

    rows_drive.sort(key=lambda x: float(x["value"]))
    rows_kt.sort(key=lambda x: float(x["value"]))

    drive_theta = [float(x["thetaB_value"]) for x in rows_drive]
    drive_abs_total = [float(x["abs_total_measured"]) for x in rows_drive]
    kt_theta = [float(x["thetaB_value"]) for x in rows_kt]
    kt_abs_total = [float(x["abs_total_measured"]) for x in rows_kt]

    checks = {
        "drive_theta_non_decreasing": _non_decreasing(drive_theta, monotonic_eps),
        "drive_abs_total_non_decreasing": _non_decreasing(
            drive_abs_total, monotonic_eps
        ),
        "kt_theta_non_increasing": _non_increasing(kt_theta, monotonic_eps),
        "kt_abs_total_non_increasing": _non_increasing(kt_abs_total, monotonic_eps),
    }
    return {
        "meta": {
            "fixture": str(mesh.relative_to(ROOT)),
            "protocol": list(protocol),
            "format": "yaml",
        },
        "sweeps": {
            "drive": rows_drive,
            "tilt_modulus_scale": rows_kt,
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
    p.add_argument("--drive-values", nargs="+", type=float, default=[3.5, 4.286, 5.0])
    p.add_argument("--kt-scales", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    p.add_argument("--monotonic-eps", type=float, default=5.0e-4)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    audit = build_scaling_audit(
        mesh=Path(args.mesh),
        protocol=tuple(DEFAULT_PROTOCOL),
        drive_values=[float(x) for x in args.drive_values],
        kt_scales=[float(x) for x in args.kt_scales],
        monotonic_eps=float(args.monotonic_eps),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(audit, sort_keys=False), encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0 if bool(audit["summary"]["all_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
