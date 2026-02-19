#!/usr/bin/env python3
"""Benchmark tilt_in mass discretization modes for flat KH reproduction.

Compares runtime and parity metrics for:
  - tilt_mass_mode_in=lumped
  - tilt_mass_mode_in=consistent

This benchmark is harness-level (flat one-leaflet KH lane) and does not change
runtime defaults.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.reproduce_flat_disk_one_leaflet import (  # noqa: E402
    DEFAULT_FIXTURE,
    run_flat_disk_one_leaflet_benchmark,
)

DEFAULT_OUT = ROOT / "benchmarks" / "outputs" / "flat_disk_tilt_mass_mode.yaml"


def _run_mode(
    *,
    mode: str,
    fixture: Path,
    refine_level: int,
    runs: int,
    theta_mode: str,
    optimize_preset: str,
) -> dict[str, Any]:
    times: list[float] = []
    report_last: dict[str, Any] | None = None
    for _ in range(int(runs)):
        t0 = time.perf_counter()
        report_last = run_flat_disk_one_leaflet_benchmark(
            fixture=fixture,
            refine_level=int(refine_level),
            outer_mode="disabled",
            smoothness_model="splay_twist",
            theta_mode=str(theta_mode),
            parameterization="kh_physical",
            optimize_preset=str(optimize_preset),
            tilt_mass_mode_in=str(mode),
        )
        times.append(float(time.perf_counter() - t0))

    assert report_last is not None
    return {
        "timing_seconds": {
            "runs": int(runs),
            "values": [float(x) for x in times],
            "mean": float(statistics.fmean(times)),
            "median": float(statistics.median(times)),
            "min": float(min(times)),
            "max": float(max(times)),
        },
        "parity": {
            "theta_factor": float(report_last["parity"]["theta_factor"]),
            "energy_factor": float(report_last["parity"]["energy_factor"]),
            "meets_factor_2": bool(report_last["parity"]["meets_factor_2"]),
        },
        "mesh": {
            "theta_star": float(report_last["mesh"]["theta_star"]),
            "total_energy": float(report_last["mesh"]["total_energy"]),
        },
        "optimize": report_last.get("optimize"),
    }


def benchmark_tilt_mass_mode(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    refine_level: int = 1,
    runs: int = 2,
    theta_mode: str = "optimize",
    optimize_preset: str = "kh_wide",
) -> dict[str, Any]:
    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    if int(runs) < 1:
        raise ValueError("runs must be >= 1.")

    lumped = _run_mode(
        mode="lumped",
        fixture=fixture_path,
        refine_level=int(refine_level),
        runs=int(runs),
        theta_mode=str(theta_mode),
        optimize_preset=str(optimize_preset),
    )
    consistent = _run_mode(
        mode="consistent",
        fixture=fixture_path,
        refine_level=int(refine_level),
        runs=int(runs),
        theta_mode=str(theta_mode),
        optimize_preset=str(optimize_preset),
    )

    t_l = float(lumped["timing_seconds"]["median"])
    t_c = float(consistent["timing_seconds"]["median"])
    speed_ratio = float(t_c / max(t_l, 1e-18))

    return {
        "meta": {
            "fixture": str(fixture_path.relative_to(ROOT)),
            "refine_level": int(refine_level),
            "runs": int(runs),
            "theta_mode": str(theta_mode),
            "optimize_preset": str(optimize_preset),
            "parameterization": "kh_physical",
            "smoothness_model": "splay_twist",
        },
        "modes": {
            "lumped": lumped,
            "consistent": consistent,
        },
        "comparison": {
            "median_time_ratio_consistent_over_lumped": speed_ratio,
            "theta_factor_delta": float(
                consistent["parity"]["theta_factor"] - lumped["parity"]["theta_factor"]
            ),
            "energy_factor_delta": float(
                consistent["parity"]["energy_factor"]
                - lumped["parity"]["energy_factor"]
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--refine-level", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument(
        "--theta-mode",
        choices=("scan", "optimize", "optimize_full"),
        default="optimize",
    )
    ap.add_argument(
        "--optimize-preset",
        choices=("none", "fast_r3", "full_accuracy_r3", "kh_wide"),
        default="kh_wide",
    )
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    report = benchmark_tilt_mass_mode(
        fixture=args.fixture,
        refine_level=args.refine_level,
        runs=args.runs,
        theta_mode=args.theta_mode,
        optimize_preset=args.optimize_preset,
    )

    out = Path(args.output)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
