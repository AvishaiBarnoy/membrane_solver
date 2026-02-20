#!/usr/bin/env python3
"""Local runtime probe for strict KH flat-disk optimize runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from tools.reproduce_flat_disk_one_leaflet import (
        DEFAULT_FIXTURE,
        run_flat_disk_one_leaflet_benchmark,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--output",
        default=str(
            ROOT
            / "benchmarks"
            / "outputs"
            / "diagnostics"
            / "flat_disk_kh_runtime_probe.yaml"
        ),
    )
    args = ap.parse_args()

    if int(args.repeats) < 1:
        raise ValueError("repeats must be >= 1")

    timings: list[float] = []
    reports: list[dict] = []
    for _ in range(int(args.repeats)):
        t0 = perf_counter()
        report = run_flat_disk_one_leaflet_benchmark(
            fixture=args.fixture,
            refine_level=1,
            outer_mode="disabled",
            smoothness_model="splay_twist",
            theta_mode="optimize",
            parameterization="kh_physical",
            optimize_preset="kh_strict_refine",
            tilt_mass_mode_in="consistent",
        )
        timings.append(float(perf_counter() - t0))
        reports.append(report)

    sorted_timings = sorted(timings)
    mid = int(len(sorted_timings) // 2)
    median = float(sorted_timings[mid])
    out = {
        "meta": {
            "fixture": str(args.fixture),
            "repeats": int(args.repeats),
            "mode": "kh_strict_refine_runtime_probe",
        },
        "timings_seconds": [float(x) for x in timings],
        "median_seconds": median,
        "best_seconds": float(min(timings)),
        "worst_seconds": float(max(timings)),
        "parity_last": {
            "theta_factor": float(reports[-1]["parity"]["theta_factor"]),
            "energy_factor": float(reports[-1]["parity"]["energy_factor"]),
        },
        "performance_last": reports[-1]["meta"]["performance"],
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
