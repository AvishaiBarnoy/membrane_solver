#!/usr/bin/env python3
"""Run tilt hotspot benchmarks with reproducible guardrails."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import platform
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

THREAD_ENV_KEYS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]

CASE_REGISTRY = {
    "tilt_relax_nested": "benchmark_tilt_relaxation",
    "kozlov_profile_light": "benchmark_kozlov_1disk_3d_profile_hard_rim_free_disk_light",
    "kozlov_tensionless_coupled": "benchmark_kozlov_1disk_3d_tensionless",
}


def _set_thread_env(pin_threads: bool) -> None:
    """Pin BLAS/OpenMP threads to 1 for stable timing."""
    if pin_threads:
        for key in THREAD_ENV_KEYS:
            os.environ.setdefault(key, "1")


def _p95(values: list[float]) -> float:
    """Compute p95 with linear interpolation."""
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * 0.95
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _load_benchmark(module_name: str):
    """Load benchmark() callable from a benchmark module."""
    mod = importlib.import_module(module_name)
    fn = getattr(mod, "benchmark", None)
    if not callable(fn):
        raise TypeError(f"Missing callable benchmark() in {module_name}")
    return fn


def _run_once(fn) -> float:
    """Run benchmark once and return seconds."""
    if "runs" in inspect.signature(fn).parameters:
        dt = float(fn(runs=1))
    else:
        dt = float(fn())
    if dt <= 0.0:
        raise ValueError(f"Non-positive runtime: {dt}")
    return dt


def _resolve_cases(raw_cases: str) -> list[tuple[str, str]]:
    """Resolve case names from CSV string."""
    names = [x.strip() for x in str(raw_cases).split(",") if x.strip()]
    unknown = [x for x in names if x not in CASE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown case(s): {', '.join(unknown)}")
    return [(name, CASE_REGISTRY[name]) for name in names]


def _run_case(name: str, module_name: str, *, warmups: int, runs: int) -> dict:
    """Run one case and return summary row."""
    fn = _load_benchmark(module_name)
    for _ in range(warmups):
        _run_once(fn)
    samples = [_run_once(fn) for _ in range(runs)]
    return {
        "name": name,
        "module": module_name,
        "warmups": warmups,
        "runs": runs,
        "samples_seconds": samples,
        "min_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
        "p95_seconds": _p95(samples),
        "max_seconds": max(samples),
        "stdev_seconds": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    }


def _compare(
    cases: list[dict], baseline: dict, max_regression_percent: float
) -> list[dict]:
    """Compare current medians against baseline medians."""
    prior = {str(row.get("name")): row for row in baseline.get("cases", [])}
    rows: list[dict] = []
    for case in cases:
        old = prior.get(case["name"])
        if old is None:
            continue
        old_m = float(old["median_seconds"])
        cur_m = float(case["median_seconds"])
        pct = ((cur_m - old_m) / old_m) * 100.0 if old_m > 0.0 else 0.0
        rows.append(
            {
                "case": case["name"],
                "baseline_seconds": old_m,
                "current_seconds": cur_m,
                "regression_percent": pct,
                "threshold_percent": max_regression_percent,
                "regressed": pct > max_regression_percent,
            }
        )
    return rows


def _print_table(cases: list[dict], comparisons: list[dict]) -> None:
    """Print summary table and optional baseline comparison."""
    print(f"{'case':<30} {'median(s)':>10} {'mean(s)':>10} {'p95(s)':>10} {'runs':>5}")
    for row in cases:
        print(
            f"{row['name']:<30}{row['median_seconds']:>10.4f}"
            f"{row['mean_seconds']:>10.4f}{row['p95_seconds']:>10.4f}{row['runs']:>5d}"
        )
    if not comparisons:
        return
    print("\nBaseline comparison (median):")
    print(f"{'case':<30} {'baseline':>10} {'current':>10} {'delta%':>9} {'status':>10}")
    for row in comparisons:
        status = "REGRESSED" if row["regressed"] else "ok"
        print(
            f"{row['case']:<30}{row['baseline_seconds']:>10.4f}"
            f"{row['current_seconds']:>10.4f}{row['regression_percent']:>9.2f}{status:>10}"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(CASE_REGISTRY.keys()))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--pin-threads", action="store_true")
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "benchmarks" / "outputs" / "tilt_perf_report.json"),
    )
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--max-regression-percent", type=float, default=10.0)
    args = parser.parse_args(argv)

    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")

    _set_thread_env(args.pin_threads)
    selected = _resolve_cases(args.cases)
    cases = [
        _run_case(name, module, warmups=args.warmups, runs=args.runs)
        for name, module in selected
    ]

    comparisons: list[dict] = []
    if args.baseline_json:
        with Path(args.baseline_json).open("r", encoding="utf-8") as handle:
            baseline = json.load(handle)
        comparisons = _compare(cases, baseline, args.max_regression_percent)

    _print_table(cases, comparisons)

    payload = {
        "meta": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "thread_env": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        },
        "config": {
            "cases": [name for name, _ in selected],
            "warmups": args.warmups,
            "runs": args.runs,
            "pin_threads": args.pin_threads,
            "baseline_json": args.baseline_json,
            "max_regression_percent": args.max_regression_percent,
        },
        "cases": cases,
        "baseline_comparison": comparisons,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote: {out_path}")

    return 2 if any(row["regressed"] for row in comparisons) else 0


if __name__ == "__main__":
    raise SystemExit(main())
