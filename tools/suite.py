#!/usr/bin/env python3
"""Run all benchmarks and track performance history.

This script executes available benchmark modules, calculates average runtimes,
and compares them against a stored history of "best" results in ``results.json``.
"""

import argparse
import cProfile
import importlib
import json
import pstats
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

# Import benchmark modules from benchmarks/ without requiring it to be a package.
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

RESULTS_FILE = BENCHMARKS_DIR / "results.json"
DEFAULT_PROFILE_DIR = BENCHMARKS_DIR / "outputs" / "profiles"

_BENCHMARK_MODULES = {
    "cube_good": "benchmark_cube_good",
    "dented_cube": "benchmark_dented_cube",
    "square_to_circle": "benchmark_square_to_circle",
    "catenoid": "benchmark_catenoid",
    "spherical_cap": "benchmark_cap",
    "two_disks_sphere": "benchmark_two_disks_sphere",
    "bending_analytic": "benchmark_bending",
    "volume_optimization": "benchmark_volume_optimization",
    "tilt_relaxation": "benchmark_tilt_relaxation",
    "kozlov_annulus_decay_length": "benchmark_kozlov_annulus_decay_length",
    "kozlov_1disk_tensionless": "benchmark_kozlov_1disk_3d_tensionless",
    "kozlov_1disk_induction_quick": "benchmark_kozlov_1disk_3d_induction_quick",
    "kozlov_1disk_profile_hard_rim": "benchmark_kozlov_1disk_3d_profile_hard_rim",
    "kozlov_1disk_profile_hard_rim_free_disk": "benchmark_kozlov_1disk_3d_profile_hard_rim_free_disk",
}


def _load_benchmarks() -> dict[str, Callable[[], float]]:
    benchmarks: dict[str, Callable[[], float]] = {}
    for name, module_name in _BENCHMARK_MODULES.items():
        mod = importlib.import_module(module_name)
        fn = getattr(mod, "benchmark", None)
        if not callable(fn):
            raise AttributeError(
                f"Benchmark module {module_name} is missing a callable `benchmark()`"
            )
        benchmarks[name] = fn
    return benchmarks


BENCHMARKS = _load_benchmarks()


def load_results():
    if not RESULTS_FILE.exists():
        return {}
    try:
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_results(results):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def run_benchmark(name, func, *, profile: bool, profile_dir: Path, profile_top: int):
    if not profile:
        return func(), None

    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / f"{name}.pstats"
    summary_path = profile_dir / f"{name}.txt"

    profiler = cProfile.Profile()
    profiler.enable()
    avg_time = func()
    profiler.disable()
    profiler.dump_stats(profile_path)

    if profile_top > 0:
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        with summary_path.open("w") as summary_file:
            stats.stream = summary_file
            stats.print_stats(profile_top)

    return avg_time, profile_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile each benchmark and emit per-case .pstats files.",
    )
    parser.add_argument(
        "--profile-dir",
        default=str(DEFAULT_PROFILE_DIR),
        help="Directory for per-benchmark profile outputs.",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=30,
        help="Number of top cumulative entries to write to .txt summaries (0 to skip).",
    )
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)

    print(f"Running benchmarking suite on {sys.platform}...")
    history = load_results()
    current_results = {}

    # Get current timestamp
    timestamp = datetime.now().isoformat()

    any_regression = False

    print(f"{'Benchmark':<25} | {'Current':<10} | {'Best':<10} | {'Change':<10}")
    print("-" * 65)

    for name, func in BENCHMARKS.items():
        # Run benchmark
        avg_time, profile_path = run_benchmark(
            name,
            func,
            profile=args.profile,
            profile_dir=profile_dir,
            profile_top=args.profile_top,
        )
        current_results[name] = {"time": avg_time, "timestamp": timestamp}

        # Compare with best
        best_record = history.get(name)
        best_time = best_record["time"] if best_record else float("inf")

        change_str = "N/A"
        if best_record:
            delta = avg_time - best_time
            pct = (delta / best_time) * 100
            change_str = f"{pct:+.1f}%"

            # Simple color coding for CLI
            if pct > 5.0:  # Regression > 5%
                change_str += " (SLOW)"
                any_regression = True
            elif pct < -5.0:  # Improvement > 5%
                change_str += " (FAST)"

        print(f"{name:<25} | {avg_time:.4f}s    | {best_time:.4f}s    | {change_str}")
        if profile_path is not None:
            print(f"{'':<25}   profile: {profile_path}")

        # Update history if this is the new best (or if no history exists)
        # We also update if the "best" was significantly slower (improvement).
        # But we don't overwrite a fast result with a slow one (regression).
        if avg_time < best_time:
            history[name] = current_results[name]

    print("-" * 65)

    if any_regression:
        print("WARNING: Performance regression detected!")
    else:
        print("Performance is stable or improved.")

    save_results(history)
    print(f"Results saved to {RESULTS_FILE}")
    if args.profile:
        print(f"Profile outputs saved to {profile_dir}")


if __name__ == "__main__":
    main()
