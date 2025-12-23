#!/usr/bin/env python3
"""Run all benchmarks and track performance history.

This script executes available benchmark modules, calculates average runtimes,
and compares them against a stored history of "best" results in ``results.json``.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add local directory to path to import benchmark modules
sys.path.append(os.path.dirname(__file__))

import benchmark_cap
import benchmark_catenoid
import benchmark_cube_good
import benchmark_dented_cube
import benchmark_square_to_circle
import benchmark_two_disks_sphere

RESULTS_FILE = Path(__file__).parent / "inputs" / "results.json"

BENCHMARKS = {
    "cube_good": benchmark_cube_good.benchmark,
    "dented_cube": benchmark_dented_cube.benchmark,
    "square_to_circle": benchmark_square_to_circle.benchmark,
    "catenoid": benchmark_catenoid.benchmark,
    "spherical_cap": benchmark_cap.benchmark,
    "two_disks_sphere": benchmark_two_disks_sphere.benchmark,
}


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


def main():
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
        avg_time = func()
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


if __name__ == "__main__":
    main()
