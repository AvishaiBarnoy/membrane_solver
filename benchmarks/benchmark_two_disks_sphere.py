#!/usr/bin/env python3
"""Benchmark for a sphere with two small flat disk patches.

This is a geometry + infrastructure benchmark (ROADMAP inclusion-disk scaffold):
- exercises minimization + refinement + constraints (pin_to_circle/pin_to_plane)
- does not depend on mean/gaussian curvature energies
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_JSON = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "inputs"
    / "bench_two_disks_sphere.json"
)
RUNS = 1


def _run_simulation(input_path: Path) -> float:
    """Execute ``main.py`` and return the elapsed time."""
    start = time.perf_counter()

    main_py_path = Path(__file__).resolve().parent.parent / "main.py"

    subprocess.run(
        [
            sys.executable,
            str(main_py_path),
            "-i",
            str(input_path),
            "--non-interactive",
            "-q",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - start


def benchmark(runs: int = RUNS) -> float:
    """Return the average runtime over ``runs`` executions."""
    times = [_run_simulation(BASE_JSON) for _ in range(runs)]
    return sum(times) / runs


if __name__ == "__main__":
    avg = benchmark()
    print(f"two_disks_sphere average runtime over {RUNS} runs: {avg:.4f}s")
