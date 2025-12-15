#!/usr/bin/env python3
"""Benchmark full minimization for the catenoid_good_min routine.

This script runs ``main.py`` on ``meshes/catenoid_good_min.json`` a few
times and reports the average wall-clock runtime.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_JSON = Path(__file__).resolve().parent.parent / "meshes" / "catenoid_good_min.json"
RUNS = 3


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
    times = []
    # Warmup run (optional, but good for stability)
    # _run_simulation(BASE_JSON)

    for i in range(runs):
        elapsed = _run_simulation(BASE_JSON)
        times.append(elapsed)

    return sum(times) / runs


if __name__ == "__main__":
    avg = benchmark()
    print(f"catenoid_good_min average runtime over {RUNS} runs: {avg:.4f}s")
