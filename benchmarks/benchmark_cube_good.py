#!/usr/bin/env python3
"""Benchmark full minimization for the cube_good_min_routine setup.

This script runs ``main.py`` on ``meshes/cube_good_min_routine.json`` a few
times and reports the average wall-clock runtime. It is intended as a simple
regression benchmark for performance tuning (e.g., geometry kernels,
steppers, constraint handling).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_JSON = Path(__file__).resolve().parent.parent / "meshes" / "cube_good_min_routine.json"
RUNS = 3


def _run_simulation(input_path: Path, output_path: Path) -> float:
    """Execute ``main.py`` and return the elapsed time."""
    start = time.perf_counter()

    main_py_path = Path(__file__).resolve().parent.parent / "main.py"

    subprocess.run(
        [
            sys.executable,
            str(main_py_path),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
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
    with TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # we can reuse the base JSON directly
        inp = BASE_JSON
        out = tmpdir / "out.json"
        times = [_run_simulation(inp, out) for _ in range(runs)]
    return sum(times) / runs


if __name__ == "__main__":
    avg = benchmark()
    print(f"cube_good_min_routine average runtime over {RUNS} runs: {avg:.4f}s")
