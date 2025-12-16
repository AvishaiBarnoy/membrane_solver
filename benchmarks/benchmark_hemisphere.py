#!/usr/bin/env python3
"""Benchmark for Pinned Hemisphere (Laplace Law verification).

This script runs ``main.py`` on ``meshes/hemisphere_start.json`` and verifies
that the final geometry approximates a hemisphere of radius 1.

The mesh starts as a flat unit hexagon. It is inflated to volume 2/3*pi*R^3.
Since the boundary is fixed at R=1, the final shape should be a hemisphere
with apex height close to 1.0.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_JSON = Path(__file__).resolve().parent.parent / "meshes" / "hemisphere_start.json"
OUTPUT_JSON = Path(__file__).resolve().parent.parent / "outputs" / "hemisphere_result.json"
RUNS = 1

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

def verify_results(output_path: Path):
    with open(output_path, 'r') as f:
        data = json.load(f)

    # Find the apex (vertex with max Z)
    max_z = -float('inf')
    for v in data["vertices"]:
        # v is [x, y, z, options] or [x, y, z]
        z = v[2]
        if z > max_z:
            max_z = z

    print(f"Final Apex Height: {max_z:.5f} (Expected ~1.000)")

    # Error tolerance: 2% is reasonable for a discretized mesh
    if abs(max_z - 1.0) > 0.02:
        print("WARNING: Apex height deviates significantly from 1.0")
    else:
        print("SUCCESS: Geometry matches theoretical hemisphere.")

def benchmark(runs: int = RUNS) -> float:
    """Return the average runtime over ``runs`` executions."""
    times = []

    # Ensure output directory exists
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    for i in range(runs):
        elapsed = _run_simulation(BASE_JSON, OUTPUT_JSON)
        times.append(elapsed)

    verify_results(OUTPUT_JSON)
    return sum(times) / runs

if __name__ == "__main__":
    avg = benchmark()
    print(f"hemisphere average runtime over {RUNS} runs: {avg:.4f}s")
