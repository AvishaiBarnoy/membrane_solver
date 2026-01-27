#!/usr/bin/env python3
"""Benchmark: tensionless Kozlov 1-disk setup (small-drive, physical scaling).

This benchmark loads the single-leaflet rim-source disk+annulus mesh and runs a
short shape+tilt relaxation in the small-drive regime with physically scaled
ratios (κ=1, k_t≈135 for 1 unit=15nm).

It is meant to be a lightweight runtime benchmark and a sanity check that the
workflow completes and produces a non-zero curvature response.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

ROOT = Path(__file__).resolve().parent.parent
MESH = (
    ROOT
    / "benchmarks"
    / "inputs"
    / "bench_kozlov_1disk_tensionless_single_leaflet_source.yaml"
)
RUNS = 3


def _run_once(*, n_steps: int = 120) -> float:
    mesh = parse_geometry(load_data(str(MESH)))
    mesh.global_parameters.update(
        {
            "bending_modulus_in": 1.0,
            "bending_modulus_out": 1.0,
            "tilt_modulus_in": 135.0,
            "tilt_modulus_out": 135.0,
            "tilt_rim_source_strength_in": 5000.0,
            "tilt_solve_mode": "coupled",
            "tilt_step_size": 0.05,
            "tilt_inner_steps": 60,
            "tilt_tol": 1e-12,
            "step_size": 0.003,
            "step_size_mode": "fixed",
        }
    )
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )

    start = time.perf_counter()
    minim.minimize(n_steps=int(n_steps))
    elapsed = time.perf_counter() - start

    z_span = float(np.ptp(mesh.positions_view()[:, 2]))
    if not np.isfinite(z_span) or z_span <= 0.0:
        raise RuntimeError("Benchmark produced invalid z-span")
    return elapsed


def benchmark(runs: int = RUNS) -> float:
    """Return average runtime over ``runs`` executions."""
    times = [_run_once() for _ in range(int(runs))]
    return float(sum(times) / float(runs))


if __name__ == "__main__":  # pragma: no cover
    avg = benchmark(1)
    print(f"Average runtime: {avg:.4f}s")
