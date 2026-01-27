#!/usr/bin/env python3
"""Benchmark: Kozlov 1-disk hard-rim profile relaxation (disk free in z)."""

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
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh
from runtime.steppers.gradient_descent import GradientDescent
from runtime.vertex_average import vertex_average

ROOT = Path(__file__).resolve().parent.parent
MESH = (
    ROOT
    / "benchmarks"
    / "inputs"
    / "bench_kozlov_1disk_profile_hard_rim_R12_free_disk.yaml"
)
RUNS = 2


def _refine_and_smooth(minimizer: Minimizer, *, smooth_passes: int = 5) -> None:
    mesh = refine_polygonal_facets(minimizer.mesh)
    mesh = refine_triangle_mesh(mesh)
    minimizer.mesh = mesh
    minimizer.enforce_constraints_after_mesh_ops(mesh)
    for _ in range(int(smooth_passes)):
        vertex_average(mesh)
    minimizer.enforce_constraints_after_mesh_ops(mesh)


def _run_once() -> float:
    mesh = parse_geometry(load_data(str(MESH)))
    mesh.global_parameters.update(
        {
            "tilt_solve_mode": "coupled",
            "tilt_step_size": 0.15,
            "tilt_inner_steps": 40,
            "tilt_tol": 1e-10,
            "step_size": 0.005,
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
    minim.minimize(n_steps=40)
    _refine_and_smooth(minim, smooth_passes=5)
    minim.minimize(n_steps=100)
    _refine_and_smooth(minim, smooth_passes=5)
    minim.minimize(n_steps=120)
    elapsed = time.perf_counter() - start

    z_span = float(np.ptp(minim.mesh.positions_view()[:, 2]))
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
