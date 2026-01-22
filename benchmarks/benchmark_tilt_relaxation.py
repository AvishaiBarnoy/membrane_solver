#!/usr/bin/env python3
"""Benchmark tilt relaxation hot-loops (no subprocess).

This benchmark runs a single inner-loop tilt relaxation pass on a moderate-size
Milestone C mesh and returns the elapsed time.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

ROOT = Path(__file__).resolve().parent.parent
MESH = ROOT / "meshes" / "caveolin" / "kozlov_annulus_milestone_c_soft_source.yaml"
RUNS = 3


def _build_minimizer(mesh) -> Minimizer:
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def _run_once(*, inner_steps: int = 50, tilt_step_size: float = 0.05) -> float:
    mesh = parse_geometry(load_data(str(MESH)))
    mesh.global_parameters.update(
        {
            "tilt_solve_mode": "nested",
            "tilt_inner_steps": int(inner_steps),
            "tilt_step_size": float(tilt_step_size),
            "tilt_tol": 0.0,
        }
    )

    minim = _build_minimizer(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.project_tilts_to_tangent()

    positions = mesh.positions_view()
    start = time.perf_counter()
    if minim._uses_leaflet_tilts():
        minim._relax_leaflet_tilts(positions=positions, mode="nested")
    else:
        minim._relax_tilts(positions=positions, mode="nested")
    return time.perf_counter() - start


def benchmark(runs: int = RUNS) -> float:
    """Return average runtime over ``runs`` executions."""
    times = [_run_once() for _ in range(int(runs))]
    return sum(times) / float(runs)


if __name__ == "__main__":
    avg = benchmark()
    print(f"tilt_relaxation average runtime over {RUNS} runs: {avg:.4f}s")
