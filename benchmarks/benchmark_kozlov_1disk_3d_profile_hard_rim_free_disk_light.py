#!/usr/bin/env python3
"""Benchmark: Kozlov 1-disk hard-rim profile (light macro)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.context import CommandContext
from commands.executor import execute_command_line
from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

ROOT = Path(__file__).resolve().parent.parent
MESH = (
    ROOT
    / "meshes"
    / "caveolin"
    / "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml"
)
MACRO = "profile_relax_light"
RUNS = 2


def _run_once() -> float:
    mesh = parse_geometry(load_data(str(MESH)))
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    ctx = CommandContext(mesh, minim, minim.stepper)

    start = time.perf_counter()
    execute_command_line(ctx, MACRO)
    elapsed = time.perf_counter() - start

    positions = ctx.mesh.positions_view()
    if not np.all(np.isfinite(positions)):
        raise RuntimeError("Benchmark produced non-finite positions")

    return elapsed


def benchmark(runs: int = RUNS) -> float:
    """Return average runtime over ``runs`` executions."""
    times = [_run_once() for _ in range(int(runs))]
    return float(sum(times) / float(runs))


if __name__ == "__main__":  # pragma: no cover
    avg = benchmark(1)
    print(f"Average runtime: {avg:.4f}s")
