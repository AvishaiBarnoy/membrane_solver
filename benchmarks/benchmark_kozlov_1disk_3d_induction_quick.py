#!/usr/bin/env python3
"""Benchmark: fast single-leaflet curvature induction (tensionless).

This benchmark targets the "single leaflet source induces curvature and tilt in
the opposite leaflet" behavior on the small annulus mesh
`meshes/caveolin/kozlov_1disk_3d_tensionless_single_leaflet_soft_source.yaml`.

It is designed to be quick (small mesh, short run) while still exercising the
shape+tilt coupling hot path.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.curvature import compute_curvature_fields
from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

ROOT = Path(__file__).resolve().parent.parent
MESH = ROOT / "benchmarks" / "inputs" / "bench_kozlov_1disk_induction_quick.yaml"
RUNS = 5


def _run_once(*, n_steps: int = 30) -> float:
    mesh = parse_geometry(load_data(str(MESH)))
    mesh.global_parameters.update(
        {
            "surface_tension": 0.0,
            "tilt_rim_source_contact_units": "solver",
            "tilt_rim_source_contact_h_in": 1.0,
            "tilt_rim_source_contact_delta_epsilon_over_a_in": 40.0,
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.15,
            "tilt_inner_steps": 40,
            "tilt_tol": 1e-10,
            "step_size": 0.006,
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

    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    t_in = np.linalg.norm(tilts_in, axis=1)
    t_out = np.linalg.norm(tilts_out, axis=1)

    boundary_vids = getattr(mesh, "boundary_vertex_ids", None) or []
    boundary_rows = {mesh.vertex_index_to_row[vid] for vid in boundary_vids}
    interior_rows = np.array(
        [row for row in range(len(mesh.vertex_ids)) if row not in boundary_rows],
        dtype=int,
    )

    if float(np.percentile(t_in[interior_rows], 90)) <= 1e-3:
        raise RuntimeError("Benchmark produced near-zero tilt_in response")
    if float(np.percentile(t_out[interior_rows], 90)) <= 1e-5:
        raise RuntimeError("Benchmark produced near-zero induced tilt_out response")

    curvature = compute_curvature_fields(
        mesh, mesh.positions_view(), mesh.vertex_index_to_row
    ).mean_curvature
    if float(np.percentile(curvature[interior_rows], 90)) <= 1e-4:
        raise RuntimeError("Benchmark produced near-zero curvature response")

    return float(elapsed)


def benchmark(runs: int = RUNS) -> float:
    """Return average runtime over ``runs`` executions."""
    times = [_run_once() for _ in range(int(runs))]
    return float(sum(times) / float(runs))


if __name__ == "__main__":  # pragma: no cover
    avg = benchmark(1)
    print(f"Average runtime: {avg:.4f}s")
