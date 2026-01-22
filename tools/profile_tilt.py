#!/usr/bin/env python3
"""Profile tilt relaxation hot-loops.

This script is for infrastructure/performance work: it runs a single tilt
relaxation pass (single-field or dual-leaflet, depending on the mesh modules)
under ``cProfile`` and writes a ``.pstats`` file plus an optional text summary.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
from pathlib import Path

# Ensure the project root is in sys.path when running directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MESH = (
    ROOT / "meshes" / "caveolin" / "kozlov_annulus_milestone_c_soft_source.yaml"
)
DEFAULT_OUTDIR = ROOT / "benchmarks" / "outputs" / "profiles"


def _build_minimizer(mesh) -> Minimizer:
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def _run_relaxation(
    *,
    mesh_path: Path,
    mode: str,
    inner_steps: int,
    step_size: float,
) -> None:
    mesh = parse_geometry(load_data(str(mesh_path)))
    mesh.global_parameters.update(
        {
            "tilt_solve_mode": str(mode),
            "tilt_inner_steps": int(inner_steps),
            "tilt_coupled_steps": int(inner_steps),
            "tilt_step_size": float(step_size),
            "tilt_tol": 0.0,
        }
    )

    minim = _build_minimizer(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.project_tilts_to_tangent()

    positions = mesh.positions_view()
    if minim._uses_leaflet_tilts():
        minim._relax_leaflet_tilts(positions=positions, mode=str(mode))
    else:
        minim._relax_tilts(positions=positions, mode=str(mode))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mesh",
        default=str(DEFAULT_MESH),
        help="Path to a JSON/YAML mesh to profile.",
    )
    p.add_argument(
        "--mode",
        choices=("nested", "coupled"),
        default="nested",
        help="Tilt solve mode to profile.",
    )
    p.add_argument(
        "--inner-steps",
        type=int,
        default=50,
        help="Number of inner tilt relaxation steps.",
    )
    p.add_argument(
        "--tilt-step-size",
        type=float,
        default=0.05,
        help="Tilt relaxation step size.",
    )
    p.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help="Directory to write .pstats/.txt outputs.",
    )
    p.add_argument(
        "--label",
        default="tilt",
        help="Basename label for output files.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=60,
        help="Number of top cumulative entries to include in the text summary (0 to skip).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mesh_path = Path(args.mesh)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pstats_path = outdir / f"{args.label}.pstats"
    summary_path = outdir / f"{args.label}.txt"

    profiler = cProfile.Profile()
    profiler.enable()
    _run_relaxation(
        mesh_path=mesh_path,
        mode=str(args.mode),
        inner_steps=int(args.inner_steps),
        step_size=float(args.tilt_step_size),
    )
    profiler.disable()
    profiler.dump_stats(pstats_path)
    print(f"wrote: {pstats_path}")

    top = int(args.top)
    if top > 0:
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        with summary_path.open("w") as f:
            stats.stream = f
            stats.print_stats(top)
        print(f"wrote: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
