"""Estimate tilt-field energy convergence under mesh refinement.

This script repeatedly relaxes the tilt field on a fixed-geometry mesh (using
the configured tilt solve mode) and reports the resulting energy at each
refinement level. It is intended to help validate the expected screened-Laplace
decay length scale implied by:

    E = 1/2 ∫ (k_s |∇t|^2 + k_t |t|^2) dA

where k_s is ``tilt_smoothness_rigidity`` and k_t is ``tilt_rigidity``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.refinement import refine_triangle_mesh
from runtime.steppers.gradient_descent import GradientDescent


def _relax_tilts(mesh, *, mode: str, step_size: float, inner_steps: int, tol: float):
    mesh.global_parameters.update(
        {
            "tilt_solve_mode": mode,
            "tilt_step_size": step_size,
            "tilt_inner_steps": inner_steps,
            "tilt_coupled_steps": inner_steps,
            "tilt_tol": tol,
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
    minim._relax_tilts(positions=mesh.positions_view(), mode=mode)
    return float(minim.compute_energy())


def _predict_1d_energy(*, ks: float, kt: float, width: float, height: float) -> float:
    lam = float(np.sqrt(ks / kt)) if kt > 0 else float("inf")
    if not np.isfinite(lam) or lam <= 0:
        return float("nan")
    return 0.5 * height * float(np.sqrt(ks * kt)) * float(np.tanh(width / lam))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh",
        default="meshes/tilt_benchmarks/tilt_source_rect_single.yaml",
        help="Path to input mesh (default: single-source rectangle).",
    )
    parser.add_argument(
        "--refines",
        type=int,
        default=4,
        help="Number of triangle refinement passes (default: 4).",
    )
    parser.add_argument(
        "--mode",
        default="nested",
        choices=["nested", "coupled"],
        help="Tilt solve mode (default: nested).",
    )
    parser.add_argument(
        "--tilt-step-size",
        type=float,
        default=0.05,
        help="Tilt gradient-descent step size (default: 0.05).",
    )
    parser.add_argument(
        "--tilt-inner-steps",
        type=int,
        default=800,
        help="Tilt inner steps per refinement level (default: 800).",
    )
    parser.add_argument(
        "--tilt-tol",
        type=float,
        default=1e-12,
        help="Tilt gradient-norm tolerance (default: 1e-12).",
    )
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    mesh = parse_geometry(load_data(mesh_path))

    ks = float(mesh.global_parameters.get("tilt_smoothness_rigidity", 0.0) or 0.0)
    kt = float(mesh.global_parameters.get("tilt_rigidity", 0.0) or 0.0)
    width = float(mesh.positions_view()[:, 0].max() - mesh.positions_view()[:, 0].min())
    height = float(
        mesh.positions_view()[:, 1].max() - mesh.positions_view()[:, 1].min()
    )
    pred = _predict_1d_energy(ks=ks, kt=kt, width=width, height=height)

    header = ["level", "verts", "facets", "energy"]
    print(" ".join(h.ljust(12) for h in header))
    if np.isfinite(pred):
        print(f"# 1D Dirichlet/Neumann estimate: {pred:.6f} (λ≈{np.sqrt(ks / kt):.3f})")

    for level in range(args.refines + 1):
        energy = _relax_tilts(
            mesh,
            mode=args.mode,
            step_size=args.tilt_step_size,
            inner_steps=args.tilt_inner_steps,
            tol=args.tilt_tol,
        )
        print(
            f"{level:<12d}"
            f"{len(mesh.vertices):<12d}"
            f"{len(mesh.facets):<12d}"
            f"{energy:<12.6f}"
        )
        if level < args.refines:
            mesh = refine_triangle_mesh(mesh)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
