#!/usr/bin/env python3
"""Estimate the tilt decay length for the Kozlov annulus benchmarks (Milestone B).

This script loads one of the flat annulus YAML benchmarks under
`meshes/caveolin/` (hard-clamped or soft-driven rim source), optionally refines
the mesh, relaxes the leaflet tilt fields in nested mode, and then estimates an
effective decay length λ by fitting

    |t(r)| ≈ A * exp(-(r - r_in) / λ)

over an interior radial range.

Notes
-----
- This is a *diagnostic benchmark*, not a unit test: it prints a single-number
  estimate to help compare parameter sets and discretizations.
- The estimate is only meaningful when the decay region is resolved (λ is not
  much smaller than the local mesh spacing).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Ensure the project root is in sys.path when running directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.refinement import refine_triangle_mesh
from runtime.steppers.gradient_descent import GradientDescent

logger = logging.getLogger("membrane_solver")

ROOT = Path(__file__).resolve().parent.parent
HARD_MESH = ROOT / "meshes" / "caveolin" / "kozlov_annulus_flat_hard_source.yaml"
SOFT_MESH = ROOT / "meshes" / "caveolin" / "kozlov_annulus_flat_soft_source.yaml"


def _relax_leaflet_tilts(mesh, *, inner_steps: int, tilt_step_size: float) -> Minimizer:
    """Relax leaflet tilts (nested mode) without advancing shape steps."""
    mesh.global_parameters.update(
        {
            "tilt_solve_mode": "nested",
            "tilt_step_size": float(tilt_step_size),
            "tilt_inner_steps": int(inner_steps),
            "tilt_tol": 1e-12,
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
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.increment_version()
    mesh._positions_cache = None
    mesh._positions_cache_version = -1
    mesh.project_tilts_to_tangent()
    minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode="nested")
    return minim


def _bin_radial_profile(
    radii: np.ndarray,
    mags: np.ndarray,
    *,
    r_min: float,
    r_max: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (r_centers, mean(|t|)) binned by radius."""
    if n_bins < 3:
        raise ValueError("n_bins must be >= 3")
    edges = np.linspace(float(r_min), float(r_max), int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full_like(centers, np.nan, dtype=float)
    for i in range(int(n_bins)):
        mask = (radii >= edges[i]) & (radii < edges[i + 1])
        if not np.any(mask):
            continue
        means[i] = float(np.mean(mags[mask]))
    good = np.isfinite(means)
    return centers[good], means[good]


def _fit_decay_length(
    r: np.ndarray,
    mags: np.ndarray,
    *,
    r_in: float,
    fit_r_min: float,
    fit_r_max: float,
    eps: float = 1e-12,
) -> float:
    """Fit log(|t|) vs r and return λ (in the same units as r)."""
    r = np.asarray(r, dtype=float)
    mags = np.asarray(mags, dtype=float)

    mask = (r >= float(fit_r_min)) & (r <= float(fit_r_max)) & (mags > float(eps))
    if int(np.count_nonzero(mask)) < 3:
        raise ValueError("Not enough points to fit decay length")

    x = r[mask] - float(r_in)
    y = np.log(mags[mask])

    slope, _intercept = np.polyfit(x, y, deg=1)
    if slope >= 0.0:
        raise ValueError(f"Non-decaying fit (slope={slope})")
    return float(-1.0 / slope)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate tilt decay length for Kozlov annulus benchmarks."
    )
    parser.add_argument(
        "--mesh",
        choices=("hard", "soft"),
        default="hard",
        help="Which Milestone B annulus benchmark to run.",
    )
    parser.add_argument(
        "--refine",
        type=int,
        default=2,
        help="Number of triangle refinement passes to apply before relaxation.",
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=1600,
        help="Nested tilt relaxation inner steps.",
    )
    parser.add_argument(
        "--tilt-step-size",
        type=float,
        default=0.05,
        help="Nested tilt relaxation step size.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=24,
        help="Number of radial bins for the profile fit.",
    )
    parser.add_argument(
        "--fit-min",
        type=float,
        default=1.2,
        help="Minimum radius to include in the decay-length fit.",
    )
    parser.add_argument(
        "--fit-max",
        type=float,
        default=2.7,
        help="Maximum radius to include in the decay-length fit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (e.g. INFO, DEBUG).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO)
    )

    mesh_path = HARD_MESH if args.mesh == "hard" else SOFT_MESH
    logger.info("Loading mesh: %s", mesh_path)
    mesh = parse_geometry(load_data(str(mesh_path)))

    for _ in range(max(int(args.refine), 0)):
        mesh = refine_triangle_mesh(mesh)

    _relax_leaflet_tilts(
        mesh,
        inner_steps=int(args.inner_steps),
        tilt_step_size=float(args.tilt_step_size),
    )

    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    mags = np.linalg.norm(mesh.tilts_in_view(), axis=1)

    r_centers, m_centers = _bin_radial_profile(
        radii,
        mags,
        r_min=float(np.min(radii)),
        r_max=float(np.max(radii)),
        n_bins=int(args.bins),
    )

    r_in = 1.0
    lam = _fit_decay_length(
        r_centers,
        m_centers,
        r_in=r_in,
        fit_r_min=float(args.fit_min),
        fit_r_max=float(args.fit_max),
        eps=1e-12,
    )

    print(f"mesh={args.mesh} refine={args.refine} inner_steps={args.inner_steps}")
    print(f"lambda_est={lam:.6g} (fit radii [{args.fit_min}, {args.fit_max}])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
