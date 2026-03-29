"""Reproduction tool for the curved one-leaflet disk parity (1_disk_3d.tex).

This script runs a theta_B scan or optimization where both the shape (z)
and the tilt field are allowed to relax at each step. This allows the mesh
to form the curved "trumpet" shape assumed by the theory.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

from core.ordered_unique_list import OrderedUniqueList
from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent
from tools.diagnostics.curved_disk_theory import (
    compute_curved_disk_theory,
    tex_reference_params,
)


def _load_mesh_from_fixture(fixture_path: Path):
    data = load_data(str(fixture_path))
    return parse_geometry(data)


def _configure_minimizer(mesh) -> Minimizer:
    m = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    m.refresh_modules()
    return m


def _run_relaxation(
    minim: Minimizer,
    *,
    theta_value: float,
    shape_steps: int = 100,
) -> float:
    """Relax shape and tilt for a fixed theta_B."""
    mesh = minim.mesh
    gp = mesh.global_parameters
    gp.set("tilt_thetaB_value", float(theta_value))

    # Step 1: Fix shape, relax tilts at this theta_B
    minim._relax_leaflet_tilts(
        positions=minim.mesh.positions_view(),
        mode=str(minim.global_params.get("tilt_solve_mode", "coupled")),
    )

    # Step 2: Full shape+tilt relaxation
    minim.minimize(n_steps=shape_steps)

    return float(minim.compute_energy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=str,
        default="tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml",
    )
    parser.add_argument("--refine", type=int, default=2)
    parser.add_argument(
        "--steps", type=int, default=120, help="Minimization steps per theta value."
    )
    parser.add_argument("--theta-min", type=float, default=0.1)
    parser.add_argument("--theta-max", type=float, default=0.25)
    parser.add_argument("--theta-count", type=int, default=7)
    args = parser.parse_args()

    theory_params = tex_reference_params()
    theory = compute_curved_disk_theory(theory_params)
    print("Theory Benchmark (Tensionless):")
    print(f"  theta_star: {theory.theta_star:.4f}")
    print(f"  phi_star:   {theory.phi_star:.4f}")
    print(f"  Total Energy: {theory.total:.4f}")

    ROOT = Path(__file__).resolve().parents[1]

    # Create a baseline mesh after refinement and kick
    base_mesh = _load_mesh_from_fixture(ROOT / args.fixture)
    from runtime.refinement import refine_triangle_mesh

    for _ in range(args.refine):
        base_mesh = refine_triangle_mesh(base_mesh)

    # Tag vertices for matching based on radius
    R = theory_params.radius
    positions = base_mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)

    # Find disk rim vertices - use a small range around theoretical R
    rim_mask = (radii > R - 0.05) & (radii < R + 0.05)
    # Vertices for outer slope - annulus just outside R
    near_mask = (radii > R + 0.05) & (radii < R + 0.3)

    for row, vid in enumerate(base_mesh.vertex_ids):
        v = base_mesh.vertices[int(vid)]
        v.options = copy.deepcopy(v.options)
        opts = v.options

        if rim_mask[row]:
            opts["rim_slope_match_group"] = "disk_rim"
            opts["tilt_thetaB_group"] = "disk_rim"
            opts["preset"] = "rim_active"
        if near_mask[row]:
            opts["rim_slope_match_outer_group"] = "near_disk"

    # Initial trumpet shape
    phi_seed = 0.05
    for row, vid in enumerate(base_mesh.vertex_ids):
        v = base_mesh.vertices[int(vid)]
        r = radii[row]
        if r > R:
            v.position[2] = phi_seed * R * np.log(r / R)
        else:
            v.position[2] = 0.0

    base_mesh.increment_version()
    print(f"Post-kick z-span: {float(np.ptp(base_mesh.positions_view()[:, 2])):.4f}")

    # Explicitly set modules to ensure they are active
    base_mesh.energy_modules = OrderedUniqueList(
        [
            "bending_tilt_in",
            "bending_tilt_out",
            "tilt_in",
            "tilt_out",
            "tilt_thetaB_contact_in",
        ]
    )
    base_mesh.constraint_modules = OrderedUniqueList(["rim_slope_match_out"])

    gp = base_mesh.global_parameters
    gp.set("bending_modulus_in", theory_params.kappa)
    gp.set("bending_modulus_out", theory_params.kappa)
    gp.set("tilt_modulus_in", theory_params.kappa_t)
    gp.set("tilt_modulus_out", theory_params.kappa_t)
    gp.set("tilt_thetaB_contact_strength_in", theory_params.drive)
    gp.set("tilt_thetaB_strength_in", 1000.0)  # Penalty stiffness
    gp.set("tilt_thetaB_contact_penalty_mode", "legacy")
    gp.set("surface_tension", theory_params.surface_tension)
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_inner_steps", 100)
    gp.set("step_size", 0.01)
    gp.set("step_size_mode", "backtracking")

    # Enable leaflet absence for the disk region
    gp.set("leaflet_out_absent_presets", ["disk"])
    gp.set("leaflet_in_absent_presets", [])

    # Enable divergence capping to handle the rim spike
    gp.set("bending_tilt_in_update_mode", "outer_near_divergence_cap_v1")
    gp.set("bending_tilt_out_update_mode", "outer_near_divergence_cap_v1")
    gp.set("benchmark_disk_radius", theory_params.radius)
    gp.set("benchmark_lambda_value", theory.lambda_value)
    gp.set("bending_tilt_assume_J0_presets_in", ["disk", "rim_active"])
    gp.set("bending_tilt_assume_J0_presets_out", ["disk", "rim_active"])

    # Theory matching alignment
    gp.set("rim_slope_match_group", "disk_rim")  # We will tag these
    gp.set("rim_slope_match_outer_group", "near_disk")  # And these
    gp.set("bending_tilt_base_term_boundary_group_out", "disk_rim")

    # CRITICAL: Disable internal thetaB optimization
    gp.set("tilt_thetaB_optimize", False)
    gp.set("line_search_reduced_energy", False)

    # Gradient Mode
    gp.set("bending_tilt_in_gradient_mode", "finite_difference")
    gp.set("bending_tilt_out_gradient_mode", "finite_difference")
    gp.set("bending_fd_eps", 1e-5)

    theta_values = np.linspace(args.theta_min, args.theta_max, args.theta_count)
    energies = []
    phi_stars = []
    z_spans = []

    print(
        f"\nRunning theta_B scan ({args.theta_count} points, {args.steps} steps each)..."
    )
    for tv in theta_values:
        # Independent sample: start from base_mesh each time
        mesh = base_mesh.copy()
        minim = _configure_minimizer(mesh)

        e_total = _run_relaxation(minim, theta_value=tv, shape_steps=args.steps)

        # Diagnostics
        from modules.constraints.rim_slope_match_out import (
            matching_residual_diagnostics,
        )

        res_diag = matching_residual_diagnostics(
            mesh, minim.global_params, mesh.positions_view()
        )
        phi_val = res_diag.get("phi_mean", 0.0)
        z_span = float(np.ptp(mesh.positions_view()[:, 2]))

        breakdown = minim.compute_energy_breakdown()
        e_bt_in = breakdown.get("bending_tilt_in", 0.0)
        e_t_in = breakdown.get("tilt_in", 0.0)
        e_bt_out = breakdown.get("bending_tilt_out", 0.0)
        e_t_out = breakdown.get("tilt_out", 0.0)
        e_cont = breakdown.get("tilt_thetaB_contact_in", 0.0)

        e_in = e_bt_in + e_t_in
        e_out = e_bt_out + e_t_out

        # Rim Tilts
        positions = mesh.positions_view()
        radii = np.linalg.norm(positions[:, :2], axis=1)
        rim_rows = np.flatnonzero(np.abs(radii - theory_params.radius) < 5e-3)
        if rim_rows.size:
            r_hat = positions[rim_rows, :2] / radii[rim_rows][:, None]
            t_in_rad = np.mean(
                np.einsum("ij,ij->i", mesh.tilts_in_view()[rim_rows, :2], r_hat)
            )
            t_out_rad = np.mean(
                np.einsum("ij,ij->i", mesh.tilts_out_view()[rim_rows, :2], r_hat)
            )
        else:
            t_in_rad = t_out_rad = float("nan")

        energies.append(e_total)
        phi_stars.append(phi_val)
        z_spans.append(z_span)
        print(
            f"  theta_B={tv:.4f} -> E={e_total:.4f}, phi_mod={phi_val:.4f}, span={z_span:.4f}"
        )
        print(
            f"    [Breakdown In] BT: {e_bt_in:.3f}, T: {e_t_in:.3f} (Sum: {e_in:.3f})"
        )
        print(
            f"    [Breakdown Out] BT: {e_bt_out:.3f}, T: {e_t_out:.3f} (Sum: {e_out:.3f})"
        )
        print(f"    [Breakdown Cont] {e_cont:.3f}")
        print(f"    [Rim] t_in: {t_in_rad:.4f}, t_out: {t_out_rad:.4f}")

    # 4. Find optimal theta_B from quadratic fit
    coeffs = np.polyfit(theta_values, energies, 2)
    theta_opt = -coeffs[1] / (2 * coeffs[0])
    energy_opt = np.polyval(coeffs, theta_opt)

    # Fit phi_star vs theta_B (should be linear phi = m * theta_B)
    m_phi = np.polyfit(theta_values, phi_stars, 1)[0]

    print("\nSimulation Results:")
    print(
        f"  theta_opt:  {theta_opt:.4f} (Factor: {theta_opt / theory.theta_star:.4f})"
    )
    print(f"  energy_opt: {energy_opt:.4f} (Factor: {energy_opt / theory.total:.4f})")
    print(f"  phi/theta:  {m_phi:.4f} (Expected: 0.5 for tensionless)")
    print(f"  Max z-span: {max(z_spans):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
