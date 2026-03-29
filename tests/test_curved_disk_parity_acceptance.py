"""Acceptance tests for curved disk parity."""

from __future__ import annotations

import numpy as np
import pytest

from tools.diagnostics.curved_disk_theory import (
    CurvedDiskTheoryParams,
    compute_curved_disk_theory,
    tex_reference_params,
)


def test_curved_theory_tensionless_baseline():
    """Verify that the curved theory matches the Section 2.2 benchmark in 1_disk_3d.tex."""
    params = tex_reference_params()
    res = compute_curved_disk_theory(params)

    # Values from docs/1_disk_3d.tex Section 2.2 and 2.3
    # A ~= 34.04, B ~= 12.57 -> theta_B* = B/(2A) ~= 0.1846
    # Total Energy ~= -1.16

    assert res.theta_star == pytest.approx(0.1846, abs=1e-3)
    assert res.total == pytest.approx(-1.16, abs=1e-2)
    assert res.phi_star == pytest.approx(res.theta_star / 2.0)


def test_curved_theory_finite_tension():
    """Sanity check for finite-tension theory logic."""
    params = CurvedDiskTheoryParams(
        kappa=1.0, kappa_t=100.0, radius=1.0, drive=1.0, surface_tension=1.0
    )
    res = compute_curved_disk_theory(params)

    assert res.psi > 0.0
    assert res.mu < 1.0
    assert res.theta_star > 0.0
    assert res.phi_star > res.theta_star / 2.0  # phi = theta / (2*mu) and mu < 1


@pytest.mark.slow
@pytest.mark.acceptance
def test_reproduce_curved_disk_parity_smoke():
    """Smoke test for the reproduction tool (moderate resolution)."""
    from pathlib import Path

    from tools.reproduce_curved_disk_parity import (
        _configure_minimizer,
        _load_mesh_from_fixture,
        _run_relaxation,
    )

    ROOT = Path(__file__).resolve().parents[1]
    fixture_path = (
        ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
    )

    mesh = _load_mesh_from_fixture(fixture_path)
    from runtime.refinement import refine_triangle_mesh

    for _ in range(1):  # Low refinement for CI speed
        mesh = refine_triangle_mesh(mesh)

    theory_params = tex_reference_params()
    theory = compute_curved_disk_theory(theory_params)

    gp = mesh.global_parameters
    gp.set("bending_modulus_in", theory_params.kappa)
    gp.set("bending_modulus_out", theory_params.kappa)
    gp.set("tilt_modulus_in", theory_params.kappa_t)
    gp.set("tilt_modulus_out", theory_params.kappa_t)
    gp.set("tilt_thetaB_contact_strength_in", theory_params.drive)
    gp.set("surface_tension", theory_params.surface_tension)
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_inner_steps", 100)
    gp.set("step_size", 0.01)

    # Minimal theory alignment needed for energy to be negative
    gp.set("leaflet_out_absent_presets", ["disk", "rim_active"])
    gp.set("leaflet_in_absent_presets", ["rim_active"])
    gp.set("bending_tilt_assume_J0_presets_in", ["disk", "rim_active"])
    gp.set("rim_slope_match_group", "disk_rim")
    gp.set("rim_slope_match_outer_group", "near_disk")
    gp.set("bending_tilt_in_gradient_mode", "finite_difference")
    gp.set("bending_tilt_out_gradient_mode", "finite_difference")

    # Tag vertices
    R = theory_params.radius
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    rim_mask = np.abs(radii - R) < 1e-3
    near_mask = (radii > R + 1e-3) & (radii < R + 0.5)

    import copy

    for row, vid in enumerate(mesh.vertex_ids):
        v = mesh.vertices[int(vid)]
        v.options = copy.deepcopy(v.options)
        if rim_mask[row]:
            v.options["rim_slope_match_group"] = "disk_rim"
            v.options["preset"] = "rim_active"
        if near_mask[row]:
            v.options["rim_slope_match_outer_group"] = "near_disk"

    minim = _configure_minimizer(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)

    # Run a single relaxation at the theoretical optimum
    energy = _run_relaxation(minim, theta_value=theory.theta_star, shape_steps=5)

    assert np.isfinite(energy)
    # At low res, energy starts negative but can diverge if steps are too large
    assert energy < 0.0
