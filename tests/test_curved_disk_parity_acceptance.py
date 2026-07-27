"""Acceptance tests for curved disk parity."""

from __future__ import annotations

import numpy as np
import pytest

from tools.diagnostics.curved_disk_theory import (
    CurvedDiskTheoryParams,
    axisymmetric_ring_topology_diagnostics,
    compare_curved_branch_energy_breakdowns,
    compare_tensionless_curved_disk_profiles,
    compute_curved_disk_theory,
    evaluate_tensionless_curved_disk_profiles,
    radial_leaflet_bending_tilt_bands,
    tex_reference_params,
    topological_leaflet_bending_tilt_regions,
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


def test_tensionless_curved_theory_profiles_match_rim_and_outer_shape_law():
    result = compute_curved_disk_theory(tex_reference_params())
    radius = float(result.params.radius)
    radii = np.array([0.0, 0.5 * radius, radius, 2.0 * radius, 4.0 * radius])

    fields = evaluate_tensionless_curved_disk_profiles(result=result, radii=radii)

    assert fields["tilt_disk"][0] == pytest.approx(0.0, abs=1.0e-15)
    assert fields["tilt_disk"][2] == pytest.approx(result.theta_star)
    assert fields["tilt_outer"][2] == pytest.approx(result.phi_star)
    assert fields["tilt_in"][1] == pytest.approx(fields["tilt_disk"][1])
    assert fields["tilt_in"][3] == pytest.approx(fields["tilt_outer"][3])
    assert fields["tilt_out"] == pytest.approx(fields["tilt_outer"])
    assert fields["slope"][2] == pytest.approx(result.phi_star)
    assert fields["slope"][3] == pytest.approx(result.phi_star / 2.0)
    assert fields["height"][3] == pytest.approx(result.phi_star * radius * np.log(2.0))
    assert fields["slope"][4] * radii[4] == pytest.approx(fields["slope"][3] * radii[3])


def test_curved_profile_comparison_is_function_level_and_height_gauge_invariant():
    result = compute_curved_disk_theory(tex_reference_params())
    radii = np.linspace(0.0, 4.0 * result.params.radius, 41)
    fields = evaluate_tensionless_curved_disk_profiles(result=result, radii=radii)
    weights = np.linspace(1.0, 2.0, radii.size)

    metrics = compare_tensionless_curved_disk_profiles(
        result=result,
        radii=radii,
        height=fields["height"] + 3.25,
        slope=fields["slope"],
        tilt_in_radial=fields["tilt_in"],
        tilt_out_radial=fields["tilt_out"],
        weights=weights,
    )

    assert metrics["height_gauge_offset"] == pytest.approx(3.25)
    assert metrics["height_rel_l2"] < 1.0e-14
    assert metrics["slope_rel_l2"] < 1.0e-14
    assert metrics["tilt_in_rel_l2"] < 1.0e-14
    assert metrics["tilt_out_rel_l2"] < 1.0e-14


def test_curved_branch_energy_comparison_identifies_ordering_reversal_source():
    localized = {
        "bending_tilt_in": 0.4108162957,
        "bending_tilt_out": 0.0330533117,
        "tilt_in": 0.8284561168,
        "tilt_out": 0.0104276053,
        "tilt_thetaB_contact_in": -2.4451936808,
    }
    trumpet = {
        "bending_tilt_in": 0.4580404649,
        "bending_tilt_out": 0.0288245940,
        "tilt_in": 0.8300421431,
        "tilt_out": 0.0098469261,
        "tilt_thetaB_contact_in": -2.4451936808,
    }

    comparison = compare_curved_branch_energy_breakdowns(
        reference=localized,
        candidate=trumpet,
    )

    assert comparison["total_delta"] == pytest.approx(0.0440007986, abs=1.0e-9)
    assert comparison["dominant_module"] == "bending_tilt_in"
    assert comparison["dominant_delta"] == pytest.approx(0.0472241692, abs=1.0e-9)
    assert comparison["module_deltas"]["bending_tilt_out"] == pytest.approx(
        -0.0042287177,
        abs=1.0e-9,
    )


def test_radial_leaflet_bands_conserve_triangle_energy():
    from pathlib import Path

    from geometry.geom_io import load_data, parse_geometry
    from modules.energy.bt_diagnostics import _total_energy_leaflet

    root = Path(__file__).resolve().parents[1]
    mesh = parse_geometry(
        load_data(root / "tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    positions = mesh.positions_view().copy()
    tilts = mesh.tilts_in_view().copy()
    radial_edges = np.array([0.0, 7.0 / 15.0, 1.0, 3.0, 13.0])

    report = radial_leaflet_bending_tilt_bands(
        mesh=mesh,
        positions=positions,
        tilts=tilts,
        radial_edges=radial_edges,
        cache_tag="in",
    )
    total = _total_energy_leaflet(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        tilts=tilts,
        kappa_key="bending_modulus_in",
        cache_tag="in",
        div_sign=-1.0,
    )

    assert report["total_energy"] == pytest.approx(total)
    assert report["assigned_energy"] == pytest.approx(total)
    assert report["unassigned_energy"] == pytest.approx(0.0, abs=1.0e-15)
    assert sum(band["triangle_count"] for band in report["bands"]) == len(mesh.facets)


def test_topological_leaflet_regions_conserve_and_partition_triangle_energy():
    from pathlib import Path

    from geometry.geom_io import load_data, parse_geometry

    root = Path(__file__).resolve().parents[1]
    mesh = parse_geometry(
        load_data(root / "tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    report = topological_leaflet_bending_tilt_regions(
        mesh=mesh,
        positions=mesh.positions_view().copy(),
        tilts=mesh.tilts_in_view().copy(),
        cache_tag="in",
    )
    regions = {region["name"]: region for region in report["regions"]}

    assert report["assigned_energy"] == pytest.approx(report["total_energy"])
    assert report["unassigned_energy"] == pytest.approx(0.0, abs=1.0e-15)
    assert sum(region["triangle_count"] for region in regions.values()) == len(
        mesh.facets
    )
    assert regions["disk"]["triangle_count"] > 0
    assert regions["rim_spanning"]["triangle_count"] > 0
    assert regions["outer"]["triangle_count"] > 0


def test_curved_profile_topology_audit_separates_clean_and_folded_lanes():
    from pathlib import Path

    from geometry.geom_io import load_data, parse_geometry

    root = Path(__file__).resolve().parents[1]
    clean = parse_geometry(
        load_data(root / "tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    folded = parse_geometry(
        load_data(
            root / "tests/fixtures/"
            "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml"
        )
    )

    clean_report = axisymmetric_ring_topology_diagnostics(clean)
    folded_report = axisymmetric_ring_topology_diagnostics(folded)

    assert clean_report["is_monotone"] is True
    assert clean_report["inversion_count"] == 0
    assert folded_report["is_monotone"] is False
    assert folded_report["inversions"] == [
        {"inner_radius": 2.833333333, "outer_radius": 2.66}
    ]


def test_clean_curved_profile_lane_represents_tensionless_log_shape():
    from pathlib import Path

    import yaml

    from geometry.geom_io import parse_geometry
    from modules.energy.bt_diagnostics import _total_energy_leaflet
    from tools.theory_parity_interface_profiles import build_curved_profile_fixture

    root = Path(__file__).resolve().parents[1]
    base_doc = yaml.safe_load(
        (
            root / "tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml"
        ).read_text(encoding="utf-8")
    )
    result = compute_curved_disk_theory(tex_reference_params())
    doc = build_curved_profile_fixture(
        base_doc=base_doc,
        lane="curved_profile_representability",
        trace_radius=result.params.radius + 0.005,
    )
    mesh = parse_geometry(doc)
    mesh.build_position_cache()
    positions = mesh.positions_view().copy()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    fields = evaluate_tensionless_curved_disk_profiles(result=result, radii=radii)
    positions[:, 2] = fields["height"]
    zero_tilts = np.zeros_like(positions)

    topology = axisymmetric_ring_topology_diagnostics(mesh)
    energy_in = _total_energy_leaflet(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        tilts=zero_tilts,
        kappa_key="bending_modulus_in",
        cache_tag="in",
        div_sign=-1.0,
    )
    energy_out = _total_energy_leaflet(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        tilts=zero_tilts,
        kappa_key="bending_modulus_out",
        cache_tag="out",
        div_sign=1.0,
    )

    assert topology["is_monotone"] is True
    assert energy_in < 1.0e-3
    assert energy_out < 1.0e-3


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
