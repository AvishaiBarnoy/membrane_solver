import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import yaml

from tools.theory_parity_interface_profiles import (
    INTERFACE_PROFILES,
    SOURCE_INNER_RADIUS,
    SOURCE_OUTER_RADIUS,
    build_curved_profile_fixture,
    build_free_outer_refinement_fixture,
    build_full_physics_fixture,
    build_full_physics_trace_fixture,
    build_profiled_fixture,
    build_scaled_fixture,
)

ROOT = Path(__file__).resolve().parent.parent


def _ring_radii(doc: dict) -> set[float]:
    return {
        round((float(v[0]) ** 2 + float(v[1]) ** 2) ** 0.5, 12) for v in doc["vertices"]
    }


def test_build_scaled_fixture_moves_target_rings_and_sets_lane() -> None:
    base_doc = yaml.safe_load(
        (
            ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
        ).read_text(encoding="utf-8")
    )
    scaled = build_scaled_fixture(
        base_doc=base_doc,
        label="candidate_lane",
        inner_radius=0.85,
        outer_radius=2.2,
    )
    radii = _ring_radii(scaled)
    assert 0.85 in radii
    assert 2.2 in radii
    assert SOURCE_INNER_RADIUS not in radii
    assert SOURCE_OUTER_RADIUS not in radii
    assert scaled["global_parameters"]["theory_parity_lane"] == "candidate_lane"
    assert (
        scaled["global_parameters"]["bending_tilt_base_term_reference_mode"]
        == "current_geometry"
    )


def test_build_profiled_fixture_applies_general_near_edge_profile() -> None:
    base_doc = yaml.safe_load(
        (
            ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
        ).read_text(encoding="utf-8")
    )
    profiled = build_profiled_fixture(
        base_doc=base_doc,
        profile="near_edge_v1",
        lane="general_near_edge_v1",
    )
    radii = _ring_radii(profiled)
    inner_radius, outer_radius = INTERFACE_PROFILES["near_edge_v1"]
    assert inner_radius in radii
    assert outer_radius in radii
    assert profiled["global_parameters"]["theory_parity_lane"] == "general_near_edge_v1"
    assert (
        profiled["global_parameters"]["bending_tilt_base_term_reference_mode"]
        == "current_geometry"
    )


def test_profile_builder_can_still_request_legacy_zero_j_reference() -> None:
    base_doc = yaml.safe_load(
        (
            ROOT / "tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml"
        ).read_text(encoding="utf-8")
    )
    profiled = build_profiled_fixture(
        base_doc=base_doc,
        profile="near_edge_v1",
        lane="legacy_control",
        base_term_reference_mode="flat_reference_zero_J0",
    )

    assert (
        profiled["global_parameters"]["bending_tilt_base_term_reference_mode"]
        == "flat_reference_zero_J0"
    )


def test_build_full_physics_fixture_sets_current_geometry_reference_mode() -> None:
    base_doc = yaml.safe_load(
        (
            ROOT
            / "tests"
            / "fixtures"
            / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml"
        ).read_text(encoding="utf-8")
    )
    full = build_full_physics_fixture(
        base_doc=base_doc, lane="physical_edge_full_coupling_v1"
    )
    assert (
        full["global_parameters"]["theory_parity_lane"]
        == "physical_edge_full_coupling_v1"
    )
    assert (
        full["global_parameters"]["bending_tilt_base_term_reference_mode"]
        == "current_geometry"
    )


def test_build_full_physics_trace_fixture_adds_trace_ring_and_current_geometry() -> (
    None
):
    base_doc = yaml.safe_load(
        (
            ROOT
            / "tests"
            / "fixtures"
            / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml"
        ).read_text(encoding="utf-8")
    )
    full = build_full_physics_trace_fixture(
        base_doc=base_doc,
        lane="physical_edge_full_coupling_trace_eps005_v1",
        trace_radius=(7.0 / 15.0) + 0.005,
    )
    radii = _ring_radii(full)
    assert 0.471666666667 in radii
    assert (
        full["global_parameters"]["theory_parity_lane"]
        == "physical_edge_full_coupling_trace_eps005_v1"
    )
    assert (
        full["global_parameters"]["bending_tilt_base_term_reference_mode"]
        == "current_geometry"
    )


def test_build_curved_profile_fixture_frees_outer_height_and_uses_physical_edge():
    base_doc = yaml.safe_load(
        (
            ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
        ).read_text(encoding="utf-8")
    )
    trace_radius = (7.0 / 15.0) + 0.005

    curved = build_curved_profile_fixture(
        base_doc=base_doc,
        lane="curved_profile_clean",
        trace_radius=trace_radius,
    )

    assert round(trace_radius, 12) in _ring_radii(curved)
    assert curved["global_parameters"]["bending_tilt_base_term_reference_mode"] == (
        "current_geometry"
    )
    assert (
        curved["global_parameters"]["rim_slope_match_mode"]
        == "physical_edge_staggered_v1"
    )
    assert (
        curved["global_parameters"]["rim_slope_match_scaffold_projector_mode"]
        == "continuity_v2"
    )
    assert (
        curved["global_parameters"]["bending_tilt_base_term_boundary_group_out"]
        == "disk"
    )
    assert curved["global_parameters"]["theory_radius"] == pytest.approx(7.0 / 15.0)
    outer_constraints = curved["definitions"]["outer_rim"]["constraints"]
    assert "pin_to_circle" in outer_constraints
    assert "pin_to_plane" not in outer_constraints
    assert curved["definitions"]["outer_rim"]["pin_to_circle_mode"] == "slide"


def test_curved_profile_fixture_projects_the_theoretical_half_split():
    import numpy as np

    from geometry.geom_io import parse_geometry
    from modules.constraints.rim_slope_match_out import enforce_constraint
    from modules.constraints.tilt_thetaB_boundary_in import (
        enforce_tilt_constraint as enforce_theta_boundary,
    )

    base_doc = yaml.safe_load(
        (
            ROOT / "tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml"
        ).read_text(encoding="utf-8")
    )
    radius = 7.0 / 15.0
    trace_radius = radius + 0.005
    theta = 0.2
    doc = build_curved_profile_fixture(
        base_doc=base_doc,
        lane="curved_profile_half_split",
        trace_radius=trace_radius,
    )
    mesh = parse_geometry(doc)
    mesh.global_parameters.set("tilt_thetaB_value", theta)

    enforce_constraint(mesh, mesh.global_parameters, context="minimize")
    enforce_theta_boundary(mesh, mesh.global_parameters)

    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    trace_rows = np.flatnonzero(np.isclose(radii, trace_radius))
    trace_height = float(np.mean(positions[trace_rows, 2]))
    radial = positions[trace_rows].copy()
    radial[:, 2] = 0.0
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    trace_tilt_out = float(
        np.mean(np.einsum("ij,ij->i", mesh.tilts_out_view()[trace_rows], radial))
    )
    trace_tilt_in = float(
        np.mean(np.einsum("ij,ij->i", mesh.tilts_in_view()[trace_rows], radial))
    )

    assert trace_height / (trace_radius - radius) == pytest.approx(0.5 * theta)
    assert trace_tilt_in == pytest.approx(0.5 * theta)
    assert trace_tilt_out == pytest.approx(0.5 * theta)


def test_free_outer_refinement_rings_have_no_support_constraints():
    base_doc = yaml.safe_load(
        (
            ROOT / "tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml"
        ).read_text(encoding="utf-8")
    )
    radius = 7.0 / 15.0
    free_radii = [radius + 0.025, radius + 0.05, radius + 0.1]
    doc = build_free_outer_refinement_fixture(
        base_doc=base_doc,
        label="free_outer_refinement",
        trace_radius=radius + 0.005,
        free_radii=free_radii,
    )

    assert (
        doc["global_parameters"]["bending_tilt_base_term_reference_mode"]
        == "current_geometry"
    )
    assert (
        doc["global_parameters"]["rim_slope_match_mode"] == "physical_edge_staggered_v1"
    )
    assert (
        doc["global_parameters"]["rim_slope_match_scaffold_projector_mode"]
        == "continuity_v2"
    )
    assert doc["global_parameters"]["parity_outer_shells"] == 0
    for ring_index, ring_radius in enumerate(free_radii, start=1):
        rows = [
            vertex
            for vertex in doc["vertices"]
            if (float(vertex[0]) ** 2 + float(vertex[1]) ** 2) ** 0.5
            == pytest.approx(ring_radius)
        ]
        assert len(rows) == 12
        for vertex in rows:
            options = vertex[3]
            assert options["outer_shell_free_index"] == ring_index
            assert "outer_shell_scaffold_index" not in options
            assert "pin_to_circle" not in options.get("constraints", [])
            assert "pin_to_plane" not in options.get("constraints", [])
            assert "pin_to_circle_group" not in options


def test_build_profiled_fixture_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError):
        build_profiled_fixture(base_doc={"vertices": []}, profile="unknown")


def test_default_family_profiles_define_a_distinct_ordered_construction_rule() -> None:
    default_lo = INTERFACE_PROFILES["default_lo"]
    default = INTERFACE_PROFILES["default"]
    default_hi = INTERFACE_PROFILES["default_hi"]

    assert default_lo is not None
    assert default is not None
    assert default_hi is not None
    assert default_lo[0] > default[0] > default_hi[0]
    assert default_lo[1] > default[1] > default_hi[1]
