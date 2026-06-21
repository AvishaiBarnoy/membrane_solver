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
