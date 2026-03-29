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


def test_build_profiled_fixture_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError):
        build_profiled_fixture(base_doc={"vertices": []}, profile="unknown")


def test_default_family_aliases_match_physical_edge_reference_profiles() -> None:
    assert (
        INTERFACE_PROFILES["default_lo"]
        == INTERFACE_PROFILES["physical_edge_family_lo"]
    )
    assert (
        INTERFACE_PROFILES["default"] == INTERFACE_PROFILES["physical_edge_primary_v1"]
    )
    assert (
        INTERFACE_PROFILES["default_hi"]
        == INTERFACE_PROFILES["physical_edge_family_hi"]
    )
