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
    build_trace_ring_fixture,
)
from tools.theory_parity_interface_sweep import parse_candidate, rank_rows

ROOT = Path(__file__).resolve().parent.parent


def _ring_radii(doc: dict) -> set[float]:
    return {
        round((float(v[0]) ** 2 + float(v[1]) ** 2) ** 0.5, 12) for v in doc["vertices"]
    }


def test_parse_candidate_supports_base_and_profile_specs() -> None:
    assert parse_candidate("coarse:base") == {"label": "coarse", "mode": "base"}
    assert parse_candidate("default:profile") == {"label": "default", "mode": "profile"}


def test_parse_candidate_rejects_invalid_specs() -> None:
    with pytest.raises(ValueError):
        parse_candidate("broken")
    with pytest.raises(ValueError):
        parse_candidate(":base")


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


def test_build_profiled_fixture_supports_generic_default_profile() -> None:
    base_doc = yaml.safe_load(
        (
            ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
        ).read_text(encoding="utf-8")
    )
    profiled = build_profiled_fixture(
        base_doc=base_doc,
        profile="default",
        lane="physical_edge_default",
    )
    radii = _ring_radii(profiled)
    inner_radius, outer_radius = INTERFACE_PROFILES["default"]
    assert inner_radius in radii
    assert outer_radius in radii
    assert (
        profiled["global_parameters"]["theory_parity_lane"] == "physical_edge_default"
    )


def test_build_trace_ring_fixture_inserts_new_first_outer_ring() -> None:
    base_doc = yaml.safe_load(
        (
            ROOT
            / "tests"
            / "fixtures"
            / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml"
        ).read_text(encoding="utf-8")
    )
    traced = build_trace_ring_fixture(
        base_doc=base_doc,
        label="trace_ring_free_geometry",
        trace_radius=0.50,
        planar_geometry=False,
    )
    radii = _ring_radii(traced)
    assert 0.5 in radii
    assert INTERFACE_PROFILES["default"][0] in radii
    assert (
        traced["global_parameters"]["theory_parity_lane"] == "trace_ring_free_geometry"
    )
    ring_opts = traced["vertices"][len(base_doc["vertices"])][3]
    assert ring_opts["preset"] == "rim"
    assert ring_opts["rim_slope_match_group"] == "rim"
    assert "pin_to_circle" in list(ring_opts.get("constraints") or [])
    assert float(ring_opts["pin_to_circle_radius"]) == pytest.approx(0.5, abs=1.0e-12)
    assert "pin_to_plane" not in list(ring_opts.get("constraints") or [])


def test_build_trace_ring_fixture_can_pin_trace_ring_to_plane() -> None:
    base_doc = yaml.safe_load(
        (
            ROOT
            / "tests"
            / "fixtures"
            / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml"
        ).read_text(encoding="utf-8")
    )
    traced = build_trace_ring_fixture(
        base_doc=base_doc,
        label="trace_ring_planar_geometry",
        trace_radius=0.50,
        planar_geometry=True,
    )
    ring_opts = traced["vertices"][len(base_doc["vertices"])][3]
    assert ring_opts["preset"] == "rim"
    assert ring_opts["rim_slope_match_group"] == "rim"
    assert "pin_to_circle" in list(ring_opts.get("constraints") or [])
    assert "pin_to_plane" in list(ring_opts.get("constraints") or [])


def test_rank_rows_prefers_score_then_runtime_then_label() -> None:
    rows = [
        {"label": "b", "score": 0.2, "runtime_s": 3.0},
        {"label": "a", "score": 0.2, "runtime_s": 2.0},
        {"label": "c", "score": 0.1, "runtime_s": 9.0},
    ]
    ranked = rank_rows(rows)
    assert [row["label"] for row in ranked] == ["c", "a", "b"]
