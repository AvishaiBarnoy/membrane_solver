import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geometry.geom_io import load_data, parse_geometry
from modules.constraints.local_interface_shells import (
    build_local_interface_shell_data,
    collect_disk_boundary_rows,
    extrapolate_trace_to_radius,
    local_interface_constraint_diagnostics,
    radial_unit_vectors,
)
from tools.theory_parity_interface_profiles import build_trace_ring_fixture


@pytest.mark.unit
def test_collect_disk_boundary_rows_finds_tagged_disk_ring() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )

    rows = collect_disk_boundary_rows(mesh, group="disk")
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[rows, :2], axis=1)

    assert rows.size > 0
    assert float(np.max(radii) - np.min(radii)) < 1.0e-8


@pytest.mark.unit
def test_build_local_interface_shell_data_orders_shells_and_matches_rows() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    positions = mesh.positions_view()

    data = build_local_interface_shell_data(mesh, positions=positions)

    assert data.disk_rows.size > 0
    assert data.rim_rows.size > 0
    assert data.outer_rows.size > 0
    assert data.disk_radius < data.rim_radius < data.outer_radius
    assert data.rim_rows_matched.size == data.outer_rows.size
    assert data.disk_rows_matched.size == data.rim_rows.size
    assert data.rim_rows_for_disk.size == data.disk_rows.size
    assert data.outer_rows_for_rim.size == data.rim_rows.size
    assert data.outer_rows_for_disk.size == data.disk_rows.size
    assert np.unique(data.rim_rows_for_disk).size == data.rim_rows_for_disk.size
    assert np.unique(data.outer_rows_for_rim).size == data.outer_rows_for_rim.size
    assert np.unique(data.outer_rows_for_disk).size == data.outer_rows_for_disk.size

    _, rim_r_hat = radial_unit_vectors(positions[data.rim_rows_matched])
    _, disk_r_hat = radial_unit_vectors(positions[data.disk_rows_matched])
    assert np.allclose(np.linalg.norm(rim_r_hat[:, :2], axis=1), 1.0, atol=1.0e-12)
    assert np.allclose(np.linalg.norm(disk_r_hat[:, :2], axis=1), 1.0, atol=1.0e-12)
    assert set(map(int, data.rim_rows_for_disk.tolist())).issubset(
        set(map(int, data.rim_rows.tolist()))
    )
    assert set(map(int, data.outer_rows_for_rim.tolist())).issubset(
        set(map(int, data.outer_rows.tolist()))
    )
    assert set(map(int, data.outer_rows_for_disk.tolist())).issubset(
        set(map(int, data.outer_rows.tolist()))
    )


@pytest.mark.unit
def test_local_interface_constraint_diagnostics_reports_available_shells() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    positions = mesh.positions_view()

    report = local_interface_constraint_diagnostics(
        mesh,
        positions=positions,
        mode="shared",
        active=True,
    )

    assert report["available"] is True
    assert report["reason"] == "ok"
    assert report["mode"] == "shared"
    assert report["active"] is True
    assert int(report["disk_count"]) > 0
    assert int(report["rim_count"]) > 0
    assert int(report["outer_count"]) > 0
    assert (
        float(report["disk_radius"])
        < float(report["rim_radius"])
        < float(report["outer_radius"])
    )
    assert report["matching_strategy"] == "nearest_azimuth"
    assert report["shell_source"] == "disk_boundary_local_shells"


@pytest.mark.unit
def test_extrapolate_trace_to_radius_uses_two_shells_and_falls_back_to_first_shell() -> (
    None
):
    target_radius = 0.5
    first_radius = 0.7
    second_radius = 0.9
    first_values = np.array([0.2, 0.4], dtype=float)
    second_values = np.array([0.3, 0.6], dtype=float)

    trace = extrapolate_trace_to_radius(
        target_radius=target_radius,
        first_radius=first_radius,
        first_values=first_values,
        second_radius=second_radius,
        second_values=second_values,
    )
    fallback = extrapolate_trace_to_radius(
        target_radius=target_radius,
        first_radius=first_radius,
        first_values=first_values,
    )

    np.testing.assert_allclose(trace, np.array([0.1, 0.2], dtype=float))
    np.testing.assert_allclose(fallback, first_values)
    assert not np.allclose(trace, first_values)


@pytest.mark.unit
def test_build_local_interface_shell_data_honors_trace_layer_radius_override() -> None:
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
        label="trace_layer_unit",
        trace_radius=(7.0 / 15.0) + 0.01,
        planar_geometry=False,
    )
    fixture_path = ROOT / "tests" / "fixtures" / "_tmp_trace_layer_unit.yaml"
    fixture_path.write_text(yaml.safe_dump(traced, sort_keys=False), encoding="utf-8")
    try:
        mesh = parse_geometry(load_data(str(fixture_path)))
        positions = mesh.positions_view()
        data = build_local_interface_shell_data(mesh, positions=positions)
    finally:
        fixture_path.unlink(missing_ok=True)

    assert data.disk_radius == pytest.approx(7.0 / 15.0, abs=1.0e-9)
    assert data.rim_radius == pytest.approx((7.0 / 15.0) + 0.01, abs=1.0e-9)
    assert data.outer_radius > data.rim_radius
