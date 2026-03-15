import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geometry.geom_io import load_data, parse_geometry
from modules.constraints.local_interface_shells import (
    build_local_interface_shell_data,
    collect_disk_boundary_rows,
    local_interface_constraint_diagnostics,
    radial_unit_vectors,
)


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

    _, rim_r_hat = radial_unit_vectors(positions[data.rim_rows_matched])
    _, disk_r_hat = radial_unit_vectors(positions[data.disk_rows_matched])
    assert np.allclose(np.linalg.norm(rim_r_hat[:, :2], axis=1), 1.0, atol=1.0e-12)
    assert np.allclose(np.linalg.norm(disk_r_hat[:, :2], axis=1), 1.0, atol=1.0e-12)


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
