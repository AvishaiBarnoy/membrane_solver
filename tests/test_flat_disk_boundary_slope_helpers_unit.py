import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geometry.geom_io import load_data, parse_geometry
from tools.reproduce_flat_disk_one_leaflet import (
    _collect_disk_boundary_rows,
    _collect_group_rows,
    _collect_outer_radial_slope_samples,
    _fit_outer_radial_slope_samples,
    _order_rows_by_angle,
)


@pytest.mark.unit
def test_collect_group_rows_finds_rim_shell() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    positions = mesh.positions_view()

    disk_rows = _order_rows_by_angle(
        positions, _collect_disk_boundary_rows(mesh, group="disk")
    )
    rim_rows = _order_rows_by_angle(
        positions,
        _collect_group_rows(mesh, option_key="rim_slope_match_group", group="rim"),
    )

    disk_r = np.linalg.norm(positions[disk_rows, :2], axis=1)
    rim_r = np.linalg.norm(positions[rim_rows, :2], axis=1)

    assert disk_rows.size > 0
    assert rim_rows.size > 0
    assert float(np.median(rim_r)) > float(np.median(disk_r))


@pytest.mark.unit
def test_collect_outer_radial_slope_samples_returns_three_shells() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    positions = mesh.positions_view()
    rim_rows = _order_rows_by_angle(
        positions,
        _collect_group_rows(mesh, option_key="rim_slope_match_group", group="rim"),
    )

    r_matrix, h_matrix, use_radii, used_counts = _collect_outer_radial_slope_samples(
        positions,
        rim_rows_matched=rim_rows,
        shell_count=3,
    )

    assert r_matrix.shape == (rim_rows.size, 4)
    assert h_matrix.shape == (rim_rows.size, 4)
    assert len(use_radii) == 3
    assert used_counts == [int(rim_rows.size)] * 4
    assert all(use_radii[i] < use_radii[i + 1] for i in range(len(use_radii) - 1))


@pytest.mark.unit
def test_outer_multistencil_fd_recovers_controlled_linear_slope() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    positions = mesh.positions_view().copy()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    rim_rows = _order_rows_by_angle(
        positions,
        _collect_group_rows(mesh, option_key="rim_slope_match_group", group="rim"),
    )
    rim_radius = float(np.median(radii[rim_rows]))
    outer_radii = np.unique(np.round(radii[radii > rim_radius + 1.0e-9], 12))

    phi_target = 0.125
    rim_r = np.linalg.norm(positions[rim_rows, :2], axis=1)
    for radius in outer_radii[:3]:
        shell_rows = _order_rows_by_angle(
            positions,
            np.flatnonzero(np.isclose(radii, float(radius), atol=1.0e-9)),
        )
        shell_r = np.linalg.norm(positions[shell_rows, :2], axis=1)
        shell_z = positions[rim_rows, 2] + phi_target * (shell_r - rim_r)
        positions[shell_rows, 2] = shell_z

    phi_linear, meta_linear = _fit_outer_radial_slope_samples(
        positions,
        rim_rows_matched=rim_rows,
        shell_count=3,
        estimator="outer_linear_fit",
    )
    phi_fd, meta_fd = _fit_outer_radial_slope_samples(
        positions,
        rim_rows_matched=rim_rows,
        shell_count=3,
        estimator="outer_multistencil_fd",
    )

    assert meta_linear["outer_slope_estimator"] == "outer_linear_fit"
    assert meta_fd["outer_slope_estimator"] == "outer_multistencil_fd"
    assert int(meta_fd["outer_slope_shell_count"]) == 3
    assert np.allclose(phi_linear, phi_target, atol=1.0e-12)
    assert np.allclose(phi_fd, phi_target, atol=1.0e-12)
    assert np.allclose(phi_fd, phi_linear, atol=1.0e-12)
