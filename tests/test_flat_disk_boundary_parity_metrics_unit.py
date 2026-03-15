import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geometry.geom_io import load_data, parse_geometry
from tools.reproduce_flat_disk_one_leaflet import (
    _boundary_at_R_parity_metrics,
    _collect_disk_boundary_rows,
    _collect_group_rows,
    _fit_outer_radial_slope_samples,
    _order_rows_by_angle,
)


def _radial_unit_vectors_local(positions: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    return r_hat


def test_boundary_at_R_parity_metrics_recovers_controlled_kink_and_leaflet_tilts() -> (
    None
):
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    positions = mesh.positions_view()
    disk_r = np.linalg.norm(positions[:, :2], axis=1)

    disk_rows = _order_rows_by_angle(
        positions, _collect_disk_boundary_rows(mesh, group="disk")
    )
    rim_rows = _order_rows_by_angle(
        positions,
        _collect_group_rows(mesh, option_key="rim_slope_match_group", group="rim"),
    )
    radii = np.linalg.norm(positions[:, :2], axis=1)
    rim_radius = float(np.median(radii[rim_rows]))
    outer_radii = np.unique(np.round(radii[radii > rim_radius + 1.0e-9], 12))
    outer_radius = float(outer_radii[0])
    outer_rows = _order_rows_by_angle(
        positions,
        np.flatnonzero(np.isclose(radii, outer_radius, atol=1.0e-9)),
    )

    phi_target = 0.2
    theta_in_target = 0.3
    theta_out_target = 0.2
    theta_disk_target = theta_in_target + phi_target

    r_hat_disk = _radial_unit_vectors_local(positions[disk_rows])
    r_hat_rim = _radial_unit_vectors_local(positions[rim_rows])

    tilts_in = mesh.tilts_in_view().copy()
    tilts_out = mesh.tilts_out_view().copy()
    tilts_in[disk_rows] = theta_disk_target * r_hat_disk
    tilts_in[rim_rows] = theta_in_target * r_hat_rim
    tilts_out[rim_rows] = theta_out_target * r_hat_rim
    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)

    rim_r = np.linalg.norm(positions[rim_rows, :2], axis=1)
    outer_r = np.linalg.norm(positions[outer_rows, :2], axis=1)
    for radius in outer_radii[:3]:
        shell_rows = _order_rows_by_angle(
            positions,
            np.flatnonzero(np.isclose(radii, float(radius), atol=1.0e-9)),
        )
        shell_r = np.linalg.norm(positions[shell_rows, :2], axis=1)
        shell_z = positions[rim_rows, 2] + phi_target * np.maximum(
            shell_r - rim_r, 1.0e-6
        )
        for row, z_val in zip(shell_rows, shell_z):
            vid = int(mesh.vertex_ids[int(row)])
            mesh.vertices[vid].position[2] = float(z_val)
    mesh.increment_version()

    metrics = _boundary_at_R_parity_metrics(mesh, theory_theta_value=0.4)

    assert metrics["sample_count"] > 0
    assert metrics["available"] is True
    assert metrics["disk_source"] == "disk_boundary_group"
    assert metrics["rim_source"] == "first_shell_outside_disk"
    assert metrics["outer_source"] == "second_shell_outside_disk"
    assert metrics["outer_slope_estimator"] == "outer_linear_fit"
    assert int(metrics["outer_slope_shell_count"]) >= 1
    assert int(metrics["disk_count"]) == len(disk_rows)
    assert int(metrics["rim_count"]) == len(rim_rows)
    assert int(metrics["outer_count"]) == len(outer_rows)
    assert float(metrics["disk_radius"]) == pytest.approx(
        np.median(disk_r[disk_rows]), abs=1e-12
    )
    assert float(metrics["rim_radius"]) == pytest.approx(np.median(rim_r), abs=1e-12)
    assert float(metrics["outer_radius"]) == pytest.approx(
        np.median(outer_r), abs=1e-12
    )
    assert float(metrics["kink_angle_mesh_median"]) == pytest.approx(
        phi_target, abs=1e-12
    )
    assert float(metrics["tilt_in_mesh_median"]) == pytest.approx(
        theta_in_target, abs=1e-12
    )
    assert float(metrics["tilt_out_mesh_median"]) == pytest.approx(
        theta_out_target, abs=1e-12
    )
    assert float(metrics["tilt_in_disk_mesh_median"]) == pytest.approx(
        theta_disk_target, abs=1e-12
    )
    assert float(metrics["kink_angle_theory"]) == pytest.approx(0.2, abs=1e-12)
    assert float(metrics["tilt_in_theory"]) == pytest.approx(0.2, abs=1e-12)
    assert float(metrics["tilt_out_theory"]) == pytest.approx(0.2, abs=1e-12)


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
    assert np.allclose(phi_linear, phi_target, atol=1e-12)
    assert np.allclose(phi_fd, phi_target, atol=1e-12)
    assert np.allclose(phi_fd, phi_linear, atol=1e-12)


def test_boundary_at_R_parity_metrics_accepts_multistencil_fd_mode() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    metrics = _boundary_at_R_parity_metrics(
        mesh,
        theory_theta_value=0.4,
        outer_slope_estimator="outer_multistencil_fd",
    )

    assert metrics["available"] is True
    assert metrics["outer_slope_estimator"] == "outer_multistencil_fd"
    assert int(metrics["outer_slope_shell_count"]) == 3
    assert np.isfinite(float(metrics["kink_angle_mesh_median"]))
