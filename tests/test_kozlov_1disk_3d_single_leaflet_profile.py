import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

pytestmark = pytest.mark.e2e


def _mesh_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "meshes",
        "caveolin",
        "kozlov_1disk_3d_tensionless_single_leaflet_profile.yaml",
    )


def _build_minimizer(mesh) -> Minimizer:
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def _collect_group_rows(mesh, key: str, value: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get(key) == value:
            rows.append(mesh.vertex_index_to_row[int(vid)])
    return np.asarray(rows, dtype=int)


def _order_by_angle(positions: np.ndarray) -> np.ndarray:
    angles = np.arctan2(positions[:, 1], positions[:, 0])
    return np.argsort(angles)


def _radial_unit_vectors(positions: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    return r_hat


def _outer_free_ring_rows(mesh, positions: np.ndarray) -> np.ndarray:
    rows: list[int] = []
    radii: list[float] = []
    for vid in mesh.vertex_ids:
        v = mesh.vertices[int(vid)]
        opts = getattr(v, "options", None) or {}
        if opts.get("pin_to_circle_group") == "outer":
            continue
        row = mesh.vertex_index_to_row[int(vid)]
        rows.append(row)
        radii.append(float(np.linalg.norm(positions[row, :2])))
    if not rows:
        return np.zeros(0, dtype=int)
    radii_arr = np.asarray(radii, dtype=float)
    r_max = float(np.max(radii_arr))
    tol = 1e-6
    rows_arr = np.asarray(rows, dtype=int)
    return rows_arr[np.abs(radii_arr - r_max) <= tol]


def test_single_leaflet_profile_behavior() -> None:
    mesh = parse_geometry(load_data(_mesh_path()))
    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=60)

    positions = mesh.positions_view()
    z_span = float(np.ptp(positions[:, 2]))
    assert z_span > 1e-4

    disk_rows = _collect_group_rows(mesh, "tilt_disk_target_group_in", "disk")
    assert disk_rows.size

    rim_rows = _collect_group_rows(mesh, "rim_slope_match_group", "rim")
    outer_rows = _collect_group_rows(mesh, "rim_slope_match_group", "outer")
    disk_ring_rows = _collect_group_rows(mesh, "rim_slope_match_group", "disk")
    assert rim_rows.size and outer_rows.size and disk_ring_rows.size

    rim_rows = rim_rows[_order_by_angle(positions[rim_rows])]
    outer_rows = outer_rows[_order_by_angle(positions[outer_rows])]

    rim_pos = positions[rim_rows]
    outer_pos = positions[outer_rows]
    r_rim = np.linalg.norm(rim_pos[:, :2], axis=1)
    r_outer = np.linalg.norm(outer_pos[:, :2], axis=1)
    dr = np.maximum(r_outer - r_rim, 1e-6)
    phi = float(np.mean((outer_pos[:, 2] - rim_pos[:, 2]) / dr))
    assert abs(phi) > 1e-4

    r_disk = np.linalg.norm(positions[disk_rows, :2], axis=1)
    r_max = float(np.max(r_disk))
    r_hat_disk = _radial_unit_vectors(positions[disk_rows])
    theta_disk = np.einsum(
        "ij,ij->i",
        mesh.tilts_in_view()[disk_rows],
        r_hat_disk,
    )
    inner_band = theta_disk[r_disk < 0.4 * r_max]
    outer_band = theta_disk[r_disk > 0.8 * r_max]
    assert float(np.mean(outer_band)) > float(np.mean(inner_band))

    rim_r_hat = _radial_unit_vectors(rim_pos)
    theta_in_rim = float(
        np.mean(np.einsum("ij,ij->i", mesh.tilts_in_view()[rim_rows], rim_r_hat))
    )
    theta_out_rim = float(
        np.mean(np.einsum("ij,ij->i", mesh.tilts_out_view()[rim_rows], rim_r_hat))
    )
    disk_ring_r_hat = _radial_unit_vectors(positions[disk_ring_rows])
    theta_disk_ring = float(
        np.mean(
            np.einsum(
                "ij,ij->i",
                mesh.tilts_in_view()[disk_ring_rows],
                disk_ring_r_hat,
            )
        )
    )
    denom = max(abs(theta_in_rim), abs(theta_disk_ring - phi), 1e-6)
    assert abs(theta_in_rim - (theta_disk_ring - phi)) / denom < 0.6
    assert abs(theta_out_rim) > 1e-4

    free_rows = _outer_free_ring_rows(mesh, positions)
    assert free_rows.size
    free_r_hat = _radial_unit_vectors(positions[free_rows])
    theta_out_far = float(
        np.mean(
            np.abs(
                np.einsum(
                    "ij,ij->i",
                    mesh.tilts_out_view()[free_rows],
                    free_r_hat,
                )
            )
        )
    )
    assert theta_out_far < 0.7 * abs(theta_out_rim)
