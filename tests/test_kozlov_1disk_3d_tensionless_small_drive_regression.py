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


def _mesh_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "meshes",
        "caveolin",
        "kozlov_1disk_3d_tensionless_single_leaflet_source.yaml",
    )


def _collect_rows(mesh, key: str, value: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get(key) == value:
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    return np.asarray(rows, dtype=int)


@pytest.mark.regression
def test_kozlov_1disk_3d_tensionless_small_drive_physical_scaling() -> None:
    """Regression: small-drive tensionless behavior with κ=1, k_t≈135 (1 unit=15nm).

    This test guards the qualitative paper behavior:
    - single-leaflet source in tilt_in induces curvature + tilt_out through shape coupling
    - the opposite leaflet responds (nonzero tilt_out), but equality is not expected
      here because the source explicitly breaks leaflet symmetry (single-leaflet drive)
    - rim matching kinematics are satisfied at the disk boundary (γ=0 tensionless)
    """
    mesh = parse_geometry(load_data(_mesh_path()))

    mesh.global_parameters.update(
        {
            "surface_tension": 0.0,
            "bending_modulus_in": 1.0,
            "bending_modulus_out": 1.0,
            "tilt_modulus_in": 135.0,
            "tilt_modulus_out": 135.0,
            "tilt_rim_source_strength_in": 5000.0,
            "tilt_solve_mode": "coupled",
            "tilt_step_size": 0.03,
            "tilt_inner_steps": 20,
            "tilt_tol": 1e-12,
            "step_size": 0.005,
            "step_size_mode": "fixed",
        }
    )

    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    minim.minimize(n_steps=25)

    positions = mesh.positions_view()
    z_span = float(np.ptp(positions[:, 2]))
    assert z_span > 1e-10

    rim_rows = _collect_rows(mesh, "rim_slope_match_group", "rim")
    outer_rows = _collect_rows(mesh, "rim_slope_match_group", "outer")
    disk_rows = _collect_rows(mesh, "rim_slope_match_group", "disk")
    assert rim_rows.size and outer_rows.size and disk_rows.size

    rim_pos = positions[rim_rows]
    outer_pos = positions[outer_rows]
    r_rim = np.linalg.norm(rim_pos[:, :2], axis=1)
    r_outer = np.linalg.norm(outer_pos[:, :2], axis=1)
    dr = np.maximum(r_outer - r_rim, 1e-9)
    phi = (outer_pos[:, 2] - rim_pos[:, 2]) / dr

    r_hat = np.zeros_like(rim_pos)
    good = r_rim > 1e-12
    r_hat[good, 0] = rim_pos[good, 0] / r_rim[good]
    r_hat[good, 1] = rim_pos[good, 1] / r_rim[good]

    t_in = mesh.tilts_in_view()
    t_out = mesh.tilts_out_view()

    theta_out_rim = np.einsum("ij,ij->i", t_out[rim_rows], r_hat)
    assert np.max(np.abs(theta_out_rim - phi)) < 2e-2

    disk_pos = positions[disk_rows]
    r_disk = np.linalg.norm(disk_pos[:, :2], axis=1)
    r_hat_disk = np.zeros_like(disk_pos)
    good_d = r_disk > 1e-12
    r_hat_disk[good_d, 0] = disk_pos[good_d, 0] / r_disk[good_d]
    r_hat_disk[good_d, 1] = disk_pos[good_d, 1] / r_disk[good_d]
    theta_disk_in = np.einsum("ij,ij->i", t_in[disk_rows], r_hat_disk)

    theta_in_rim = np.einsum("ij,ij->i", t_in[rim_rows], r_hat)
    if theta_disk_in.size == theta_in_rim.size:
        assert np.max(np.abs(theta_in_rim - (theta_disk_in - phi))) < 2e-2

    mags_in = np.linalg.norm(t_in, axis=1)
    mags_out = np.linalg.norm(t_out, axis=1)

    r_all = np.linalg.norm(positions[:, :2], axis=1)
    annulus_mask = (r_all > 1.1) & (r_all < 5.5)
    assert np.any(annulus_mask)

    mean_in = float(np.mean(mags_in[annulus_mask]))
    mean_out = float(np.mean(mags_out[annulus_mask]))
    assert mean_in > 1e-12
    assert mean_out > 1e-12
    ratio = mean_out / max(mean_in, 1e-30)
    assert ratio > 1e-6
