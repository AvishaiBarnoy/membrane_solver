import os
import sys

import numpy as np

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
        "kozlov_1disk_3d_tensionless_bilayer_profile.yaml",
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


def _radial_unit_vectors(positions: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    return r_hat


def test_bilayer_profile_tilts_decay_in_outer_region() -> None:
    mesh = parse_geometry(load_data(_mesh_path()))
    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=60)

    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    rim_rows = _collect_group_rows(mesh, "rim_slope_match_group", "rim")
    assert rim_rows.size
    r_rim = float(np.mean(r[rim_rows]))

    # Exclude the rim itself; the asymmetric rim-matching condition allows
    # t_in != t_out at r=R even in the bilayer profile setup.
    mask = r >= (r_rim + 1e-3)
    rows = np.where(mask)[0]
    assert rows.size

    r_hat_outer = _radial_unit_vectors(positions[rows])
    theta_in_outer = np.einsum("ij,ij->i", mesh.tilts_in_view()[rows], r_hat_outer)
    theta_out_outer = np.einsum("ij,ij->i", mesh.tilts_out_view()[rows], r_hat_outer)

    inner_mask = r <= (r_rim + 1e-6)
    inner_rows = np.where(inner_mask)[0]
    assert inner_rows.size
    r_hat_inner = _radial_unit_vectors(positions[inner_rows])
    theta_in_inner = np.einsum(
        "ij,ij->i", mesh.tilts_in_view()[inner_rows], r_hat_inner
    )
    theta_out_inner = np.einsum(
        "ij,ij->i", mesh.tilts_out_view()[inner_rows], r_hat_inner
    )

    # Expect decay in both leaflets away from the disk.
    outer_in_p90 = float(np.quantile(np.abs(theta_in_outer), 0.9))
    outer_out_p90 = float(np.quantile(np.abs(theta_out_outer), 0.9))
    inner_in_p90 = float(np.quantile(np.abs(theta_in_inner), 0.9))
    inner_out_p90 = float(np.quantile(np.abs(theta_out_inner), 0.9))

    assert outer_in_p90 < 0.3 * (inner_in_p90 + 1e-12)
    assert outer_out_p90 < 0.3 * (inner_out_p90 + 1e-12)
