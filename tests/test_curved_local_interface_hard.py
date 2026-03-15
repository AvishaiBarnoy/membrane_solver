import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geometry.geom_io import load_data, parse_geometry
from modules.constraints import curved_local_interface_hard


def _build_mesh():
    return parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )


def _mean_outer_radial_residual(mesh) -> float:
    data = curved_local_interface_hard._build_matching_data(mesh, mesh.positions_view())
    assert data is not None
    rim_rows = np.asarray(data["rim_rows"], dtype=int)
    valid = np.asarray(data["valid"], dtype=bool)
    r_dir = np.asarray(data["r_dir"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)
    tilts_out = mesh.tilts_out_view()
    residual = np.einsum("ij,ij->i", tilts_out[rim_rows], r_dir) - phi
    return float(np.mean(residual[valid]))


def test_curved_local_interface_hard_dense_matches_sparse_constraint() -> None:
    mesh = _build_mesh()
    positions = mesh.positions_view()

    row_constraints = curved_local_interface_hard.constraint_gradients_tilt_rows_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
    )
    dense_constraints = curved_local_interface_hard.constraint_gradients_tilt_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
    )

    assert row_constraints is not None
    assert dense_constraints is not None
    assert len(row_constraints) == 1
    assert len(dense_constraints) == 1
    assert row_constraints[0][0] is None
    assert dense_constraints[0][0] is None
    rows, vecs = row_constraints[0][1]
    dense = np.zeros_like(positions)
    np.add.at(dense, rows, vecs)
    assert np.allclose(dense_constraints[0][1], dense, atol=1.0e-12)


def test_curved_local_interface_hard_projection_zeros_mean_outer_residual_on_curved_shells() -> (
    None
):
    mesh = _build_mesh()
    data = curved_local_interface_hard._build_matching_data(mesh, mesh.positions_view())
    assert data is not None
    rim_rows = np.asarray(data["rim_rows"], dtype=int)
    valid = np.asarray(data["valid"], dtype=bool)
    r_dir = np.asarray(data["r_dir"], dtype=float)

    tilts_out = mesh.tilts_out_view().copy(order="F")
    profile = np.linspace(-0.05, 0.08, int(np.sum(valid)))
    tilts_out[rim_rows[valid]] = profile[:, None] * r_dir[valid]
    mesh.set_tilts_out_from_array(tilts_out)

    before = _mean_outer_radial_residual(mesh)
    assert abs(before) > 1.0e-8

    curved_local_interface_hard.enforce_tilt_constraint(
        mesh, global_params=mesh.global_parameters
    )

    after = _mean_outer_radial_residual(mesh)
    assert after == pytest.approx(0.0, abs=1.0e-10)


def test_curved_local_interface_hard_projection_zeros_mean_outer_residual_on_flattened_shells() -> (
    None
):
    mesh = _build_mesh()
    positions = mesh.positions_view()
    positions[:, 2] = 0.0
    mesh.increment_version()

    data = curved_local_interface_hard._build_matching_data(mesh, mesh.positions_view())
    assert data is not None
    rim_rows = np.asarray(data["rim_rows"], dtype=int)
    valid = np.asarray(data["valid"], dtype=bool)
    r_dir = np.asarray(data["r_dir"], dtype=float)

    tilts_out = mesh.tilts_out_view().copy(order="F")
    profile = np.linspace(-0.04, 0.03, int(np.sum(valid)))
    tilts_out[rim_rows[valid]] = profile[:, None] * r_dir[valid]
    mesh.set_tilts_out_from_array(tilts_out)

    before = _mean_outer_radial_residual(mesh)
    assert abs(before) > 1.0e-8

    curved_local_interface_hard.enforce_tilt_constraint(
        mesh, global_params=mesh.global_parameters
    )

    after = _mean_outer_radial_residual(mesh)
    assert after == pytest.approx(0.0, abs=1.0e-10)
