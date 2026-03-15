import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from modules.constraints import rim_slope_match_local_out


def _load_mesh():
    return parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )


def test_local_rim_matching_data_uses_boundary_shell_family() -> None:
    mesh = _load_mesh()
    positions = mesh.positions_view()

    data = rim_slope_match_local_out._build_matching_data(
        mesh, mesh.global_parameters, positions
    )
    assert data is not None

    radii = np.linalg.norm(positions[:, :2], axis=1)
    rim_radius = float(np.median(radii[data["rim_rows"]]))
    outer_radius = float(np.median(radii[data["outer_rows"]]))
    disk_radius = float(np.median(radii[data["disk_rows_matched"]]))

    assert disk_radius < rim_radius < outer_radius
    assert (disk_radius, rim_radius, outer_radius) == pytest.approx(
        (0.466666666666667, 1.0, 2.8333333333333335), abs=1.0e-9
    )
    assert int(np.sum(data["valid"])) > 0


def test_local_rim_matching_dense_matches_sparse_constraint() -> None:
    mesh = _load_mesh()
    positions = mesh.positions_view()

    row_constraints = rim_slope_match_local_out.constraint_gradients_tilt_rows_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
    )
    dense_constraints = rim_slope_match_local_out.constraint_gradients_tilt_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
    )

    assert row_constraints is not None
    assert dense_constraints is not None
    assert len(dense_constraints) == len(row_constraints)

    for (row_in, row_out), (dense_in, dense_out) in zip(
        row_constraints, dense_constraints
    ):
        if row_in is None:
            assert dense_in is None
        else:
            rows, vecs = row_in
            expected_in = np.zeros_like(positions)
            np.add.at(expected_in, rows, vecs)
            assert dense_in is not None
            assert dense_in == pytest.approx(expected_in, abs=1.0e-12)
        if row_out is None:
            assert dense_out is None
        else:
            rows, vecs = row_out
            expected_out = np.zeros_like(positions)
            np.add.at(expected_out, rows, vecs)
            assert dense_out is not None
            assert dense_out == pytest.approx(expected_out, abs=1.0e-12)


def test_local_rim_matching_enforce_tilt_constraint_matches_local_targets() -> None:
    mesh = _load_mesh()
    positions = mesh.positions_view()
    data = rim_slope_match_local_out._build_matching_data(
        mesh, mesh.global_parameters, positions
    )
    assert data is not None

    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")
    tilts_in[:] = 0.0
    tilts_out[:] = 0.0
    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)
    mesh.global_parameters.set("tilt_thetaB_value", 0.4)

    rim_slope_match_local_out.enforce_tilt_constraint(
        mesh, global_params=mesh.global_parameters
    )

    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    normals = mesh.vertex_normals(positions=mesh.positions_view())

    for i, ok in enumerate(data["valid"]):
        if not ok:
            continue
        row = int(data["rim_rows"][i])
        n = normals[row]
        r_dir = data["r_hat"][i] - float(np.dot(data["r_hat"][i], n)) * n
        r_dir /= np.linalg.norm(r_dir)
        t_out_rad = float(np.dot(tilts_out[row], r_dir))
        t_in_rad = float(np.dot(tilts_in[row], r_dir))
        assert t_out_rad == pytest.approx(float(data["phi"][i]), abs=1.0e-9)
        assert t_in_rad == pytest.approx(0.4 - float(data["phi"][i]), abs=1.0e-9)
