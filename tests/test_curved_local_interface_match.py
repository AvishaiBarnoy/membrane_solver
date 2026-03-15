import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geometry.geom_io import load_data, parse_geometry
from modules.constraints import curved_local_interface_match
from modules.constraints.local_interface_shells import build_local_interface_shell_data


def _load_mesh():
    return parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )


def test_curved_local_interface_data_uses_shared_shell_builder() -> None:
    mesh = _load_mesh()
    positions = mesh.positions_view()

    data = curved_local_interface_match._build_local_interface_data(
        mesh,
        mesh.global_parameters,
        positions,
    )
    shell_data = build_local_interface_shell_data(mesh, positions=positions)

    assert data is not None
    assert data["shell_source"] == "disk_boundary_local_shells"
    assert data["matching_strategy"] == "nearest_azimuth"
    assert data["projection_mode"] == "vector_average"
    assert data["disk_radius"] == pytest.approx(shell_data.disk_radius, abs=1e-12)
    assert data["rim_radius"] == pytest.approx(shell_data.rim_radius, abs=1e-12)
    assert data["outer_radius"] == pytest.approx(shell_data.outer_radius, abs=1e-12)
    assert data["disk_rows"].shape == data["rim_rows"].shape
    assert data["rim_rows"].shape == data["outer_rows"].shape


def test_curved_local_interface_dense_tilt_gradients_match_sparse_rows() -> None:
    mesh = _load_mesh()
    positions = mesh.positions_view()

    row_constraints = curved_local_interface_match.constraint_gradients_tilt_rows_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map={},
    )
    dense_constraints = curved_local_interface_match.constraint_gradients_tilt_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map={},
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
