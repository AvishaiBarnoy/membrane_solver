import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from modules.constraints import tilt_vector_match_rim


def _ring(n_theta: int, r: float, z: float, opts: dict) -> list[list]:
    verts: list[list] = []
    for i in range(n_theta):
        theta = 2.0 * np.pi * i / float(n_theta)
        x = float(r * np.cos(theta))
        y = float(r * np.sin(theta))
        verts.append([x, y, z, dict(opts)])
    return verts


def test_tilt_vector_match_rim_enforces_in_plane_matching_per_leaflet():
    n_theta = 12
    disk_opts = {"tilt_vector_match_group": "cav1", "tilt_vector_match_role": "disk"}
    rim_opts = {"tilt_vector_match_group": "cav1", "tilt_vector_match_role": "rim"}

    vertices = []
    vertices += _ring(n_theta, 0.8, 0.0, disk_opts)
    vertices += _ring(n_theta, 1.0, 0.0, rim_opts)

    # Triangulation isn't needed for this constraint-only test; keep a minimal mesh.
    edges = []
    for i in range(n_theta):
        edges.append([i, (i + 1) % n_theta])
        edges.append([n_theta + i, n_theta + (i + 1) % n_theta])
        edges.append([i, n_theta + i])
    faces = []

    mesh = parse_geometry(
        {
            "global_parameters": {"tilt_vector_match_mode": "average"},
            "constraint_modules": ["tilt_vector_match_rim"],
            "energy_modules": [],
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "instructions": [],
        }
    )

    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")

    disk_rows = np.arange(0, n_theta, dtype=int)
    rim_rows = np.arange(n_theta, 2 * n_theta, dtype=int)
    # Make disk and rim different in-plane tilts.
    tilts_in[disk_rows, 0] = 1.0
    tilts_in[rim_rows, 1] = 2.0
    tilts_out[disk_rows, 0] = -0.5
    tilts_out[rim_rows, 1] = 0.25

    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)

    tilt_vector_match_rim.enforce_tilt_constraint(
        mesh, global_params=mesh.global_parameters
    )

    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()

    # With average mode and symmetric geometry, in-plane components should match.
    assert np.allclose(tilts_in[disk_rows, :2], tilts_in[rim_rows, :2], atol=1e-10)
    assert np.allclose(tilts_out[disk_rows, :2], tilts_out[rim_rows, :2], atol=1e-10)


def test_tilt_vector_match_rim_row_constraints_match_dense():
    n_theta = 10
    disk_opts = {"tilt_vector_match_group": "cav1", "tilt_vector_match_role": "disk"}
    rim_opts = {"tilt_vector_match_group": "cav1", "tilt_vector_match_role": "rim"}

    vertices = _ring(n_theta, 0.8, 0.0, disk_opts) + _ring(n_theta, 1.0, 0.0, rim_opts)
    edges = [[i, (i + 1) % n_theta] for i in range(n_theta)]
    faces = []

    mesh = parse_geometry(
        {
            "global_parameters": {"tilt_vector_match_mode": "average"},
            "constraint_modules": ["tilt_vector_match_rim"],
            "energy_modules": [],
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "instructions": [],
        }
    )

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    dense = tilt_vector_match_rim.constraint_gradients_tilt_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=index_map,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
    )
    rows = tilt_vector_match_rim.constraint_gradients_tilt_rows_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=index_map,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
    )

    assert dense is not None
    assert rows is not None
    assert len(dense) == len(rows)
    for (dense_in, dense_out), (row_in, row_out) in zip(dense, rows):
        if dense_in is None:
            assert row_in is None
        else:
            assert row_in is not None
            rebuilt_in = np.zeros_like(positions)
            np.add.at(rebuilt_in, row_in[0], row_in[1])
            assert np.allclose(rebuilt_in, dense_in, atol=1e-12, rtol=0.0)
        if dense_out is None:
            assert row_out is None
        else:
            assert row_out is not None
            rebuilt_out = np.zeros_like(positions)
            np.add.at(rebuilt_out, row_out[0], row_out[1])
            assert np.allclose(rebuilt_out, dense_out, atol=1e-12, rtol=0.0)
