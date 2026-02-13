import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from modules.constraints import tilt_leaflet_match_rim


def _ring(n_theta: int, r: float, z: float, opts: dict) -> list[list]:
    verts: list[list] = []
    for i in range(n_theta):
        theta = 2.0 * np.pi * i / float(n_theta)
        x = float(r * np.cos(theta))
        y = float(r * np.sin(theta))
        verts.append([x, y, z, dict(opts)])
    return verts


def _build_mesh(n_theta: int = 12):
    rim_opts = {"tilt_leaflet_match_group": "rim"}
    vertices = _ring(n_theta, 1.0, 0.0, rim_opts)
    edges = [[i, (i + 1) % n_theta] for i in range(n_theta)]
    faces = []
    return parse_geometry(
        {
            "global_parameters": {"tilt_leaflet_match_group": "rim"},
            "constraint_modules": ["tilt_leaflet_match_rim"],
            "energy_modules": [],
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "instructions": [],
        }
    )


def test_tilt_leaflet_match_rim_enforces_in_plane_matching():
    mesh = _build_mesh()
    mesh.build_position_cache()
    n = len(mesh.vertex_ids)
    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")

    rows = np.arange(n, dtype=int)
    tilts_in[rows, 0] = 1.2
    tilts_out[rows, 1] = -0.8
    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)

    tilt_leaflet_match_rim.enforce_tilt_constraint(
        mesh, global_params=mesh.global_parameters
    )

    out_in = mesh.tilts_in_view()
    out_out = mesh.tilts_out_view()
    assert np.allclose(out_in[rows, :2], out_out[rows, :2], atol=1e-10)


def test_tilt_leaflet_match_rim_row_constraints_match_dense():
    mesh = _build_mesh()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    dense = tilt_leaflet_match_rim.constraint_gradients_tilt_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=index_map,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
    )
    rows = tilt_leaflet_match_rim.constraint_gradients_tilt_rows_array(
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
        assert dense_in is not None and dense_out is not None
        assert row_in is not None and row_out is not None
        rebuilt_in = np.zeros_like(positions)
        rebuilt_out = np.zeros_like(positions)
        np.add.at(rebuilt_in, row_in[0], row_in[1])
        np.add.at(rebuilt_out, row_out[0], row_out[1])
        assert np.allclose(rebuilt_in, dense_in, atol=1e-12, rtol=0.0)
        assert np.allclose(rebuilt_out, dense_out, atol=1e-12, rtol=0.0)
