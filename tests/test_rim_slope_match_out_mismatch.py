import math

import numpy as np

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import parse_geometry
from modules.constraints.rim_slope_match_out import (
    constraint_gradients_array,
    constraint_gradients_rows_array,
    constraint_gradients_tilt_array,
    constraint_gradients_tilt_rows_array,
)
from modules.energy.rim_slope_match_out import compute_energy_and_gradient_array


def _ring_vertices(radius: float, count: int, group: str):
    verts = []
    for i in range(count):
        angle = 2.0 * math.pi * i / count
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        verts.append([x, y, 0.0, {"rim_slope_match_group": group}])
    return verts


def _build_mismatch_mesh():
    data = {
        "vertices": _ring_vertices(1.0, 4, "rim") + _ring_vertices(2.0, 8, "outer"),
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "global_parameters": {
            "rim_slope_match_group": "rim",
            "rim_slope_match_outer_group": "outer",
            "rim_slope_match_strength": 1.0,
            "rim_slope_match_center": [0.0, 0.0, 0.0],
            "rim_slope_match_normal": [0.0, 0.0, 1.0],
        },
    }
    return parse_geometry(data)


def test_rim_slope_match_out_constraints_allow_mismatched_counts():
    mesh = _build_mismatch_mesh()
    positions = mesh.positions_view()
    grads = constraint_gradients_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
    )
    assert grads is not None
    assert len(grads) > 0


def test_rim_slope_match_out_rows_match_dense_for_shape_constraints():
    mesh = _build_mismatch_mesh()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    dense = constraint_gradients_array(
        mesh, mesh.global_parameters, positions=positions, index_map=index_map
    )
    rows = constraint_gradients_rows_array(
        mesh, mesh.global_parameters, positions=positions, index_map=index_map
    )
    assert dense is not None
    assert rows is not None
    assert len(dense) == len(rows)

    for dense_g, (row_ids, row_vecs) in zip(dense, rows):
        rebuilt = np.zeros_like(positions)
        np.add.at(rebuilt, row_ids, row_vecs)
        assert np.allclose(rebuilt, dense_g, atol=1e-12, rtol=0.0)


def test_rim_slope_match_out_rows_match_dense_for_tilt_constraints():
    mesh = _build_mismatch_mesh()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    dense = constraint_gradients_tilt_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=index_map,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
    )
    rows = constraint_gradients_tilt_rows_array(
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


def test_rim_slope_match_out_energy_allows_mismatched_counts():
    mesh = _build_mismatch_mesh()
    positions = mesh.positions_view()
    param_resolver = ParameterResolver(mesh.global_parameters)

    tilts_out = np.zeros((len(mesh.vertex_ids), 3), dtype=float)
    for row, vid in enumerate(mesh.vertex_ids):
        opts = mesh.vertices[int(vid)].options
        if opts.get("rim_slope_match_group") == "rim":
            r = positions[row, :2]
            nrm = np.linalg.norm(r)
            if nrm > 1e-12:
                tilts_out[row, 0] = 0.1 * r[0] / nrm
                tilts_out[row, 1] = 0.1 * r[1] / nrm

    energy = compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=None,
        tilts_out=tilts_out,
    )
    assert energy > 0.0
