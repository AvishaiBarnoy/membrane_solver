import math

import numpy as np

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import parse_geometry
from modules.constraints.rim_slope_match_out import constraint_gradients_array
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
