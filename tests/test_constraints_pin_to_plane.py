import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from modules.constraints import pin_to_plane
from runtime.constraint_manager import ConstraintModuleManager


def test_pin_to_plane_projects_tagged_vertices_and_edge_endpoints() -> None:
    data = {
        "vertices": [
            [0.0, 0.0, 1.0, {"constraints": ["pin_to_plane"]}],  # should project
            [1.0, 0.0, 2.0],  # should project via edge constraint
            [0.0, 1.0, 3.0],  # should project via edge constraint
            [0.0, 2.0, 4.0],  # untouched
        ],
        "edges": [
            [1, 2, {"constraints": ["pin_to_plane"]}],
        ],
        "faces": [],
        "global_parameters": {
            "pin_to_plane_normal": [0.0, 0.0, 1.0],
            "pin_to_plane_point": [0.0, 0.0, 0.0],
        },
    }
    mesh = parse_geometry(data)
    manager = ConstraintModuleManager(mesh.constraint_modules)
    manager.enforce_all(mesh)
    assert np.isclose(mesh.vertices[0].position[2], 0.0)
    assert np.isclose(mesh.vertices[1].position[2], 0.0)
    assert np.isclose(mesh.vertices[2].position[2], 0.0)
    assert np.isclose(mesh.vertices[3].position[2], 4.0)


def test_pin_to_plane_constraint_gradients_match_finite_difference() -> None:
    data = {
        "vertices": [
            [0.3, -0.4, 1.2, {"constraints": ["pin_to_plane"]}],
            [1.0, 0.0, 0.2],
        ],
        "edges": [[0, 1]],
        "faces": [],
        "global_parameters": {
            "pin_to_plane_normal": [0.0, 0.0, 1.0],
            "pin_to_plane_point": [0.0, 0.0, 0.0],
        },
    }
    mesh = parse_geometry(data)
    grads = pin_to_plane.constraint_gradients(mesh, mesh.global_parameters)
    assert grads is not None
    g_plane = next(g for g in grads if 0 in g)[0]

    v = mesh.vertices[0].position.copy()
    n = np.array([0.0, 0.0, 1.0], dtype=float)
    p = np.array([0.0, 0.0, 0.0], dtype=float)

    def residual(x: np.ndarray) -> float:
        return float(np.dot(x - p, n))

    eps = 1e-7
    fd = np.zeros(3, dtype=float)
    for axis in range(3):
        xp = v.copy()
        xm = v.copy()
        xp[axis] += eps
        xm[axis] -= eps
        fd[axis] = (residual(xp) - residual(xm)) / (2.0 * eps)

    assert np.allclose(g_plane, fd, atol=1e-8, rtol=1e-6)


def test_pin_to_plane_constraint_gradients_array_matches_dict() -> None:
    data = {
        "vertices": [
            [0.3, -0.4, 1.2, {"constraints": ["pin_to_plane"]}],
            [1.2, 0.1, -0.5, {"constraints": ["pin_to_plane"]}],
            [2.0, 0.5, 0.0],
        ],
        "edges": [[0, 2, {"constraints": ["pin_to_plane"]}]],
        "faces": [],
        "global_parameters": {
            "pin_to_plane_normal": [0.0, 0.0, 1.0],
            "pin_to_plane_point": [0.0, 0.0, 0.0],
        },
    }
    mesh = parse_geometry(data)
    mesh.build_position_cache()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    g_dict = pin_to_plane.constraint_gradients(mesh, mesh.global_parameters)
    g_arr = pin_to_plane.constraint_gradients_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map=index_map,
    )

    assert g_dict is not None and g_arr is not None
    assert len(g_dict) == len(g_arr)

    for gC, gA in zip(g_dict, g_arr):
        dense = np.zeros_like(positions)
        for vidx, vec in gC.items():
            dense[index_map[int(vidx)]] += vec
        assert np.allclose(dense, gA, atol=1e-12, rtol=0.0)
