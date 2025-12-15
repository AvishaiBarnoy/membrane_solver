import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
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
