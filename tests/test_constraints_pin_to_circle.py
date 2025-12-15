import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager


def test_pin_to_circle_projects_vertices_and_edges() -> None:
    radius = 2.0
    data = {
        "vertices": [
            [3.0, 0.0, 5.0, {"constraints": ["pin_to_circle"]}],
            [0.0, 3.0, -1.0],  # via edge constraint
            [0.0, 0.0, 0.0],  # via edge constraint
            [0.0, 0.0, 0.0],  # untouched
        ],
        "edges": [
            [1, 2, {"constraints": ["pin_to_circle"]}],
        ],
        "faces": [],
        "global_parameters": {
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            "pin_to_circle_radius": radius,
        },
    }
    mesh = parse_geometry(data)
    manager = ConstraintModuleManager(mesh.constraint_modules)
    manager.enforce_all(mesh)

    center = np.array([0.0, 0.0, 0.0])
    for vidx in (0, 1, 2):
        pos = mesh.vertices[vidx].position
        assert np.isclose(pos[2], 0.0)
        assert np.isclose(np.linalg.norm(pos - center), radius)

    assert np.allclose(mesh.vertices[3].position, np.array([0.0, 0.0, 0.0]))
