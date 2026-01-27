import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from modules.constraints import pin_to_plane


def test_pin_to_plane_slide_projects_to_group_centroid_plane():
    data = {
        "global_parameters": {
            "pin_to_plane_normal": [0.0, 0.0, 1.0],
        },
        "constraint_modules": ["pin_to_plane"],
        "energy_modules": [],
        "vertices": [
            [
                0.0,
                0.0,
                1.0,
                {
                    "constraints": ["pin_to_plane"],
                    "pin_to_plane_mode": "slide",
                    "pin_to_plane_group": "disk",
                },
            ],
            [
                1.0,
                0.0,
                3.0,
                {
                    "constraints": ["pin_to_plane"],
                    "pin_to_plane_mode": "slide",
                    "pin_to_plane_group": "disk",
                },
            ],
            [0.0, 1.0, -2.0],
        ],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "instructions": [],
    }
    mesh = parse_geometry(data)

    pin_to_plane.enforce_constraint(mesh)

    z0 = mesh.vertices[0].position[2]
    z1 = mesh.vertices[1].position[2]
    z2 = mesh.vertices[2].position[2]

    assert np.isclose(z0, 2.0)
    assert np.isclose(z1, 2.0)
    assert np.isclose(z2, -2.0)
