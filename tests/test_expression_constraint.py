import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Mesh, Vertex
from modules.constraints import expression as expr_constraint


def test_expression_constraint_moves_vertex_to_target():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(
        0,
        np.array([0.0, 0.0, 0.0]),
        options={
            "constraint_expression": "x",
            "constraint_target": 1.0,
        },
    )
    mesh.build_position_cache()

    gp = GlobalParameters({"expression_eps": 1e-6})
    expr_constraint.enforce_constraint(mesh, tol=1e-8, max_iter=10, global_params=gp)

    assert np.isclose(mesh.vertices[0].position[0], 1.0, atol=1e-5)
