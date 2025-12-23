import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Mesh, Vertex
from modules.energy import expression as expr_energy
from parameters.global_parameters import GlobalParameters


def test_expression_energy_vertex():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([1.0, 2.0, 3.0]), options={"expression": "x + y + z"})
    mesh.build_position_cache()

    gp = GlobalParameters({"expression_eps": 1e-6})
    E, grad = expr_energy.compute_energy_and_gradient(mesh, gp, None)

    assert np.isclose(E, 6.0)
    assert 0 in grad
    assert np.allclose(grad[0], np.array([1.0, 1.0, 1.0]), rtol=1e-4, atol=1e-4)


def test_expression_energy_edge_length():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([2.0, 0.0, 0.0]))
    mesh.edges[1] = Edge(1, 0, 1, options={"expression": "x"})
    mesh.build_position_cache()

    gp = GlobalParameters({"expression_eps": 1e-6})
    E, _ = expr_energy.compute_energy_and_gradient(mesh, gp, None, compute_gradient=False)

    assert np.isclose(E, 2.0)
