from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.minimizer import Minimizer
from modules.steppers.gradient_descent import GradientDescent
from geometry.entities import Mesh, Vertex, Edge, Facet, Body
from parameters.global_parameters import GlobalParameters
from runtime.refinement import refine_polygonal_facets
import numpy as np

def create_quad():

    mesh = Mesh()

    # A unit square in the XY plane
    v0 = Vertex(0, np.array([0, 0, 0]))
    v1 = Vertex(1, np.array([1 , 0, 0]))
    v2 = Vertex(2, np.array([1 , 1, 0]))
    v3 = Vertex(3, np.array([0, 1, 0]))
    vertices = [v0, v1, v2, v3]

    e0 = Edge(1, v0.index, v1.index)
    e1 = Edge(2, v1.index, v2.index)
    e2 = Edge(3, v2.index, v3.index)
    e3 = Edge(4, v3.index, v0.index)
    edges = [e0, e1, e2, e3]

    facet = Facet(0, [e0.index, e1.index, e2.index, e3.index],
                  options={"energy": "surface"})
    facets = [facet]

    body = Body(0, [facets[0].index], options={"target_volume": 1.0,
                                               "energy": "volume"})
    bodies = [body]

    mesh = Mesh()
    for i in vertices: mesh.vertices[i.index] = i
    for i in edges: mesh.edges[i.index] = i
    for i in facets: mesh.facets[i.index] = i
    for i in bodies: mesh.bodies[i.index] = i

    return mesh

def test_minimizer_with_mock_energy_manager():
    # Mock mesh and global parameters
    mock_mesh = create_quad()
    mock_mesh = refine_polygonal_facets(mock_mesh)
    print("Facets after refinement:", mock_mesh.facets)

    mock_global_params = GlobalParameters()
    mock_mesh.energy_modules = ["surface", "volume"]

    # Mock energy manager
    mock_energy_manager = MagicMock()
    mock_energy_manager.get_energy_function.return_value = lambda obj, params: (0.0, {})

    # Mock stepper
    mock_stepper = GradientDescent()

    # Initialize minimizer
    minimizer = Minimizer(mock_mesh, mock_global_params, mock_stepper, mock_energy_manager)

    # Run minimization
    result = minimizer.minimize()

    # Assertions
    assert result is not None, "Minimizer should return a result"
    assert mock_energy_manager.get_energy_function.call_count > 0, (
        "Expected 'get_energy_function' to have been called"
    )
