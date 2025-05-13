from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.minimizer import Minimizer
from modules.steppers.gradient_descent import GradientDescent
from geometry.entities import Mesh, Vertex, Edge, Facet, Body
from parameters.global_parameters import GlobalParameters
from runtime.refinement import refine_polygonal_facets
from runtime.energy_manager import EnergyModuleManager
import numpy as np
import pytest

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
    mock_global_params = GlobalParameters()
    mock_mesh.energy_modules = ["surface", "volume"]

    # Mock energy manager
    mock_energy_manager = MagicMock()
    #mock_energy_manager.get_energy_function.return_value = lambda obj, params: (0.0, {})

    # Mock energy functions
    mock_surface_energy_function = lambda obj, params: (1.0, {0: np.array([0.1, 0.1, 0.1])})
    mock_volume_energy_function = lambda obj, params: (2.0, {1: np.array([0.2, 0.2, 0.2])})

    # Mock get_module to return the mocked energy functions
    mock_energy_manager.get_module.side_effect = lambda mod: (
        mock_surface_energy_function if mod == "surface" else mock_volume_energy_function
    )

    # Mock stepper
    mock_stepper = GradientDescent()

    # Initialize minimizer
    minimizer = Minimizer(mock_mesh, mock_global_params, mock_stepper, mock_energy_manager)

    # Run minimization
    result = minimizer.minimize()

    # Assertions
    assert result is not None, "Minimizer should return a result"
    assert mock_energy_manager.get_module.call_count > 0, (
        "Expected 'get_module' to have been called"
    )

def test_get_module():
    # Mock module names
    module_names = ["line_tension1", "line_tension2", "edge1"]

    # Initialize the EnergyModuleManager
    energy_manager = EnergyModuleManager(module_names)

    print(energy_manager.modules)
    # Test retrieving loaded modules
    for module_name in module_names:
        module = energy_manager.get_module(module_name)
        print(module)
        assert module is not None, f"Module '{module_name}' should be loaded"
        assert hasattr(module, "compute_energy_and_gradient"), (
            f"Module '{module_name}' should have a 'compute_energy_and_gradient' function"
        )

    # Test retrieving a non-existent module
    with pytest.raises(KeyError, match="Energy module 'non_existent' not found."):
        energy_manager.get_module("non_existent")

def test_get_energy_function():
    # Mock module names
    module_names = ["surface", "volume"]

    # Mock modules with different energy functions
    mock_surface_module = MagicMock()
    mock_surface_module.calculate_energy = MagicMock(return_value="surface_energy")
    mock_surface_module.compute_energy_and_gradient = MagicMock(return_value="surface_gradient")

    mock_volume_module = MagicMock()
    mock_volume_module.calculate_energy = MagicMock(return_value="volume_energy")

    # Initialize the EnergyModuleManager
    energy_manager = EnergyModuleManager(module_names)
    energy_manager.modules = {
        "surface": mock_surface_module,
        "volume": mock_volume_module
    }

    # Test retrieving generic energy function
    energy_function = energy_manager.get_energy_function("surface", "facet")
    assert energy_function() == "surface_energy", "Expected 'calculate_energy' to be used for 'surface'"

    # Test retrieving type-specific energy function
    energy_function = energy_manager.get_energy_function("volume", "body")
    assert energy_function() == "volume_energy", "Expected 'calculate_volume_energy' to be used for 'volume'"

    # Test fallback to 'compute_energy_and_gradient'
    #energy_function = energy_manager.get_energy_function("surface", "facet")
    #assert energy_function() == "surface_gradient", "Expected 'compute_energy_and_gradient' to be used for 'surface'"

    # Test missing energy function
    #with pytest.raises(AttributeError, match="Module 'volume' must define either 'calculate_energy' or 'calculate_volume_energy' for type 'volume'"):
    #    energy_manager.get_energy_function("volume", "facet")
