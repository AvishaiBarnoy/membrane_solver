import pytest
import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry.entities import Mesh, Vertex, Edge, Facet
from parameters.global_parameters import GlobalParameters
from modules.energy.surface import compute_energy_and_gradient, calculate_surface_energy
from parameters.resolver import ParameterResolver

def test_compute_energy_and_gradient():
    # Create a simple triangular mesh
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2 = Vertex(2, np.array([0.0, 1.0, 0.0]))
    vertices = {0: v0, 1: v1, 2: v2}

    e0 = Edge(1, 0, 1)
    e1 = Edge(2, 1, 2)
    e2 = Edge(3, 2, 0)
    edges = {1: e0, 2: e1, 3: e2}

    facet = Facet(0, [1, 2, 3], options={"surface_tension": 2.0})
    facets = {0: facet}

    mesh = Mesh(vertices=vertices, edges=edges, facets=facets)

    # Define global parameters
    global_params = GlobalParameters()
    global_params.surface_tension = 1.0  # Default surface tension

    # Create a parameter resolver
    param_resolver = ParameterResolver(global_params)

    # Compute energy and gradient
    E, grad = compute_energy_and_gradient(mesh, global_params, param_resolver)

    # Expected energy: γ * area
    # Area of the triangle = 0.5 * |v1 - v0 x v2 - v0|
    v0_pos, v1_pos, v2_pos = v0.position, v1.position, v2.position
    expected_area = 0.5 * np.linalg.norm(np.cross(v1_pos - v0_pos, v2_pos - v0_pos))
    expected_energy = 2.0 * expected_area  # γ = 2.0 from facet options

    # Assert energy
    assert np.isclose(E, expected_energy), f"Expected energy {expected_energy}, got {E}"

    # Assert gradient
    assert len(grad) == 3, "Gradient should have entries for all 3 vertices"
    for vertex_index, gradient_vector in grad.items():
        assert gradient_vector.shape == (3,), f"Gradient for vertex {vertex_index} should be a 3D vector"
        # Gradient values are harder to assert directly, but we can ensure they are finite
        assert np.all(np.isfinite(gradient_vector)), f"Gradient for vertex {vertex_index} contains invalid values"

def test_surface_energy_known_values():
    #from energy_modules.surface import SurfaceEnergy
    from modules.energy.surface import compute_energy_and_gradient

    # Define a single triangle of area = 0.5 (right triangle with legs 1)
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2 = Vertex(2, np.array([0.0, 1.0, 0.0]))
    vertices = {0: v0, 1: v1, 2: v2}

    e0 = Edge(1, 0, 1)
    e1 = Edge(2, 1, 2)
    e2 = Edge(3, 2, 0)
    edges = {1: e0, 2: e1, 3: e2}

    facet = Facet(0, [1, 2, 3], options={"surface_tension": 2.0})
    facets = {0: facet}

    mesh = Mesh(vertices=vertices, edges=edges, facets=facets)
    global_params = GlobalParameters()
    global_params.surface_tension = 1.0  # Default surface tension
    param_resolver = ParameterResolver(global_params)

    # Uniform surface tension of 2.0
    for f in mesh.facets.values():
        f.options['surface_tension'] = 2.0

    energy, _ = compute_energy_and_gradient(mesh, global_params, param_resolver)

    expected_energy = 0.5 * 2.0  # area * tension
    assert abs(energy - expected_energy) < 1e-12, f"Expected {expected_energy}, got {energy}"


def test_calculate_surface_energy():
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2 = Vertex(2, np.array([0.0, 1.0, 0.0]))
    vertices = {0: v0, 1: v1, 2: v2}

    e0 = Edge(1, 0, 1)
    e1 = Edge(2, 1, 2)
    e2 = Edge(3, 2, 0)
    edges = {1: e0, 2: e1, 3: e2}

    facet = Facet(0, [1, 2, 3], options={"surface_tension": 2.0})
    facets = {0: facet}

    mesh = Mesh(vertices=vertices, edges=edges, facets=facets)
    global_params = GlobalParameters()

    energy = calculate_surface_energy(mesh, global_params)
    expected_energy = 0.5 * 2.0
    assert np.isclose(energy, expected_energy)
