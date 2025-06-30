import numpy as np
from geometry.entities import Mesh, Vertex, Edge, Facet, Body
from parameters.global_parameters import GlobalParameters
from modules.energy.volume import compute_energy_and_gradient, calculate_volume_energy
from parameters.resolver import ParameterResolver

def test_volume_energy_and_gradient():
    # Vertices of a unit tetrahedron
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2 = Vertex(2, np.array([0.0, 1.0, 0.0]))
    v3 = Vertex(3, np.array([0.0, 0.0, 1.0]))
    vertices = {0: v0, 1: v1, 2: v2, 3: v3}

    # Edges (indices start at 1)
    e1 = Edge(1, 0, 1)
    e2 = Edge(2, 1, 2)
    e3 = Edge(3, 2, 0)
    e4 = Edge(4, 0, 3)
    e5 = Edge(5, 1, 3)
    e6 = Edge(6, 2, 3)
    edges = {1: e1, 2: e2, 3: e3, 4: e4, 5: e5, 6: e6}

    # Facets (triangle faces, using edge indices)
    # Face 0-1-2
    f0 = Facet(0, [1, 2, 3])
    # Face 0-1-3
    f1 = Facet(1, [1, 5, -4])
    # Face 0-2-3
    f2 = Facet(2, [3, 6, -4])
    # Face 1-2-3
    f3 = Facet(3, [2, 6, -5])
    facets = {0: f0, 1: f1, 2: f2, 3: f3}

    # Body (the tetrahedron)
    body = Body(0, [0, 1, 2, 3], options={"target_volume": 0.1,
                                          "volume_stiffness": 2.0},
                target_volume=0.1)
    bodies = {0: body}

    mesh = Mesh(vertices=vertices, edges=edges, facets=facets, bodies=bodies,
               energy_modules=["volume"])

    # Set up global parameters
    global_params = GlobalParameters()
    global_params.volume_stiffness = 2.0

    param_resolver = ParameterResolver(global_params)

    # Compute energy and gradient
    E, grad = compute_energy_and_gradient(mesh, global_params, param_resolver)

    # Analytical volume of unit tetrahedron: 1/6
    expected_volume = 1.0 / 6.0
    k = 2.0
    V0 = 0.1
    expected_energy = 0.5 * k * (expected_volume - V0) ** 2

    print(f"target volume: {mesh.bodies[0]}")


    print(f"body:\n{body}")
    assert np.isclose(E, expected_energy), f"Expected energy {expected_energy}, got {E}"

    # Test gradient: should be a dict with 4 entries, each a 3D vector
    assert isinstance(grad, dict)
    assert set(grad.keys()) == {0, 1, 2, 3}
    for g in grad.values():
        assert g.shape == (3,)
        assert np.all(np.isfinite(g))


def test_calculate_volume_energy():
    """Ensure the standalone energy calculator uses the correct volume method."""
    # Reuse the tetrahedron setup from the previous test
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2 = Vertex(2, np.array([0.0, 1.0, 0.0]))
    v3 = Vertex(3, np.array([0.0, 0.0, 1.0]))
    vertices = {0: v0, 1: v1, 2: v2, 3: v3}

    e1 = Edge(1, 0, 1)
    e2 = Edge(2, 1, 2)
    e3 = Edge(3, 2, 0)
    e4 = Edge(4, 0, 3)
    e5 = Edge(5, 1, 3)
    e6 = Edge(6, 2, 3)
    edges = {1: e1, 2: e2, 3: e3, 4: e4, 5: e5, 6: e6}

    f0 = Facet(0, [1, 2, 3])
    f1 = Facet(1, [1, 5, -4])
    f2 = Facet(2, [3, 6, -4])
    f3 = Facet(3, [2, 6, -5])
    facets = {0: f0, 1: f1, 2: f2, 3: f3}

    body = Body(0, [0, 1, 2, 3], options={"target_volume": 0.1,
                                          "volume_stiffness": 2.0},
                target_volume=0.1)
    bodies = {0: body}

    mesh = Mesh(vertices=vertices, edges=edges, facets=facets, bodies=bodies)

    global_params = GlobalParameters()
    global_params.volume_stiffness = 2.0

    energy = calculate_volume_energy(mesh, global_params)

    expected_volume = 1.0 / 6.0
    k = 2.0
    V0 = 0.1
    expected_energy = 0.5 * k * (expected_volume - V0) ** 2
    assert np.isclose(energy, expected_energy)

    # Test compute_volume_gradient directly
    grad_vol = body.compute_volume_gradient(mesh)
    assert isinstance(grad_vol, dict)
    assert set(grad_vol.keys()) == {0, 1, 2, 3}
    for g in grad_vol.values():
        assert g.shape == (3,)
        assert np.all(np.isfinite(g))
