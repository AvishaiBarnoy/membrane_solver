import numpy as np

from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from modules.constraints.volume import enforce_constraint


def create_tetrahedron(target_volume: float) -> Mesh:
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

    body = Body(
        0,
        [0, 1, 2, 3],
        target_volume=target_volume,
        options={"target_volume": target_volume},
    )
    bodies = {0: body}

    mesh = Mesh(vertices=vertices, edges=edges, facets=facets, bodies=bodies)
    return mesh


def test_enforce_volume_constraint():
    target = 0.2
    mesh = create_tetrahedron(target)
    body = mesh.bodies[0]

    initial_volume = body.compute_volume(mesh)
    assert not np.isclose(initial_volume, target)

    enforce_constraint(mesh)

    new_volume = body.compute_volume(mesh)
    assert np.isclose(new_volume, target)
