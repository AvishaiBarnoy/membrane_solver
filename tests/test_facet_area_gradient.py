from geometry.entities import Mesh, Vertex, Edge, Facet, Body
import numpy as np

def test_compute_area_gradient():
    # Create a simple triangular facet
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2 = Vertex(2, np.array([0.0, 1.0, 0.0]))
    vertices = {0: v0, 1: v1, 2: v2}

    e0 = Edge(1, 0, 1)
    e1 = Edge(2, 1, 2)
    e2 = Edge(3, 2, 0)
    edges = {1: e0, 2: e1, 3: e2}

    facet = Facet(0, [1, 2, 3])

    mesh = Mesh(vertices=vertices, edges=edges, facets={0: facet})

    # Compute the area gradient
    grad = facet.compute_area_gradient(mesh)

    # Assert the gradient values
    assert len(grad) == 3
    for g in grad.values():
        assert g.shape == (3,)
