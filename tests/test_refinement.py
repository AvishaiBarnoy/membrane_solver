import numpy as np
from geometry.geometry_entities import Vertex, Edge, Facet
from runtime.geometry_refinement import refine_geometry
from parameters.global_parameters import GlobalParameters

def make_quad_facet():
    v0 = Vertex([0, 0, 0], 0)
    v1 = Vertex([1, 0, 0], 1)
    v2 = Vertex([1, 1, 0], 2)
    v3 = Vertex([0, 1, 0], 3)
    vertices = [v0, v1, v2, v3]

    e0 = Edge(v0, v1, 0)
    e1 = Edge(v1, v2, 1)
    e2 = Edge(v2, v3, 2)
    e3 = Edge(v3, v0, 3)

    facet = Facet([e0, e1, e2, e3], 0)
    return vertices, [facet]

def test_refine_quad_to_triangles():
    vertices, facets = make_quad_facet()
    global_params = GlobalParameters({})

    v_new, f_new = refine_geometry(vertices, facets, global_params)

    # Original quad should become 4 vertices + 1 centroid = 5
    assert len(v_new) == 5

    # Should result in 4 triangle facets
    assert len(f_new) == 4

    # All new facets should have 3 edges
    for f in f_new:
        assert len(f.edges) == 3

    # Normals should point in same direction (test with cross product)
    normals = []
    for f in f_new:
        a = f.edges[0].tail.position
        b = f.edges[0].head.position
        c = f.edges[1].head.position
        n = np.cross(b - a, c - a)
        normals.append(n / np.linalg.norm(n))

    first = normals[0]
    for n in normals[1:]:
        assert np.dot(first, n) > 0.99  # nearly same direction

