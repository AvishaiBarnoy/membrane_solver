import numpy as np
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry.geometry_entities import Vertex, Edge, Facet, Body
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh
from parameters.global_parameters import GlobalParameters

def get_triangle_normal(a, b, c):
    n = np.cross(b - a, c - a)
    return n / np.linalg.norm(n)

def test_square_refinement_preserves_normals():
    v0 = Vertex([0, 0, 0], 0)
    v1 = Vertex([1, 0, 0], 1)
    v2 = Vertex([1, 1, 0], 2)
    v3 = Vertex([0, 1, 0], 3)
    vertices = [v0, v1, v2, v3]

    e0 = Edge(v0, v1, 0)
    e1 = Edge(v1, v2, 1)
    e2 = Edge(v2, v3, 2)
    e3 = Edge(v3, v0, 3)
    edges = [e0, e1, e2, e3]

    facet = Facet([e0, e1, e2, e3], index=0)
    body = Body([facet], index=0)
    facets = [facet]
    bodies = [body]

    global_params = GlobalParameters({})

    # Normal of parent
    a, b, c = v0.position, v1.position, v2.position
    parent_normal = get_triangle_normal(a, b, c)

    v_ref, e_ref, f_ref, _ = refine_polygonal_facets(vertices, edges, facets, bodies, global_params)

    for i, f in enumerate(f_ref):
        a = f.edges[0].tail.position
        b = f.edges[0].head.position
        c = f.edges[1].head.position
        n = get_triangle_normal(a, b, c)
        dot = np.dot(n, parent_normal)
        assert dot > 0.99, f"Refined facet {i} normal is flipped (dot={dot:.3f})"

def test_triangle_refinement_preserves_normals():
    v0 = Vertex([0, 0, 0], 0)
    v1 = Vertex([1, 0, 0], 1)
    v2 = Vertex([0.5, 1, 0], 2)
    vertices = [v0, v1, v2]

    e0 = Edge(v0, v1, 0)
    e1 = Edge(v1, v2, 1)
    e2 = Edge(v2, v0, 2)
    edges = [e0, e1, e2]

    facet = Facet([e0, e1, e2], index=0)
    body = Body([facet], index=0)
    facets = [facet]
    bodies = [body]

    global_params = GlobalParameters({})

    # Normal of parent
    parent_normal = get_triangle_normal(v0.position, v1.position, v2.position)

    v_ref, e_ref, f_ref, _ = refine_triangle_mesh(vertices, edges, facets, bodies)

    for i, f in enumerate(f_ref):
        a = f.edges[0].tail.position
        b = f.edges[0].head.position
        c = f.edges[1].head.position
        n = get_triangle_normal(a, b, c)
        dot = np.dot(n, parent_normal)
        assert dot > 0.99, f"Refined triangle {i} normal flipped (dot={dot:.3f})"

