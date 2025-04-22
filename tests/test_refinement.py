import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry.geometry_entities import Vertex, Edge, Facet, Body
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh
from parameters.global_parameters import GlobalParameters

def create_quad():
    # A unit square in the XY plane
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

    facet = Facet([e0, e1, e2, e3], 0)
    facets = [facet]

    body = Body(facets, index=0, options={"target_volume": 0})
    return vertices, edges, facets, [body]

def test_triangle_refinement_updates_bodies():
    vertices, edges, facets, bodies = create_quad()
    global_params = GlobalParameters({})

    # Testing initial refinement 
    v_tri, e_tri, f_tri, b_tri = refine_polygonal_facets(vertices, edges, facets, bodies, global_params)
    assert len(v_tri) == 5, "Initial triangulation of square should add a vertex at centroid, 5 total."
    unique_edges = []
    for e in e_tri:
        temp = list(e.tail.position)+list(e.head.position)
        #print(temp)
        if temp not in unique_edges:
            unique_edges.append(temp)
    for u in unique_edges: print(u)
    print(len(unique_edges))

    assert len(e_tri) == 8, "Initial triangulation of square should end with 8 edges."
    assert all(len(f.edges) == 3 for f in f_tri), "All refined facets must be triangles"
    assert len(f_tri) == 4, "Initial triangulation of square should end with 4 facets."
    assert all(isinstance(f, Facet) for f in b_ref[0].facets), "All body facets must be Facets"
    assert len(b_ref[0].facets) == len(f_ref), "Body should include all refined facets"

    # Testing refinement
    v_ref, e_ref, f_ref, b_ref = refine_triangle_mesh(v_tri, e_tri, f_tri, b_tri)
    # result = refine_triangle_mesh(v_tri, e_tri, f_tri, b_tri)
    # print(result)  # <-- temporarily print the result structure

    print(f"# of facets in f_ref: {len(f_ref)}")
    print(f"# of facets in body: {len(b_ref[0].facets)}")
    print("Facet indices in body:", [f.index for f in b_ref[0].facets])

    assert all(len(f.edges) == 3 for f in f_ref), "All refined facets must be triangles"
    assert all(isinstance(f, Facet) for f in b_ref[0].facets), "All body facets must be Facets"
    assert len(b_ref[0].facets) == len(f_ref), "Body should include all refined facets"

