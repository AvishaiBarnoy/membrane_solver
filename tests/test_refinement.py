import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry.entities import Vertex, Edge, Facet, Body, Mesh
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh
from parameters.global_parameters import GlobalParameters

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

    facet = Facet(0, [e0.index, e1.index, e2.index, e3.index])
    facets = [facet]

    body = Body(0, [facets[0].index], options={"target_volume": 0})
    bodies = [body]

    mesh = Mesh()
    for i in vertices: mesh.vertices[i.index] = i
    for i in edges: mesh.edges[i.index] = i
    for i in facets: mesh.facets[i.index] = i
    for i in bodies: mesh.bodies[i.index] = i

    return mesh

def test_triangle_refinement_updates_bodies():
    mesh = create_quad()
    global_params = GlobalParameters({})

    # Testing polygonal refinement 
    mesh_tri = refine_polygonal_facets(mesh)
    assert len(mesh_tri.vertices) == len(mesh.vertices) + len(mesh.facets), "Initial triangulation of square should add a vertex at centroid, 5 total."
    assert len(mesh_tri.edges) == len(mesh.edges) * 2, "Initial triangulation of square should end with 8 edges."
    assert all(len(mesh_tri.facets[f_idx].edge_indices) == 3 for f_idx in mesh_tri.facets.keys()), "All refined facets must be triangles"
    assert len(mesh_tri.facets) == len(mesh.vertices), "Initial triangulation of square should end with 4 facets."
    #assert all(isinstance(f, Facet) for f in mesh.b_tri[0].facets), "All body facets must be Facets"
    assert len(mesh_tri.bodies[0].facet_indices) == len(mesh_tri.facets), "Body should include all refined facets"


    # Testing triangular refinement
    mesh_ref = refine_triangle_mesh(mesh_tri)

    assert len(mesh_ref.vertices) == len(mesh_tri.vertices) + len(mesh_tri.edges), "Refinemenet should add len(edges) new vertex per facet"
    assert len(mesh_ref.edges) == 2 * len(mesh_tri.edges) + 3 * len(mesh_tri.facets), "Refining splits edges and adds 3 more for each facet"
    assert len(mesh_ref.facets) == 2**len(mesh_tri.facets), "Refiningt increases number of facets by factor of 2^k"
    assert all(len(mesh_ref.facets[f_idx].edge_indices) == 3 for f_idx in mesh_ref.facets.keys()), "All refined facets must be triangles"
    assert len(mesh_ref.bodies[0].facet_indices) == len(mesh_ref.facets), "Body should include all refined facets"

def test_child_facets_are_closed_loops():
    mesh = create_quad()

    # 1. check loop on initial triangulation 
    mesh2 = refine_polygonal_facets(mesh)
    for facet_idx in mesh2.facets.keys():
        # grab the three edges in order
        # check chaining: edge.head == next_edge.tail (mod 3)
        facet = mesh2.facets[facet_idx]

        for i in range(3):
            e_curr = mesh2.get_edge(facet.edge_indices[i])
            e_next = mesh2.get_edge(facet.edge_indices[(i + 1) % 3])
            assert e_curr.head_index == e_next.tail_index, (
                f"Facet {facet.index} is not a closed loop: "
                f"edge {e_curr.index}.head={e_curr.head_index!r} ≠ "
                f"edge {e_next.index}.tail={e_next.tail_index!r}"
            )


    # 2. check loop in runtime triangulation
    mesh3 = refine_polygonal_facets(mesh2)

    for facet_idx in mesh3.facets.keys():
        # grab the three edges in order
        # check chaining: edge.head == next_edge.tail (mod 3)
        facet = mesh2.facets[facet_idx]
        for i in range(3):
            e_curr = mesh3.get_edge(facet.edge_indices[i])
            e_next = mesh3.get_edge(facet.edge_indices[(i + 1) % 3])
            assert e_curr.head_index == e_next.tail_index, (
                f"Facet {facet.index} is not a closed loop: "
                f"edge {e_curr.index}.head={e_curr.head_index!r} ≠ "
                f"edge {e_next.index}.tail={e_next.tail_index!r}"
            )
