import numpy as np
import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry.entities import Vertex, Edge, Facet, Body, Mesh
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh

def get_triangle_normal(a, b, c):
    n = np.cross(b - a, c - a)
    return n / np.linalg.norm(n)

def test_square_refinement_preserves_normals():
    v0 = Vertex(0, np.array([0, 0, 0]))
    v1 = Vertex(1, np.array([1, 0, 0]))
    v2 = Vertex(2, np.array([1, 1, 0]))
    v3 = Vertex(3, np.array([0, 1, 0]))
    vertices = [v0, v1, v2, v3]

    e0 = Edge(1, v0.index, v1.index)
    e1 = Edge(2, v1.index, v2.index)
    e2 = Edge(3, v2.index, v3.index)
    e3 = Edge(4, v3.index, v0.index)
    edges = [e0, e1, e2, e3]

    facet = Facet(0, [e0.index, e1.index, e2.index, e3.index])
    facets = [facet]
    body = Body(0, [f.index for f in facets])
    bodies = [body]


    mesh = Mesh()
    for i in vertices: mesh.vertices[i.index] = i
    for i in edges: mesh.edges[i.index] = i
    for i in facets: mesh.facets[i.index] = i
    for i in bodies: mesh.bodies[i.index] = i
    # Normal of parent
    a, b, c = v0.position, v1.position, v2.position
    parent_normal = get_triangle_normal(a, b, c)
    mesh_tri = refine_polygonal_facets(mesh)
    for f_idx in mesh_tri.facets.keys():
        facet = mesh_tri.facets[f_idx]
        a = mesh_tri.vertices[mesh_tri.get_edge(facet.edge_indices[0]).tail_index].position
        b = mesh_tri.vertices[mesh_tri.get_edge(facet.edge_indices[0]).head_index].position
        c = mesh_tri.vertices[mesh_tri.get_edge(facet.edge_indices[1]).head_index].position
        n = get_triangle_normal(a, b, c)
        dot = np.dot(n, parent_normal)
        assert dot > 0.99, f"Refined facet {mesh_tri.facets[f_idx]} normal is flipped (dot={dot:.3f})"

    mesh_ref = refine_triangle_mesh(mesh_tri)
    for f_idx in mesh_ref.facets.keys():
        facet = mesh_ref.facets[f_idx]
        a = mesh_ref.vertices[mesh_ref.get_edge(facet.edge_indices[0]).tail_index].position
        b = mesh_ref.vertices[mesh_ref.get_edge(facet.edge_indices[0]).head_index].position
        c = mesh_ref.vertices[mesh_ref.get_edge(facet.edge_indices[1]).head_index].position
        n = get_triangle_normal(a, b, c)
        dot = np.dot(n, parent_normal)
        assert dot > 0.99, f"Refined facet {mesh_ref.facets[f_idx]} normal is flipped (dot={dot:.3f})"

def test_triangle_refinement_preserves_normals():
    v0 = Vertex(0, np.array([0, 0, 0]))
    v1 = Vertex(1, np.array([1, 0, 0]))
    v2 = Vertex(2, np.array([0.5, 1, 0]))
    vertices = [v0, v1, v2]

    e0 = Edge(1, v0.index, v1.index)
    e1 = Edge(2, v1.index, v2.index)
    e2 = Edge(3, v2.index, v0.index)
    edges = [e0, e1, e2]

    facet = Facet(0, [e0.index, e1.index, e2.index])
    facets = [facet]
    body = Body(0, [f.index for f in facets])
    bodies = [body]


    mesh = Mesh()
    for i in vertices: mesh.vertices[i.index] = i
    for i in edges: mesh.edges[i.index] = i
    for i in facets: mesh.facets[i.index] = i
    for i in bodies: mesh.bodies[i.index] = i

    # Normal of parent
    parent_normal = get_triangle_normal(v0.position, v1.position, v2.position)

    mesh_tri = refine_polygonal_facets(mesh)
    assert len(mesh_tri.facets) == len(facets), "refine polygonal should not affect triangle facets"

    mesh_ref = refine_triangle_mesh(mesh_tri)
    for f_idx in mesh_ref.facets.keys():
        facet = mesh_ref.facets[f_idx]
        a = mesh_ref.vertices[mesh_ref.get_edge(facet.edge_indices[0]).tail_index].position
        b = mesh_ref.vertices[mesh_ref.get_edge(facet.edge_indices[0]).head_index].position
        c = mesh_ref.vertices[mesh_ref.get_edge(facet.edge_indices[1]).head_index].position

        n = get_triangle_normal(a, b, c)
        dot = np.dot(n, parent_normal)
        assert dot > 0.99, f"Refined facet {mesh_ref.facets[f_idx]} normal is flipped (dot={dot:.3f})"

def test_facet_normal_reversal():
    # Create triangle vertices
    v0 = Vertex(index=0, position=np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(index=1, position=np.array([1.0, 0.0, 0.0]))
    v2 = Vertex(index=2, position=np.array([0.0, 1.0, 0.0]))
    vertices = {0: v0, 1: v1, 2: v2}

    # Create edges and reversed edges
    e0 = Edge(index=1, tail_index=0, head_index=1)
    e1 = Edge(index=2, tail_index=1, head_index=2)
    e2 = Edge(index=3, tail_index=2, head_index=0)
    edges = {1: e0, 2: e1, 3: e2}

    mesh = Mesh()
    mesh.vertices = vertices
    mesh.edges = edges

    facet = Facet(index=0, edge_indices=[1, 2, 3])
    reversed_facet = Facet(index=1, edge_indices=[-3, -2, -1])

    n1 = facet.normal(mesh)
    n2 = reversed_facet.normal(mesh)

    assert np.allclose(n1, -n2), "Reversed facet should flip normal"

