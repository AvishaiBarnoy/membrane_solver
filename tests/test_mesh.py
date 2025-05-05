from geometry.entities import Vertex, Edge, Facet, Body, Mesh
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh
import numpy as np

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

def test_get_edge_directionality():
    # Create a sample mesh and edges
    mesh = Mesh()
    mesh.edges = {
        1: Edge(tail_index=0, head_index=1, index=1, refine=True),
        2: Edge(tail_index=1, head_index=2, index=2)
    }

    # Forward access
    edge_fwd = mesh.get_edge(1)
    assert edge_fwd.tail_index == 0
    assert edge_fwd.head_index == 1
    assert edge_fwd.refine is True

    # Reverse access
    edge_rev = mesh.get_edge(-1)
    assert edge_rev.tail_index == 1
    assert edge_rev.head_index == 0
    assert edge_rev.refine is True
    assert edge_rev.head_index == edge_fwd.tail_index
    assert edge_rev.tail_index == edge_fwd.head_index

    # Ensure it's not the same object (unless you're using proxy)
    #assert edge_fwd is not edge_rev

def test_types_in_mesh():
    mesh = create_quad()

    # testing initial loading
    assert all(type(v) == Vertex for v in mesh.vertices.values()), "Not all vertices are Vertex instances"
    assert all(type(e) == Edge for e in mesh.edges.values()), "Not all edges are Edge instances"
    assert all(type(f) == Facet for f in mesh.facets.values()), "Not all facets are Facet instances"
    assert all(type(b) == Body for b in mesh.bodies.values()), "Not all bodies are Body instances"

    # testing after polygonal refinement
    # refine polygonal
    # TODO: add this

    # testing after triangular refinement 
    # refine triangular
    # TODO: add this
