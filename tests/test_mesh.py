import numpy as np

from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh


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

    body = Body(0, [facet.index], options={"target_volume": 0})
    bodies = [body]

    mesh = Mesh()
    for v in vertices:
        mesh.vertices[v.index] = v
    for e in edges:
        mesh.edges[e.index] = e
    for f in facets:
        mesh.facets[f.index] = f
    for b in bodies:
        mesh.bodies[b.index] = b

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

def test_types_in_mesh():
    mesh = create_quad()

    # testing initial loading
    assert all(
        isinstance(v, Vertex) for v in mesh.vertices.values()
    ), "Not all vertices are Vertex instances"
    assert all(
        isinstance(e, Edge) for e in mesh.edges.values()
    ), "Not all edges are Edge instances"
    assert all(
        isinstance(f, Facet) for f in mesh.facets.values()
    ), "Not all facets are Facet instances"
    assert all(
        isinstance(b, Body) for b in mesh.bodies.values()
    ), "Not all bodies are Body instances"

    # testing after polygonal refinement
    mesh_poly = refine_polygonal_facets(mesh)
    assert all(
        isinstance(v, Vertex) for v in mesh_poly.vertices.values()
    ), "Vertices corrupted by polygonal refinement"
    assert all(
        isinstance(e, Edge) for e in mesh_poly.edges.values()
    ), "Edges corrupted by polygonal refinement"
    assert all(
        isinstance(f, Facet) for f in mesh_poly.facets.values()
    ), "Facets corrupted by polygonal refinement"
    assert all(
        isinstance(b, Body) for b in mesh_poly.bodies.values()
    ), "Bodies corrupted by polygonal refinement"

    # testing after triangular refinement
    mesh_tri = refine_triangle_mesh(mesh_poly)
    assert all(
        isinstance(v, Vertex) for v in mesh_tri.vertices.values()
    ), "Vertices corrupted by triangular refinement"
    assert all(
        isinstance(e, Edge) for e in mesh_tri.edges.values()
    ), "Edges corrupted by triangular refinement"
    assert all(
        isinstance(f, Facet) for f in mesh_tri.facets.values()
    ), "Facets corrupted by triangular refinement"
    assert all(
        isinstance(b, Body) for b in mesh_tri.bodies.values()
    ), "Bodies corrupted by triangular refinement"

def test_indices_match_keys():
    mesh = create_quad()

    # Every dictionary key must equal the object's .index
    for vid, v in mesh.vertices.items():
        assert vid == v.index, f"Vertex key {vid} != Vertex.index {v.index}"
    for eid, e in mesh.edges.items():
        assert eid == e.index, f"Edge key {eid} != Edge.index {e.index}"
    for fid, f in mesh.facets.items():
        assert fid == f.index, f"Facet key {fid} != Facet.index {f.index}"
    for bid, b in mesh.bodies.items():
        assert bid == b.index, f"Body key {bid} != Body.index {b.index}"

    mesh_poly = refine_polygonal_facets(mesh)
    for vid, v in mesh_poly.vertices.items():
        assert vid == v.index, f"Vertex key {vid} != Vertex.index {v.index}"
    for eid, e in mesh_poly.edges.items():
        assert eid == e.index, f"Edge key {eid} != Edge.index {e.index}"
    for fid, f in mesh_poly.facets.items():
        assert fid == f.index, f"Facet key {fid} != Facet.index {f.index}"
    for bid, b in mesh_poly.bodies.items():
        assert bid == b.index, f"Body key {bid} != Body.index {b.index}"

    mesh_tri = refine_triangle_mesh(mesh_poly)
    for vid, v in mesh_tri.vertices.items():
        assert vid == v.index, f"Vertex key {vid} != Vertex.index {v.index}"
    for eid, e in mesh_tri.edges.items():
        assert eid == e.index, f"Edge key {eid} != Edge.index {e.index}"
    for fid, f in mesh_tri.facets.items():
        assert fid == f.index, f"Facet key {fid} != Facet.index {f.index}"
    for bid, b in mesh_tri.bodies.items():
        assert bid == b.index, f"Body key {bid} != Body.index {b.index}"
