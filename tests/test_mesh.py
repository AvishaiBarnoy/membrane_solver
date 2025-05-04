from geometry.entities import Vertex, Edge, Facet, Body, Mesh
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh

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

