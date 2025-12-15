import os
import sys

import numpy as np

# Adjust import paths for this testing environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.refinement import refine_triangle_mesh


def test_build_connectivity_maps():
    mesh = Mesh()

    # Define 3 vertices forming a triangle
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([0.0, 1.0, 0.0]))
    }

    # Define 3 edges: 0->1, 1->2, 2->0
    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 0)
    }

    # Define one facet using these 3 edges (triangle)
    mesh.facets = {
        0: Facet(0, edge_indices=[1, 2, 3])
    }

    # Build connectivity
    mesh.build_connectivity_maps()

    # Assertions
    assert mesh.vertex_to_edges[0] == {1, 3}
    assert mesh.vertex_to_edges[1] == {1, 2}
    assert mesh.vertex_to_edges[2] == {2, 3}

    assert mesh.edge_to_facets[1] == {0}
    assert mesh.edge_to_facets[2] == {0}
    assert mesh.edge_to_facets[3] == {0}

    assert mesh.vertex_to_facets[0] == {0}
    assert mesh.vertex_to_facets[1] == {0}
    assert mesh.vertex_to_facets[2] == {0}

def create_simple_triangle_mesh():
    mesh = Mesh()

    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([0.0, 1.0, 0.0]))
    }

    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 0)
    }

    mesh.facets = {
        0: Facet(0, edge_indices=[1, 2, 3])
    }

    return mesh

def test_connectivity_after_refinement():
    mesh = create_simple_triangle_mesh()
    mesh.build_connectivity_maps()

    # Check pre-refinement connectivity
    assert mesh.vertex_to_facets[0] == {0}
    assert mesh.vertex_to_edges[0] == {1, 3}
    assert mesh.edge_to_facets[1] == {0}

    # Apply refinement
    mesh = refine_triangle_mesh(mesh)

    # Should be updated automatically inside refine_mesh
    assert len(mesh.facets) > 1  # refinement added facets
    assert any(len(f.edge_indices) == 3 for f in mesh.facets.values())  # still triangles

    # Verify connectivity maps still valid
    for v_id, connected_edges in mesh.vertex_to_edges.items():
        for e_id in connected_edges:
            edge = mesh.edges[e_id]
            assert v_id == edge.tail_index or v_id == edge.head_index

    for e_id, connected_facets in mesh.edge_to_facets.items():
        for f_id in connected_facets:
            assert e_id in [abs(ei) for ei in mesh.facets[f_id].edge_indices]

    for v_id, connected_facets in mesh.vertex_to_facets.items():
        for f_id in connected_facets:
            edge_indices = mesh.facets[f_id].edge_indices
            edge_vertices = set()
            for ei in edge_indices:
                edge = mesh.get_edge(ei)
                edge_vertices.update([edge.tail_index, edge.head_index])
            assert v_id in edge_vertices
