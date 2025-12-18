import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.equiangulation import (
    equiangulate_mesh,
    find_connecting_edge,
    flip_edge_safe,
    get_off_vertex,
    get_oriented_edge,
    should_flip_edge,
)


def build_square_two_triangles():
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([1.0, 1.0, 0.0])),
        3: Vertex(3, np.array([0.0, 1.0, 0.0])),
    }
    # Boundary edges + diagonal (0->2)
    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 0, 2),
        4: Edge(4, 2, 3),
        5: Edge(5, 3, 0),
    }
    # Two triangles sharing diagonal edge 3: (0,1,2) and (0,2,3)
    mesh.facets = {
        0: Facet(0, [1, 2, -3]),
        1: Facet(1, [3, 4, 5]),
    }
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def test_get_off_vertex_returns_none_for_non_triangle(caplog):
    mesh = build_square_two_triangles()
    edge = mesh.edges[3]
    bad = Facet(99, [1, 2, 3, 4])
    with caplog.at_level("WARNING"):
        assert get_off_vertex(mesh, bad, edge) is None
    assert "not a triangle" in caplog.text


def test_find_connecting_edge_and_oriented_edge():
    mesh = build_square_two_triangles()

    # In facet 0, edge 2 connects vertices (1,2)
    found = find_connecting_edge(mesh, 1, 2, candidate_edges=[1, 2, -3])
    assert found == 2

    # Orientation: edge 2 is 1->2, so 2 is forward and -2 is reverse
    assert get_oriented_edge(mesh, 1, 2, 2) == 2
    assert get_oriented_edge(mesh, 2, 1, 2) == -2


def test_flip_edge_safe_successful():
    mesh = build_square_two_triangles()
    f0 = mesh.facets[0]
    f1 = mesh.facets[1]

    ok = flip_edge_safe(mesh, edge_idx=3, facet1=f0, facet2=f1, new_edge_idx=6)
    assert ok is True
    assert 3 not in mesh.edges
    assert 6 in mesh.edges
    assert set(abs(e) for e in f0.edge_indices) == {1, 5, 6}
    assert set(abs(e) for e in f1.edge_indices) == {2, 4, 6}


def test_should_flip_edge_returns_false_for_degenerate_off_angle():
    mesh = build_square_two_triangles()
    # Collapse one off vertex onto the shared edge to trigger b1*c1 == 0 path.
    mesh.vertices[1].position = mesh.vertices[0].position.copy()
    edge = mesh.edges[3]
    f0 = mesh.facets[0]
    f1 = mesh.facets[1]
    assert should_flip_edge(mesh, edge, f0, f1) is False


def test_equiangulate_mesh_returns_original_on_invalid_mesh(caplog):
    mesh = build_square_two_triangles()
    # Break connectivity so initial validation fails.
    mesh.facets[0].edge_indices = [999, 2, -3]
    with caplog.at_level("WARNING"):
        out = equiangulate_mesh(mesh, max_iterations=1)
    assert out is mesh
    assert "Skipping equiangulation" in caplog.text
