import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex


def _build_single_triangle_mesh() -> Mesh:
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([0.0, 1.0, 0.0])),
    }
    mesh.edges = {1: Edge(1, 0, 1), 2: Edge(2, 1, 2), 3: Edge(3, 2, 0)}
    mesh.facets = {0: Facet(0, edge_indices=[1, 2, 3])}
    return mesh


def test_build_connectivity_maps_is_cached_without_topology_change():
    mesh = _build_single_triangle_mesh()
    mesh.increment_topology_version()

    mesh.build_connectivity_maps()
    mesh.vertex_to_edges[0].add(999)

    mesh.build_connectivity_maps()
    # If the call is cached, it should not rebuild and should preserve the mutation.
    assert 999 in mesh.vertex_to_edges[0]


def test_build_connectivity_maps_rebuilds_after_topology_bump():
    mesh = _build_single_triangle_mesh()
    mesh.increment_topology_version()

    mesh.build_connectivity_maps()
    mesh.vertex_to_edges[0].add(999)
    assert 999 in mesh.vertex_to_edges[0]

    mesh.increment_topology_version()
    mesh.build_connectivity_maps()
    # After a topology bump, connectivity maps should rebuild and drop mutations.
    assert 999 not in mesh.vertex_to_edges[0]


def test_boundary_vertex_ids_invalidates_on_topology_bump():
    mesh = _build_single_triangle_mesh()
    mesh.increment_topology_version()

    boundary_1 = mesh.boundary_vertex_ids
    assert boundary_1 == {0, 1, 2}

    mesh.vertices[3] = Vertex(3, np.array([1.0, 1.0, 0.0]))
    mesh.edges[4] = Edge(4, 2, 3)
    mesh.edges[5] = Edge(5, 3, 0)
    # Triangle 1: (0,2,3) shares edge (2,0) with the original triangle.
    mesh.facets[1] = Facet(1, edge_indices=[-3, 4, 5])

    mesh.increment_topology_version()
    boundary_2 = mesh.boundary_vertex_ids
    assert boundary_2 == {0, 1, 2, 3}
