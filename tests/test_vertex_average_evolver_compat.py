import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.vertex_average import vertex_average


def _build_two_triangle_mesh(*, flip_second: bool) -> Mesh:
    """Create two triangles sharing an edge, optionally with opposing orientation."""
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[2] = Vertex(2, np.array([0.0, 1.0, 0.0]))
    mesh.vertices[3] = Vertex(3, np.array([1.0, 1.0, 0.0]))

    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 3)
    mesh.edges[3] = Edge(3, 3, 2)
    mesh.edges[4] = Edge(4, 2, 0)
    mesh.edges[5] = Edge(5, 1, 2)

    # First triangle (0,1,2) uses edges 1,5,4.
    mesh.facets[0] = Facet(0, [1, 5, 4])

    # Second triangle uses (1,3,2) with either consistent or flipped orientation.
    if flip_second:
        mesh.facets[1] = Facet(1, [-3, -2, -5])
    else:
        mesh.facets[1] = Facet(1, [2, 3, -5])

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    mesh.increment_topology_version()
    return mesh


def test_vertex_average_ignores_facet_orientation_for_weights() -> None:
    """Regression: averaging should not depend on facet orientation sign."""
    mesh_a = _build_two_triangle_mesh(flip_second=False)
    mesh_b = _build_two_triangle_mesh(flip_second=True)

    vertex_average(mesh_a)
    vertex_average(mesh_b)

    for vid in mesh_a.vertices:
        assert np.allclose(mesh_a.vertices[vid].position, mesh_b.vertices[vid].position)
