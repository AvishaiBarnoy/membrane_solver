import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.exceptions import InvalidEdgeIndexError
from geometry.entities import Edge, Facet, Mesh, Vertex


def build_minimal_mesh():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.facets[0] = Facet(0, [1])
    return mesh


def test_get_edge_with_zero_index_raises_custom_error():
    mesh = build_minimal_mesh()
    with pytest.raises(InvalidEdgeIndexError) as excinfo:
        mesh.get_edge(0)
    assert "Edge IDs are 1-based" in str(excinfo.value)


def test_connectivity_builder_raises_on_zero_index():
    mesh = build_minimal_mesh()
    mesh.facets[1] = Facet(1, [0, 1])
    with pytest.raises(InvalidEdgeIndexError):
        mesh.build_connectivity_maps()
