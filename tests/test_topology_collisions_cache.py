import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Mesh, Vertex
from runtime.topology import detect_vertex_edge_collisions


def test_detect_vertex_edge_collisions_builds_cache():
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([0.0, 1.0, 0.0])),
    }
    mesh.edges = {1: Edge(1, 0, 1), 2: Edge(2, 1, 2)}

    # Should not crash even if mesh.vertex_ids is None.
    collisions = detect_vertex_edge_collisions(mesh, threshold=1e-6)
    assert isinstance(collisions, list)
