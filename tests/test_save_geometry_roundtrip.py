import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from geometry.geom_io import load_data, parse_geometry, save_geometry


def test_save_geometry_reindexes_sparse_ids(tmp_path):
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        5: Vertex(5, np.array([1.0, 0.0, 0.0])),
        9: Vertex(9, np.array([0.0, 1.0, 0.0])),
    }
    # Sparse edge IDs, including a large value.
    mesh.edges = {
        1: Edge(1, 0, 5),
        500: Edge(500, 5, 9),
        9001: Edge(9001, 9, 0),
    }
    # Single triangle using the sparse edges.
    mesh.facets = {100: Facet(100, [1, 500, 9001])}
    mesh.bodies = {7: Body(7, [100], target_volume=None, options={})}

    out_path = tmp_path / "roundtrip.json"
    save_geometry(mesh, str(out_path))

    data = load_data(str(out_path))
    roundtrip = parse_geometry(data)

    # A successful parse implies all face edge indices exist in the edge list.
    roundtrip.validate_edge_indices()

    assert len(roundtrip.vertices) == 3
    assert len(roundtrip.edges) == 3
    assert len(roundtrip.facets) == 1
    assert len(roundtrip.bodies) == 1
