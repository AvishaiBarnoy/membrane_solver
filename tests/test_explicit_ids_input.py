import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry


def test_parse_geometry_supports_explicit_vertex_edge_face_ids():
    data = {
        "vertices": {
            "10": [0.0, 0.0, 0.0],
            20: [1.0, 0.0, 0.0],
            30: [0.0, 1.0, 0.0],
        },
        "edges": {
            1: [10, 20],
            2: [20, 30],
            3: [30, 10],
        },
        "faces": {
            100: [1, 2, 3],
            101: ["r1", 2, 3],
        },
        "instructions": [],
    }
    mesh = parse_geometry(data)

    assert set(mesh.vertices) == {10, 20, 30}
    assert np.allclose(mesh.vertices[20].position, np.array([1.0, 0.0, 0.0]))

    assert set(mesh.edges) == {1, 2, 3}
    assert mesh.edges[1].tail_index == 10
    assert mesh.edges[1].head_index == 20

    assert set(mesh.facets) == {100, 101}
    assert mesh.facets[100].edge_indices == [1, 2, 3]
    assert mesh.facets[101].edge_indices[0] == -1


def test_parse_geometry_supports_explicit_body_ids():
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "bodies": {
            "7": {
                "faces": [0],
                "target_volume": 0.0,
            }
        },
        "instructions": [],
    }
    mesh = parse_geometry(data)
    assert set(mesh.bodies) == {7}
    body = mesh.bodies[7]
    assert body.index == 7
    assert body.facet_indices == [0]
