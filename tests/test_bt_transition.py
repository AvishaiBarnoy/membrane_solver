import numpy as np
import pytest

from geometry.entities import Vertex
from modules.energy.bt_transition import (
    _OUTER_TRANSITION_OPERATOR_SHELLS,
    _outer_transition_operator_payload,
)


class MockMesh:
    def __init__(self, vertices_data, tris):
        self.vertices = {
            vid: Vertex(vid, np.array(pos), opts)
            for vid, (pos, opts) in vertices_data.items()
        }
        self.vertex_ids = np.array(sorted(self.vertices.keys()), dtype=int)
        self._vertex_ids_version = 1
        self._facet_loops_version = 1
        self.global_parameters = {}

    def build_position_cache(self):
        pass


@pytest.fixture
def transition_mesh():
    # Setup a mesh with some "partial" and some "full" participation vertices
    # This is complex to mock fully, but we can check if the payload builder runs
    vertices_data = {
        0: ([0.0, 0.0, 0.0], {"preset": "disk"}),
        1: ([1.0, 0.0, 0.0], {"preset": "disk"}),
        2: ([0.0, 1.0, 0.0], {"preset": "disk"}),
        3: ([1.0, 1.0, 0.0], {"preset": "rim"}),
    }
    # T0: [0, 1, 2], T1: [1, 2, 3]
    tri_rows_full = np.array([[0, 1, 2], [1, 2, 3]])
    tri_rows = np.array([[1, 2, 3]])  # Only keep one triangle

    mesh = MockMesh(vertices_data, tri_rows_full)
    return mesh, tri_rows_full, tri_rows


def test_outer_transition_operator_payload_runs(transition_mesh):
    mesh, tri_rows_full, tri_rows = transition_mesh
    payload = _outer_transition_operator_payload(
        mesh, tri_rows_full=tri_rows_full, tri_rows=tri_rows
    )
    assert "domain" in payload
    assert "transition_mask" in payload
    assert "patch_vertex_mask" in payload


def test_transition_shells_constant():
    assert len(_OUTER_TRANSITION_OPERATOR_SHELLS) > 0
    assert all(isinstance(s, float) for s in _OUTER_TRANSITION_OPERATOR_SHELLS)
