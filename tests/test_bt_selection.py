import numpy as np
import pytest

from geometry.entities import Vertex
from modules.energy.bt_selection import (
    _apply_inner_divergence_update_mode,
    _base_term_region_zero_rows,
    _collect_group_rows,
    _collect_preset_rows,
    _interior_mask_leaflet,
    _shared_rim_support_transition_triangle_mask,
)


class MockMesh:
    def __init__(self, vertices_data):
        self.vertex_ids = list(vertices_data.keys())
        # Ensure we use real Vertex objects with correct arguments
        self.vertices = {
            vid: Vertex(index=vid, position=np.array(pos), options=opts)
            for vid, (pos, opts) in vertices_data.items()
        }
        self._vertex_ids_version = 1
        self._topology_version = 1
        self.boundary_vertex_ids = []
        self.global_parameters = {}

    def build_position_cache(self):
        pass


@pytest.fixture
def basic_mesh():
    # v0 at (0,0), v1 at (1,0), v2 at (0,1), v3 at (2,0)
    vertices_data = {
        0: ([0.0, 0.0, 0.0], {"preset": "disk", "rim_slope_match_group": "group1"}),
        1: ([1.0, 0.0, 0.0], {"preset": "rim", "rim_slope_match_group": "group2"}),
        2: ([0.0, 1.0, 0.0], {"preset": "other", "rim_slope_match_group": "group1"}),
        3: ([2.0, 0.0, 0.0], {"preset": "rim"}),
    }
    return MockMesh(vertices_data)


def test_collect_preset_rows(basic_mesh):
    index_map = {0: 0, 1: 1, 2: 2, 3: 3}

    # Selection by preset tag
    rows = _collect_preset_rows(
        basic_mesh, presets=("disk",), cache_tag="test", index_map=index_map
    )
    assert np.array_equal(rows, [0])

    rows = _collect_preset_rows(
        basic_mesh, presets=("rim",), cache_tag="test2", index_map=index_map
    )
    assert set(rows) == {1, 3}

    # Empty/no-match behavior
    rows = _collect_preset_rows(
        basic_mesh, presets=(), cache_tag="test3", index_map=index_map
    )
    assert rows.size == 0


def test_collect_group_rows(basic_mesh):
    index_map = {0: 0, 1: 1, 2: 2, 3: 3}

    # Selection by group ID
    # _collect_group_rows uses _BASE_TERM_BOUNDARY_OPTION_KEYS
    # which includes "rim_slope_match_group"
    rows = _collect_group_rows(basic_mesh, group="group1", index_map=index_map)
    assert set(rows) == {0, 2}

    rows = _collect_group_rows(basic_mesh, group="group2", index_map=index_map)
    assert np.array_equal(rows, [1])

    rows = _collect_group_rows(basic_mesh, group="nonexistent", index_map=index_map)
    assert rows.size == 0


def test_base_term_region_zero_rows(basic_mesh):
    index_map = {0: 0, 1: 1, 2: 2, 3: 3}
    global_params = {
        "bending_tilt_base_term_region_mode": "disk_only_base_term_v1",
        "bending_tilt_base_term_region_radius": 1.5,
        "bending_tilt_assume_J0_center_x": 0.0,
        "bending_tilt_assume_J0_center_y": 0.0,
    }

    # disk_only_base_term_v1 zeroes out rows > radius for cache_tag="in"
    rows = _base_term_region_zero_rows(
        basic_mesh, global_params, cache_tag="in", index_map=index_map
    )
    assert np.array_equal(rows, [3])  # v3 is at 2.0 > 1.5

    # physical_disk_split_v1 zeroes out rows <= radius for cache_tag="out"
    global_params["bending_tilt_base_term_region_mode"] = "physical_disk_split_v1"
    rows = _base_term_region_zero_rows(
        basic_mesh, global_params, cache_tag="out", index_map=index_map
    )
    assert set(rows) == {0, 1, 2}  # v0, v1, v2 are <= 1.5


def test_interior_mask_leaflet(basic_mesh):
    index_map = {0: 0, 1: 1, 2: 2, 3: 3}
    basic_mesh.boundary_vertex_ids = [3]
    global_params = {"bending_tilt_base_term_boundary_group_in": "group1"}

    # Boundary vertex (3) and group1 vertices (0, 2) should be False
    mask = _interior_mask_leaflet(
        basic_mesh, global_params, cache_tag="in", index_map=index_map
    )
    assert np.array_equal(mask, [False, True, False, False])


def test_apply_inner_divergence_update_mode(basic_mesh):
    # Use real mode name
    global_params = {
        "bending_tilt_in_update_mode": "outer_near_divergence_cap_v1",
        "benchmark_disk_radius": 1.0,
        "benchmark_lambda_value": 0.1,
        "bending_tilt_assume_J0_center_x": 0.0,
        "bending_tilt_assume_J0_center_y": 0.0,
    }

    pos = np.array(
        [
            [1.2, 0.0, 0.0],
            [1.3, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [1.0, 0.0, 0.0],  # rim vertex
        ]
    )
    all_tri_rows = np.array([[0, 1, 2], [3, 3, 3]])
    div_term = np.array(
        [10.0, 2.0]
    )  # 10.0 is the one to cap, 2.0 is the rim mag reference

    updated = _apply_inner_divergence_update_mode(
        basic_mesh,
        global_params,
        positions=pos,
        tri_rows=all_tri_rows,
        cache_tag="in",
        div_term=div_term,
    )

    # cap_magnitude = 1.05 * median(abs(div_term[rim_mask])) = 1.05 * 2.0 = 2.1
    # 10.0 > 2.1 so it should be capped to 2.1
    assert updated[0] == pytest.approx(2.1)
    assert updated[1] == 2.0


def test_shared_rim_support_transition_triangle_mask(basic_mesh):
    tri_rows = np.array([[0, 1, 2], [1, 2, 3]])
    global_params = {
        "rim_slope_match_mode": "shared_rim_staggered_v1",
        "rim_slope_match_outer_group": "support",
        "rim_slope_match_group": "rim",
        "rim_slope_match_disk_group": "disk",
    }

    # Tag vertices
    basic_mesh.vertices[0].options["rim_slope_match_group"] = "support"
    basic_mesh.vertices[1].options["rim_slope_match_group"] = "rim"
    basic_mesh.vertices[2].options["rim_slope_match_group"] = "disk"
    basic_mesh.vertices[3].options["rim_slope_match_group"] = "free"

    # Triangle [0, 1, 2] has vertex 0 (support) -> should be masked
    # Triangle [1, 2, 3] does NOT have vertex 0 -> should NOT be masked
    mask = _shared_rim_support_transition_triangle_mask(
        basic_mesh, global_params, tri_rows
    )
    assert np.array_equal(mask, [True, False])
