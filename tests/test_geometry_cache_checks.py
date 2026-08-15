import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.cache_checks import (
    barycentric_cache_valid,
    connectivity_cache_valid,
    field_mask_cache_valid,
    geometry_freeze_cache_active,
    is_cached_positions,
    p1_triangle_cache_valid,
    topology_cache_valid,
    triangle_areas_cache_valid,
    vector_field_cache_needs_rebind,
    vertex_normals_cache_valid,
)


def test_is_cached_positions_identity_semantics():
    cached = np.zeros((2, 3))
    assert is_cached_positions(None, cached)
    assert is_cached_positions(cached, cached)
    assert not is_cached_positions(np.zeros((2, 3)), cached)


def test_triangle_and_vertex_cache_predicates():
    arr = np.zeros(3)
    assert triangle_areas_cache_valid(
        is_cached_pos=True, cached_version=2, mesh_version=2, cached_areas=arr
    )
    assert not triangle_areas_cache_valid(
        is_cached_pos=False, cached_version=2, mesh_version=2, cached_areas=arr
    )
    assert vertex_normals_cache_valid(
        is_cached_pos=True,
        cached_values=np.zeros((2, 3)),
        cached_version=5,
        mesh_version=5,
        cached_loops_version=7,
        loops_version=7,
    )
    assert not vertex_normals_cache_valid(
        is_cached_pos=True,
        cached_values=None,
        cached_version=5,
        mesh_version=5,
        cached_loops_version=7,
        loops_version=7,
    )


def test_field_mask_cache_requires_matching_flag_and_row_versions():
    mask = np.array([True, False])
    assert field_mask_cache_valid(
        cached_mask=mask,
        cached_flags_version=2,
        flags_version=2,
        cached_vertex_version=5,
        vertex_version=5,
        expected_size=2,
    )
    assert not field_mask_cache_valid(
        cached_mask=mask,
        cached_flags_version=2,
        flags_version=3,
        cached_vertex_version=5,
        vertex_version=5,
        expected_size=2,
    )


def test_geometry_freeze_cache_predicate_preserves_live_and_frozen_identity_rules():
    live = np.zeros((2, 3))
    frozen = live.copy()
    common = {
        "positions_cache": live,
        "positions_cache_version": 3,
        "mesh_version": 3,
        "freeze_depth": 1,
        "freeze_version": 3,
        "freeze_loops_version": 4,
        "loops_version": 4,
        "freeze_positions_id": id(frozen),
    }

    assert geometry_freeze_cache_active(positions=None, **common)
    assert geometry_freeze_cache_active(positions=live, **common)
    assert geometry_freeze_cache_active(positions=frozen, **common)
    assert not geometry_freeze_cache_active(
        positions=frozen, **{**common, "freeze_loops_version": 5}
    )


def test_vector_field_cache_rebind_predicate_requires_shape_count_and_row_epoch():
    values = np.zeros((2, 3))
    common = {
        "cached_values": values,
        "expected_count": 2,
        "cached_count": 2,
        "cached_vertex_version": 7,
        "vertex_version": 7,
    }
    assert not vector_field_cache_needs_rebind(**common)
    assert vector_field_cache_needs_rebind(**{**common, "cached_count": 3})
    assert vector_field_cache_needs_rebind(
        **{**common, "cached_values": np.zeros((2, 2))}
    )


def test_topology_cache_predicates_require_epoch_and_connectivity_counts():
    assert topology_cache_valid(cached_version=4, topology_version=4)
    assert not topology_cache_valid(cached_version=3, topology_version=4)
    assert connectivity_cache_valid(
        cached_version=4,
        topology_version=4,
        cached_counts=(3, 3, 1),
        current_counts=(3, 3, 1),
    )
    assert not connectivity_cache_valid(
        cached_version=4,
        topology_version=4,
        cached_counts=(3, 3, 1),
        current_counts=(4, 3, 1),
    )


def test_barycentric_and_p1_cache_predicates():
    vals = np.zeros(4)
    assert barycentric_cache_valid(
        use_cache=True,
        cached_version=1,
        mesh_version=1,
        cached_rows_version=3,
        loops_version=3,
        cached_values=vals,
        expected_size=4,
    )
    assert not barycentric_cache_valid(
        use_cache=True,
        cached_version=1,
        mesh_version=1,
        cached_rows_version=3,
        loops_version=3,
        cached_values=vals,
        expected_size=5,
    )

    g = np.zeros((1, 3))
    assert p1_triangle_cache_valid(
        use_cache=True,
        cached_version=2,
        mesh_version=2,
        cached_rows_version=4,
        loops_version=4,
        cached_area=np.zeros(1),
        cached_g0=g,
        cached_g1=g,
        cached_g2=g,
    )
    assert not p1_triangle_cache_valid(
        use_cache=True,
        cached_version=2,
        mesh_version=2,
        cached_rows_version=4,
        loops_version=4,
        cached_area=None,
        cached_g0=g,
        cached_g1=g,
        cached_g2=g,
    )
