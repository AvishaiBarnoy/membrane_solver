import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.cache_checks import (
    barycentric_cache_valid,
    is_cached_positions,
    p1_triangle_cache_valid,
    triangle_areas_cache_valid,
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
