"""Cache validity predicates for geometry cache helpers."""

from __future__ import annotations

import numpy as np


def is_cached_positions(
    positions: np.ndarray | None, positions_cache: np.ndarray | None
) -> bool:
    """Return whether ``positions`` refers to the active mesh position cache."""
    return positions is None or positions is positions_cache


def triangle_areas_cache_valid(
    *,
    is_cached_pos: bool,
    cached_version: int,
    mesh_version: int,
    cached_areas: np.ndarray | None,
) -> bool:
    """Return whether triangle areas/normals cache can be reused."""
    return is_cached_pos and cached_version == mesh_version and cached_areas is not None


def barycentric_cache_valid(
    *,
    use_cache: bool,
    cached_version: int,
    mesh_version: int,
    cached_rows_version: int,
    loops_version: int,
    cached_values: np.ndarray | None,
    expected_size: int,
) -> bool:
    """Return whether cached barycentric vertex areas can be reused."""
    return (
        use_cache
        and cached_version == mesh_version
        and cached_rows_version == loops_version
        and cached_values is not None
        and len(cached_values) == expected_size
    )


def vertex_normals_cache_valid(
    *,
    is_cached_pos: bool,
    cached_values: np.ndarray | None,
    cached_version: int,
    mesh_version: int,
    cached_loops_version: int,
    loops_version: int,
) -> bool:
    """Return whether cached per-vertex normals can be reused."""
    return (
        is_cached_pos
        and cached_values is not None
        and cached_version == mesh_version
        and cached_loops_version == loops_version
    )


def p1_triangle_cache_valid(
    *,
    use_cache: bool,
    cached_version: int,
    mesh_version: int,
    cached_rows_version: int,
    loops_version: int,
    cached_area: np.ndarray | None,
    cached_g0: np.ndarray | None,
    cached_g1: np.ndarray | None,
    cached_g2: np.ndarray | None,
) -> bool:
    """Return whether cached P1 triangle gradients can be reused."""
    return (
        use_cache
        and cached_version == mesh_version
        and cached_rows_version == loops_version
        and cached_area is not None
        and cached_g0 is not None
        and cached_g1 is not None
        and cached_g2 is not None
    )
