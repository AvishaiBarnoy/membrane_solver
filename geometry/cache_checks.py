"""Cache validity predicates for geometry cache helpers."""

from __future__ import annotations

import numpy as np


def is_cached_positions(
    positions: np.ndarray | None, positions_cache: np.ndarray | None
) -> bool:
    """Return whether ``positions`` refers to the active mesh position cache."""
    return positions is None or positions is positions_cache


def field_mask_cache_valid(
    *,
    cached_mask: np.ndarray | None,
    cached_flags_version: int,
    flags_version: int,
    cached_vertex_version: int,
    vertex_version: int,
    expected_size: int,
) -> bool:
    """Return whether a fixed-field mask matches its flag and row epochs."""
    return (
        cached_mask is not None
        and cached_flags_version == flags_version
        and cached_vertex_version == vertex_version
        and len(cached_mask) == expected_size
    )


def geometry_freeze_cache_active(
    *,
    positions: np.ndarray | None,
    positions_cache: np.ndarray | None,
    positions_cache_version: int,
    mesh_version: int,
    freeze_depth: int,
    freeze_version: int,
    freeze_loops_version: int,
    loops_version: int,
    freeze_positions_id: int | None,
) -> bool:
    """Return whether positions are eligible for the current geometry cache."""
    if positions is None:
        positions = positions_cache
    if positions is positions_cache:
        return positions_cache_version == mesh_version
    if freeze_depth <= 0:
        return False
    if freeze_version != mesh_version or freeze_loops_version != loops_version:
        return False
    return freeze_positions_id is not None and id(positions) == freeze_positions_id


def vector_field_cache_needs_rebind(
    *,
    cached_values: np.ndarray | None,
    expected_count: int,
    cached_count: int,
    cached_vertex_version: int,
    vertex_version: int,
) -> bool:
    """Return whether a dense three-vector field needs row rebinding."""
    return (
        cached_values is None
        or cached_values.shape != (expected_count, 3)
        or cached_count != expected_count
        or cached_vertex_version != vertex_version
    )


def topology_cache_valid(*, cached_version: int, topology_version: int) -> bool:
    """Return whether a topology-only cache matches the current epoch."""
    return cached_version == topology_version


def connectivity_cache_valid(
    *,
    cached_version: int,
    topology_version: int,
    cached_counts: tuple[int, int, int],
    current_counts: tuple[int, int, int],
) -> bool:
    """Return whether connectivity maps match their epoch and collection sizes."""
    return (
        topology_cache_valid(
            cached_version=cached_version, topology_version=topology_version
        )
        and cached_counts == current_counts
    )


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
