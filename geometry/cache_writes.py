"""Helper writers for geometry cache fields on ``Mesh``."""

from __future__ import annotations

import numpy as np


def store_triangle_area_normals_cache(
    mesh, *, areas: np.ndarray, normals: np.ndarray
) -> None:
    """Store triangle areas and normals cache payload on a mesh."""
    mesh._cached_tri_areas = areas
    mesh._cached_tri_normals = normals
    mesh._cached_tri_areas_version = mesh._version


def store_barycentric_vertex_areas_cache(mesh, *, vertex_areas: np.ndarray) -> None:
    """Store barycentric vertex area cache payload on a mesh."""
    mesh._cached_barycentric_vertex_areas = vertex_areas
    mesh._cached_barycentric_vertex_areas_version = mesh._version
    mesh._cached_barycentric_vertex_areas_rows_version = mesh._facet_loops_version


def store_vertex_normals_cache(mesh, *, normals: np.ndarray) -> None:
    """Store vertex normal cache payload on a mesh."""
    mesh._cached_vertex_normals = normals
    mesh._cached_vertex_normals_version = mesh._version
    mesh._cached_vertex_normals_loops_version = mesh._facet_loops_version


def store_p1_triangle_grad_cache(
    mesh,
    *,
    area: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
) -> None:
    """Store P1 triangle shape-gradient cache payload on a mesh."""
    mesh._cached_p1_tri_areas = area
    mesh._cached_p1_tri_g0 = g0
    mesh._cached_p1_tri_g1 = g1
    mesh._cached_p1_tri_g2 = g2
    mesh._cached_p1_tri_grads_version = mesh._version
    mesh._cached_p1_tri_grads_rows_version = mesh._facet_loops_version
