"""Vectorized triangle geometry helpers used by ``geometry.entities``."""

from __future__ import annotations

import numpy as np


def _fast_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cross products for arrays of 3D vectors."""
    x = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    y = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    z = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    out = np.empty(x.shape + (3,), dtype=x.dtype)
    out[..., 0] = x
    out[..., 1] = y
    out[..., 2] = z
    return out


def triangle_normals_and_areas(
    positions: np.ndarray, tri_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return unnormalized triangle normals and triangle areas."""
    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]
    normals = _fast_cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(normals, axis=1)
    return normals, areas


def triangle_normals(positions: np.ndarray, tri_rows: np.ndarray) -> np.ndarray:
    """Return unnormalized triangle normals."""
    normals, _ = triangle_normals_and_areas(positions, tri_rows)
    return normals


def barycentric_vertex_areas_from_triangles(
    *,
    n_verts: int,
    tri_rows: np.ndarray,
    areas: np.ndarray,
) -> np.ndarray:
    """Accumulate one-third of each triangle area onto its three vertices."""
    vertex_areas = np.zeros(n_verts, dtype=float)
    if areas.size == 0:
        return vertex_areas
    area_thirds = areas / 3.0
    np.add.at(vertex_areas, tri_rows[:, 0], area_thirds)
    np.add.at(vertex_areas, tri_rows[:, 1], area_thirds)
    np.add.at(vertex_areas, tri_rows[:, 2], area_thirds)
    return vertex_areas


def vertex_unit_normals_from_triangles(
    *,
    n_verts: int,
    tri_rows: np.ndarray,
    tri_normals: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Accumulate triangle normals to vertices and normalize to unit length."""
    normals = np.zeros((n_verts, 3), dtype=float)
    if tri_rows.size == 0:
        return normals
    np.add.at(normals, tri_rows[:, 0], tri_normals)
    np.add.at(normals, tri_rows[:, 1], tri_normals)
    np.add.at(normals, tri_rows[:, 2], tri_normals)
    lens = np.linalg.norm(normals, axis=1)
    mask = lens >= eps
    normals[mask] /= lens[mask][:, None]
    return normals


def p1_triangle_shape_gradients(
    positions: np.ndarray, tri_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return P1 per-triangle area and shape gradients (g0, g1, g2)."""
    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    n = _fast_cross(v1 - v0, v2 - v0)
    n2 = np.einsum("ij,ij->i", n, n)
    denom = np.maximum(n2, 1e-20)

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    g0 = _fast_cross(n, e0) / denom[:, None]
    g1 = _fast_cross(n, e1) / denom[:, None]
    g2 = _fast_cross(n, e2) / denom[:, None]
    area = 0.5 * np.sqrt(np.maximum(n2, 0.0))
    return area, g0, g1, g2
