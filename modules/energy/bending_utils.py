"""Utility and geometry helpers for Helfrich/Willmore bending energy."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.curvature import compute_curvature_data
from geometry.entities import Mesh


def _vertex_normals(
    mesh: Mesh, positions: np.ndarray, tri_rows: np.ndarray
) -> np.ndarray:
    normals = np.zeros((len(mesh.vertex_ids), 3), dtype=float)

    if tri_rows.size == 0:
        return normals

    tri_pos = positions[tri_rows]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]
    tri_normals = np.cross(v1 - v0, v2 - v0)

    np.add.at(normals, tri_rows[:, 0], tri_normals)
    np.add.at(normals, tri_rows[:, 1], tri_normals)
    np.add.at(normals, tri_rows[:, 2], tri_normals)
    nrm = np.linalg.norm(normals, axis=1)
    mask = nrm > 1e-15
    normals[mask] /= nrm[mask, None]

    return normals


def _compute_effective_areas(
    mesh: Mesh,
    positions: np.ndarray,
    tri_rows: np.ndarray,
    weights: np.ndarray,
    index_map: Dict[int, int],
    *,
    cache_token: str = "default",
    compute_vertex_areas: bool = True,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Compute effective vertex areas by redistributing boundary vertex contributions.

    Returns (vertex_areas_eff, va0, va1, va2).
    """
    _ = index_map
    is_cached_pos = mesh._geometry_cache_active(positions)
    key_vertex = f"vertex_areas_eff::{cache_token}"
    key_va0 = f"va0_eff::{cache_token}"
    key_va1 = f"va1_eff::{cache_token}"
    key_va2 = f"va2_eff::{cache_token}"
    if is_cached_pos and mesh._curvature_version == mesh._version:
        c = mesh._curvature_cache
        if len(c.get(key_va0, ())) == int(tri_rows.shape[0]):
            vertex_cached = c.get(key_vertex)
            if vertex_cached is not None:
                return (
                    vertex_cached if compute_vertex_areas else None,
                    c[key_va0],
                    c[key_va1],
                    c[key_va2],
                )
            if not compute_vertex_areas:
                return None, c[key_va0], c[key_va1], c[key_va2]

    n_verts = len(mesh.vertex_ids)
    if tri_rows.size == 0:
        return (
            np.zeros(n_verts) if compute_vertex_areas else None,
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
        )

    use_raw_area_cache = False
    if is_cached_pos and mesh._curvature_version == mesh._version:
        c = mesh._curvature_cache
        va0_raw = np.asarray(c.get("va0_raw", ()), dtype=float)
        va1_raw = np.asarray(c.get("va1_raw", ()), dtype=float)
        va2_raw = np.asarray(c.get("va2_raw", ()), dtype=float)
        nf = int(tri_rows.shape[0])
        if va0_raw.shape == (nf,) and va1_raw.shape == (nf,) and va2_raw.shape == (nf,):
            use_raw_area_cache = True
            va0 = va0_raw
            va1 = va1_raw
            va2 = va2_raw

    if not use_raw_area_cache:
        v0 = positions[tri_rows[:, 0]]
        v1 = positions[tri_rows[:, 1]]
        v2 = positions[tri_rows[:, 2]]

        e0 = v2 - v1
        e1 = v0 - v2
        e2 = v1 - v0

        l0_sq = np.einsum("ij,ij->i", e0, e0)
        l1_sq = np.einsum("ij,ij->i", e1, e1)
        l2_sq = np.einsum("ij,ij->i", e2, e2)

        c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]

        n = np.cross(v1 - v0, v2 - v0)
        tri_areas = 0.5 * np.linalg.norm(n, axis=1)
        tri_areas = np.maximum(tri_areas, 1e-12)

        is_obtuse_v0 = c0 < 0
        is_obtuse_v1 = c1 < 0
        is_obtuse_v2 = c2 < 0
        any_obtuse = is_obtuse_v0 | is_obtuse_v1 | is_obtuse_v2

        va0 = np.where(~any_obtuse, (l1_sq * c1 + l2_sq * c2) / 8.0, 0.0)
        va1 = np.where(~any_obtuse, (l2_sq * c2 + l0_sq * c0) / 8.0, 0.0)
        va2 = np.where(~any_obtuse, (l0_sq * c0 + l1_sq * c1) / 8.0, 0.0)

        va0 = np.where(is_obtuse_v0, tri_areas / 2.0, va0)
        va0 = np.where(is_obtuse_v1 | is_obtuse_v2, tri_areas / 4.0, va0)
        va1 = np.where(is_obtuse_v1, tri_areas / 2.0, va1)
        va1 = np.where(is_obtuse_v0 | is_obtuse_v2, tri_areas / 4.0, va1)
        va2 = np.where(is_obtuse_v2, tri_areas / 2.0, va2)
        va2 = np.where(is_obtuse_v0 | is_obtuse_v1, tri_areas / 4.0, va2)

    boundary_rows = np.array(
        [
            index_map[vid]
            for vid in (mesh.boundary_vertex_ids or [])
            if vid in index_map
        ],
        dtype=int,
    )
    is_boundary = np.zeros(n_verts, dtype=bool)
    if boundary_rows.size:
        is_boundary[boundary_rows] = True

    tri_is_b = is_boundary[tri_rows]
    interior_mask = ~tri_is_b
    interior_counts = np.sum(interior_mask, axis=1)

    va_eff = np.stack([va0, va1, va2], axis=1)

    mask_has_interior = interior_counts > 0
    mask_some_boundary = np.any(tri_is_b, axis=1)
    to_redistribute = mask_has_interior & mask_some_boundary

    if np.any(to_redistribute):
        b_area_sums = np.sum(va_eff * tri_is_b, axis=1)
        extra_per_int = np.zeros_like(b_area_sums)
        extra_per_int[to_redistribute] = (
            b_area_sums[to_redistribute] / interior_counts[to_redistribute]
        )

        va_eff[to_redistribute] = (
            va_eff[to_redistribute] * interior_mask[to_redistribute]
            + interior_mask[to_redistribute] * extra_per_int[to_redistribute, None]
        )

    if is_cached_pos and mesh._curvature_version == mesh._version:
        mesh._curvature_cache[key_va0] = va_eff[:, 0]
        mesh._curvature_cache[key_va1] = va_eff[:, 1]
        mesh._curvature_cache[key_va2] = va_eff[:, 2]
    if not compute_vertex_areas:
        return None, va_eff[:, 0], va_eff[:, 1], va_eff[:, 2]

    vertex_areas_eff = np.zeros(n_verts, dtype=float)
    np.add.at(vertex_areas_eff, tri_rows[:, 0], va_eff[:, 0])
    np.add.at(vertex_areas_eff, tri_rows[:, 1], va_eff[:, 1])
    np.add.at(vertex_areas_eff, tri_rows[:, 2], va_eff[:, 2])

    if is_cached_pos and mesh._curvature_version == mesh._version:
        mesh._curvature_cache[key_vertex] = vertex_areas_eff

    return vertex_areas_eff, va_eff[:, 0], va_eff[:, 1], va_eff[:, 2]


def _mean_curvature_vectors(
    mesh: Mesh, positions: np.ndarray, index_map: Dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Hn, |Hn|, A, weights, tri_rows)."""
    mesh.build_position_cache()
    k_vecs, vertex_areas, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )
    if tri_rows.size == 0:
        n = len(mesh.vertex_ids)
        return (
            np.zeros((n, 3), dtype=float),
            np.zeros(n, dtype=float),
            np.zeros(n, dtype=float),
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=int),
        )

    safe_areas = np.maximum(vertex_areas, 1e-12)
    h_vecs = k_vecs / (2.0 * safe_areas[:, None])
    h_mag = np.linalg.norm(h_vecs, axis=1)
    return h_vecs, h_mag, vertex_areas, weights, tri_rows
