"""Connection-aware transport helpers for tangent vector fields."""

from __future__ import annotations

import numpy as np


def _normalize_rows(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row-normalized vectors and their norms."""
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    out = np.zeros_like(vectors)
    good = norms > 1.0e-15
    out[good] = vectors[good] / norms[good][:, None]
    return out, norms


def _cross_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return skew matrices for row-wise cross products."""
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]
    out = np.zeros((vectors.shape[0], 3, 3), dtype=float)
    out[:, 0, 1] = -z
    out[:, 0, 2] = y
    out[:, 1, 0] = z
    out[:, 1, 2] = -x
    out[:, 2, 0] = -y
    out[:, 2, 1] = x
    return out


def minimal_rotation_transport(
    src_normals: np.ndarray, dst_normals: np.ndarray
) -> np.ndarray:
    """Return row-wise minimal-rotation transport matrices from src to dst normals."""
    src_unit, _ = _normalize_rows(src_normals)
    dst_unit, _ = _normalize_rows(dst_normals)
    if src_unit.shape != dst_unit.shape or src_unit.ndim != 2 or src_unit.shape[1] != 3:
        raise ValueError("src_normals and dst_normals must both have shape (N, 3).")

    n = src_unit.shape[0]
    eye = np.broadcast_to(np.eye(3, dtype=float), (n, 3, 3)).copy()
    cross = np.cross(src_unit, dst_unit)
    dot = np.einsum("ij,ij->i", src_unit, dst_unit)

    good = (dot > -1.0 + 1.0e-10) & (np.linalg.norm(cross, axis=1) > 1.0e-15)
    if np.any(good):
        k = _cross_matrix(cross[good])
        denom = (1.0 / (1.0 + dot[good]))[:, None, None]
        eye_good = eye[good]
        eye[good] = eye_good + k + np.matmul(k, k) * denom

    anti = dot <= -1.0 + 1.0e-10
    if np.any(anti):
        src_anti = src_unit[anti]
        axis0 = np.tile(np.array([1.0, 0.0, 0.0], dtype=float), (src_anti.shape[0], 1))
        axis1 = np.tile(np.array([0.0, 1.0, 0.0], dtype=float), (src_anti.shape[0], 1))
        use_axis1 = np.abs(src_anti[:, 0]) > 0.9
        seed = np.where(use_axis1[:, None], axis1, axis0)
        tangent = seed - np.einsum("ij,ij->i", seed, src_anti)[:, None] * src_anti
        tangent, tangent_norm = _normalize_rows(tangent)
        if np.any(tangent_norm <= 1.0e-15):
            raise ValueError("Failed to build antiparallel transport axis.")
        eye[anti] = 2.0 * np.einsum("ni,nj->nij", tangent, tangent) - np.eye(3)

    return eye


def transport_vectors(vectors: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Apply row-wise transport rotations to vectors."""
    vectors = np.asarray(vectors, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3).")
    if rotations.shape != (vectors.shape[0], 3, 3):
        raise ValueError("rotations must have shape (N, 3, 3).")
    return np.einsum("nij,nj->ni", rotations, vectors)


def transport_vertex_tilts_to_triangle_planes(
    positions: np.ndarray,
    tri_rows: np.ndarray,
    normals: np.ndarray,
    tilts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transport per-vertex tilts into triangle tangent planes."""
    positions = np.asarray(positions, dtype=float)
    tri_rows = np.asarray(tri_rows, dtype=np.int32)
    normals = np.asarray(normals, dtype=float)
    tilts = np.asarray(tilts, dtype=float)
    if tri_rows.ndim != 2 or tri_rows.shape[1] != 3:
        raise ValueError("tri_rows must have shape (N_triangles, 3).")
    if positions.shape != normals.shape or positions.shape != tilts.shape:
        raise ValueError("positions, normals, and tilts must all have shape (N, 3).")
    if tri_rows.size == 0:
        zeros = np.zeros((0, 3), dtype=float)
        return zeros, zeros, zeros, zeros

    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]
    tri_normals = np.cross(v1 - v0, v2 - v0)
    tri_normals, tri_norms = _normalize_rows(tri_normals)
    if np.any(tri_norms <= 1.0e-15):
        raise ValueError("Degenerate triangle encountered during transport.")

    n0 = normals[tri_rows[:, 0]]
    n1 = normals[tri_rows[:, 1]]
    n2 = normals[tri_rows[:, 2]]
    t0 = tilts[tri_rows[:, 0]]
    t1 = tilts[tri_rows[:, 1]]
    t2 = tilts[tri_rows[:, 2]]

    r0 = minimal_rotation_transport(n0, tri_normals)
    r1 = minimal_rotation_transport(n1, tri_normals)
    r2 = minimal_rotation_transport(n2, tri_normals)

    t0_tri = transport_vectors(t0, r0)
    t1_tri = transport_vectors(t1, r1)
    t2_tri = transport_vectors(t2, r2)
    return t0_tri, t1_tri, t2_tri, tri_normals


def triangle_plane_transport_data(
    mesh,
    positions: np.ndarray,
    tri_rows: np.ndarray,
    *,
    cache_tag: str = "default",
) -> dict[str, np.ndarray]:
    """Return geometry-only transport data, using mesh-versioned caching when safe."""
    positions = np.asarray(positions, dtype=float)
    tri_rows = np.asarray(tri_rows, dtype=np.int32)
    if tri_rows.ndim != 2 or tri_rows.shape[1] != 3:
        raise ValueError("tri_rows must have shape (N_triangles, 3).")
    if tri_rows.size == 0:
        zeros = np.zeros((0, 3), dtype=float)
        zeros_rot = np.zeros((0, 3, 3), dtype=float)
        return {
            "tri_normals": zeros,
            "r0": zeros_rot,
            "r1": zeros_rot,
            "r2": zeros_rot,
            "r0_t": zeros_rot,
            "r1_t": zeros_rot,
            "r2_t": zeros_rot,
        }

    cacheable = False
    tri_rows_ref = None
    if mesh is not None:
        live_positions = mesh.positions_view()
        tri_rows_ref, _ = mesh.triangle_row_cache()
        cacheable = np.shares_memory(positions, live_positions) and (
            tri_rows_ref is not None and np.shares_memory(tri_rows, tri_rows_ref)
        )

    cache_key = None
    if cacheable:
        cache_key = (
            int(mesh._version),
            int(getattr(mesh, "_topology_version", -1)),
            str(cache_tag),
        )
        cached = getattr(mesh, "_triangle_plane_transport_cache", None)
        if isinstance(cached, dict) and cache_key in cached:
            return cached[cache_key]

    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]
    tri_normals = np.cross(v1 - v0, v2 - v0)
    tri_normals, tri_norms = _normalize_rows(tri_normals)
    if np.any(tri_norms <= 1.0e-15):
        raise ValueError("Degenerate triangle encountered during transport.")

    vertex_normals = np.zeros_like(positions)
    raw_tri_normals = np.cross(v1 - v0, v2 - v0)
    np.add.at(vertex_normals, tri_rows[:, 0], raw_tri_normals)
    np.add.at(vertex_normals, tri_rows[:, 1], raw_tri_normals)
    np.add.at(vertex_normals, tri_rows[:, 2], raw_tri_normals)
    vertex_normals, _ = _normalize_rows(vertex_normals)

    n0 = vertex_normals[tri_rows[:, 0]]
    n1 = vertex_normals[tri_rows[:, 1]]
    n2 = vertex_normals[tri_rows[:, 2]]
    r0 = minimal_rotation_transport(n0, tri_normals)
    r1 = minimal_rotation_transport(n1, tri_normals)
    r2 = minimal_rotation_transport(n2, tri_normals)
    out = {
        "tri_normals": tri_normals,
        "r0": r0,
        "r1": r1,
        "r2": r2,
        "r0_t": np.swapaxes(r0, 1, 2),
        "r1_t": np.swapaxes(r1, 1, 2),
        "r2_t": np.swapaxes(r2, 1, 2),
    }
    if cache_key is not None:
        cached = getattr(mesh, "_triangle_plane_transport_cache", None)
        if not isinstance(cached, dict):
            cached = {}
        cached[cache_key] = out
        mesh._triangle_plane_transport_cache = cached
    return out


def edge_transport_pairs(mesh) -> np.ndarray:
    """Return unique mesh edges as vertex-row index pairs."""
    pairs: list[tuple[int, int]] = []
    for edge in mesh.edges.values():
        tail = mesh.vertex_index_to_row.get(int(edge.tail_index))
        head = mesh.vertex_index_to_row.get(int(edge.head_index))
        if tail is None or head is None or tail == head:
            continue
        i, j = int(tail), int(head)
        if i > j:
            i, j = j, i
        pairs.append((i, j))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(sorted(set(pairs)), dtype=np.int32)


__all__ = [
    "edge_transport_pairs",
    "minimal_rotation_transport",
    "triangle_plane_transport_data",
    "transport_vectors",
    "transport_vertex_tilts_to_triangle_planes",
]
