# geometry/curvature.py

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross


def compute_curvature_data(
    mesh: Mesh, positions: np.ndarray, index_map: Dict[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute curvature vectors, dual areas, and the cotangent weight matrix.

    Returns:
        k_vecs: (N_verts, 3) integrated curvature vectors
        vertex_areas: (N_verts,) dual vertex areas
        weights: (N_tri, 3) cotangent weights for each triangle angle
        tri_rows: (N_tri, 3) vertex indices for triangles
    """
    n_verts = len(mesh.vertex_ids)
    tri_rows, _ = mesh.triangle_row_cache()

    if tri_rows is None:
        return np.zeros((n_verts, 3)), np.zeros(n_verts), np.array([]), np.array([])

    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    def get_cot(a, b):
        dot = np.einsum("ij,ij->i", a, b)
        cross_mag = np.linalg.norm(_fast_cross(a, b), axis=1)
        return dot / np.maximum(cross_mag, 1e-12)

    # Cotangents opposite to edges
    c0 = get_cot(-e1, e2)
    c1 = get_cot(-e2, e0)
    c2 = get_cot(-e0, e1)

    # Store weights for analytical gradient
    weights = np.column_stack([c0, c1, c2])

    k_vecs = np.zeros((n_verts, 3), dtype=float)
    # K_i = 0.5 * sum_j (cot alpha + cot beta) * (v_i - v_j)
    # Tri (0,1,2) contrib to K0: 0.5 * (c1*(v0-v2) + c2*(v0-v1))
    np.add.at(k_vecs, tri_rows[:, 0], 0.5 * (c1[:, None] * -e1 + c2[:, None] * e2))
    np.add.at(k_vecs, tri_rows[:, 1], 0.5 * (c2[:, None] * -e2 + c0[:, None] * e0))
    np.add.at(k_vecs, tri_rows[:, 2], 0.5 * (c0[:, None] * -e0 + c1[:, None] * e1))

    tri_areas = 0.5 * np.linalg.norm(_fast_cross(e1, e2), axis=1)
    vertex_areas = np.zeros(n_verts, dtype=float)
    np.add.at(vertex_areas, tri_rows[:, 0], tri_areas / 3.0)
    np.add.at(vertex_areas, tri_rows[:, 1], tri_areas / 3.0)
    np.add.at(vertex_areas, tri_rows[:, 2], tri_areas / 3.0)

    return k_vecs, vertex_areas, weights, tri_rows
