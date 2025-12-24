# geometry/curvature.py

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross


def compute_integrated_curvature_vectors(
    mesh: Mesh, positions: np.ndarray, index_map: Dict[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute integrated mean curvature vectors K_i and dual vertex areas A_i.
    Uses the Cotangent Weights discretization.

    Returns:
        K_vecs: (N_verts, 3) integrated curvature vectors
        vertex_areas: (N_verts,) dual vertex areas (barycentric)
    """
    n_verts = len(mesh.vertex_ids)
    tri_rows, _ = mesh.triangle_row_cache()

    if tri_rows is None:
        return np.zeros((n_verts, 3)), np.zeros(n_verts)

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

    c0 = get_cot(-e1, e2)
    c1 = get_cot(-e2, e0)
    c2 = get_cot(-e0, e1)

    k_vecs = np.zeros((n_verts, 3), dtype=float)
    np.add.at(k_vecs, tri_rows[:, 0], 0.5 * (c1[:, None] * -e1 + c2[:, None] * e2))
    np.add.at(k_vecs, tri_rows[:, 1], 0.5 * (c2[:, None] * -e2 + c0[:, None] * e0))
    np.add.at(k_vecs, tri_rows[:, 2], 0.5 * (c0[:, None] * -e0 + c1[:, None] * e1))

    tri_areas = 0.5 * np.linalg.norm(_fast_cross(e1, e2), axis=1)
    vertex_areas = np.zeros(n_verts, dtype=float)
    np.add.at(vertex_areas, tri_rows[:, 0], tri_areas / 3.0)
    np.add.at(vertex_areas, tri_rows[:, 1], tri_areas / 3.0)
    np.add.at(vertex_areas, tri_rows[:, 2], tri_areas / 3.0)

    return k_vecs, vertex_areas
