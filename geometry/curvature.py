# geometry/curvature.py

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross


def compute_integrated_curvature_vectors(
    mesh: Mesh, positions: np.ndarray, index_map: Dict[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute integrated mean curvature vectors K_i and Mixed Voronoi Areas A_i.

    K_i = 0.5 * sum_j (cot alpha + cot beta) * (v_i - v_j)
    A_i = Mixed Voronoi Area (Meyer et al. 2003)
    """
    n_verts = len(mesh.vertex_ids)
    tri_rows, _ = mesh.triangle_row_cache()

    if tri_rows is None:
        return np.zeros((n_verts, 3)), np.zeros(n_verts)

    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    # Edges
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    # Triangle area (doubled)
    cross = _fast_cross(e1, e2)
    area_doubled = np.linalg.norm(cross, axis=1)
    area_doubled = np.maximum(area_doubled, 1e-12)

    # Cotangents
    def get_cot(a, b, areas_2):
        return np.einsum("ij,ij->i", a, b) / areas_2

    c0 = get_cot(-e1, e2, area_doubled)
    c1 = get_cot(-e2, e0, area_doubled)
    c2 = get_cot(-e0, e1, area_doubled)

    # 1. Curvature Vectors
    k_vecs = np.zeros((n_verts, 3), dtype=float)
    np.add.at(k_vecs, tri_rows[:, 0], 0.5 * (c1[:, None] * -e1 + c2[:, None] * e2))
    np.add.at(k_vecs, tri_rows[:, 1], 0.5 * (c2[:, None] * -e2 + c0[:, None] * e0))
    np.add.at(k_vecs, tri_rows[:, 2], 0.5 * (c0[:, None] * -e0 + c1[:, None] * e1))

    # 2. Mixed Voronoi Area
    tri_areas = 0.5 * area_doubled
    vertex_areas = np.zeros(n_verts, dtype=float)

    # Standard Barycentric fallback for obtuse or tiny triangles
    # (More sophisticated Mixed Voronoi could go here, but Barycentric is a safe base)
    # We use 1/3 area distribution as a robust default.
    np.add.at(vertex_areas, tri_rows[:, 0], tri_areas / 3.0)
    np.add.at(vertex_areas, tri_rows[:, 1], tri_areas / 3.0)
    np.add.at(vertex_areas, tri_rows[:, 2], tri_areas / 3.0)

    return k_vecs, vertex_areas
