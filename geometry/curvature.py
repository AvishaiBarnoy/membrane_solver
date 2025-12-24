# geometry/curvature.py

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross


def compute_curvature_data(
    mesh: Mesh, positions: np.ndarray, index_map: Dict[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute integrated mean curvature vectors K_i, Mixed Voronoi Areas A_i,
    and the cotangent weights.

    References: Meyer et al. (2003) 'Discrete Differential-Geometry Operators'
    """
    n_verts = len(mesh.vertex_ids)
    tri_rows, _ = mesh.triangle_row_cache()

    if tri_rows is None:
        return np.zeros((n_verts, 3)), np.zeros(n_verts), np.array([]), np.array([])

    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    # Edges
    e0 = v2 - v1  # opposite v0
    e1 = v0 - v2  # opposite v1
    e2 = v1 - v0  # opposite v2

    # Squared edge lengths
    l0_sq = np.einsum("ij,ij->i", e0, e0)
    l1_sq = np.einsum("ij,ij->i", e1, e1)
    l2_sq = np.einsum("ij,ij->i", e2, e2)

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

    # 1. Curvature Vectors (Integrated)
    k_vecs = np.zeros((n_verts, 3), dtype=float)
    np.add.at(k_vecs, tri_rows[:, 0], 0.5 * (c1[:, None] * -e1 + c2[:, None] * e2))
    np.add.at(k_vecs, tri_rows[:, 1], 0.5 * (c2[:, None] * -e2 + c0[:, None] * e0))
    np.add.at(k_vecs, tri_rows[:, 2], 0.5 * (c0[:, None] * -e0 + c1[:, None] * e1))

    # 2. Mixed Voronoi Area
    # Formulas from Meyer et al. 2003
    tri_areas = 0.5 * area_doubled
    vertex_areas = np.zeros(n_verts, dtype=float)

    # Check for obtuse angles
    # cot < 0 means angle > 90 deg
    is_obtuse_v0 = c0 < 0
    is_obtuse_v1 = c1 < 0
    is_obtuse_v2 = c2 < 0
    any_obtuse = is_obtuse_v0 | is_obtuse_v1 | is_obtuse_v2

    # Case A: Non-obtuse triangle (Standard Voronoi)
    # Area contrib to v0 = 1/8 * ( l1^2 * cot_beta + l2^2 * cot_gamma )
    va0 = np.where(~any_obtuse, (l1_sq * c1 + l2_sq * c2) / 8.0, 0.0)
    va1 = np.where(~any_obtuse, (l2_sq * c2 + l0_sq * c0) / 8.0, 0.0)
    va2 = np.where(~any_obtuse, (l0_sq * c0 + l1_sq * c1) / 8.0, 0.0)

    # Case B: Obtuse triangle
    # If angle at v is obtuse, area = T_area / 2
    # If other angle is obtuse, area = T_area / 4
    va0 = np.where(is_obtuse_v0, tri_areas / 2.0, va0)
    va0 = np.where(is_obtuse_v1 | is_obtuse_v2, tri_areas / 4.0, va0)

    va1 = np.where(is_obtuse_v1, tri_areas / 2.0, va1)
    va1 = np.where(is_obtuse_v0 | is_obtuse_v2, tri_areas / 4.0, va1)

    va2 = np.where(is_obtuse_v2, tri_areas / 2.0, va2)
    va2 = np.where(is_obtuse_v0 | is_obtuse_v1, tri_areas / 4.0, va2)

    np.add.at(vertex_areas, tri_rows[:, 0], va0)
    np.add.at(vertex_areas, tri_rows[:, 1], va1)
    np.add.at(vertex_areas, tri_rows[:, 2], va2)

    weights = np.column_stack([c0, c1, c2])
    return k_vecs, vertex_areas, weights, tri_rows
