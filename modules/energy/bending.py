# modules/energy/bending.py

from typing import Dict

import numpy as np

from geometry.entities import Mesh, _fast_cross
from runtime.logging_config import setup_logging

logger = setup_logging("membrane_solver.log")


def _all_facets_are_triangles(mesh: Mesh) -> bool:
    """Return ``True`` if facet loops exist and all are triangles."""
    if not getattr(mesh, "facet_vertex_loops", None):
        return False
    tri_rows, tri_facets = mesh.triangle_row_cache()
    return tri_rows is not None and len(tri_facets) == len(mesh.facets)


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray,
) -> float:
    """
    Vectorized implementation of Bending Energy (Squared Mean Curvature Integral).
    Uses the Cotangent Weights discretization of the Laplace-Beltrami operator.

    Energy E = sum_i ( ||K_i||^2 / (4 * A_i) )
    where K_i is the integrated mean curvature vector and A_i is the dual area.
    """
    if not _all_facets_are_triangles(mesh):
        # Falling back to zero for non-triangle meshes for now
        # Standard DDG operators are defined for triangles.
        return 0.0

    n_verts = len(mesh.vertex_ids)
    tri_rows, _ = mesh.triangle_row_cache()

    # 1. Gather triangle vertex positions
    # tri_rows: (N_tri, 3) -> indices into 'positions'
    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    # 2. Compute edge vectors
    # e0: v1 -> v2, e1: v2 -> v0, e2: v0 -> v1
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    # 3. Compute cotangents for each angle in each triangle
    # cot(angle at v0) = (e1 . -e2) / ||e1 x -e2||
    def get_cot(a, b):
        dot = np.einsum("ij,ij->i", a, b)
        # cross magnitude is 2 * area
        cross = _fast_cross(a, b)
        cross_mag = np.linalg.norm(cross, axis=1)
        # Avoid division by zero for degenerate triangles
        return dot / np.maximum(cross_mag, 1e-12)

    # Angles opposite to edges:
    # cot_alpha (at v0, opposite e0): e1 . -e2
    # cot_beta  (at v1, opposite e1): e2 . -e0
    # cot_gamma (at v2, opposite e2): e0 . -e1
    c0 = get_cot(-e1, e2)
    c1 = get_cot(-e2, e0)
    c2 = get_cot(-e0, e1)

    # 4. Compute integrated curvature vector K_i at each vertex
    # K_i = 0.5 * sum_j (cot alpha + cot beta) * (v_i - v_j)
    # Term-wise: for triangle (0,1,2):
    # K_0 contrib: 0.5 * [ c1*(v0-v2) + c2*(v0-v1) ]
    # K_1 contrib: 0.5 * [ c2*(v1-v0) + c0*(v1-v2) ]
    # K_2 contrib: 0.5 * [ c0*(v2-v1) + c1*(v2-v0) ]

    k_vecs = np.zeros((n_verts, 3), dtype=float)

    # Scatter-add contributions from each triangle
    # We use 0.5 * cot * edge_vector
    np.add.at(k_vecs, tri_rows[:, 0], 0.5 * (c1[:, None] * -e1 + c2[:, None] * e2))
    np.add.at(k_vecs, tri_rows[:, 1], 0.5 * (c2[:, None] * -e2 + c0[:, None] * e0))
    np.add.at(k_vecs, tri_rows[:, 2], 0.5 * (c0[:, None] * -e0 + c1[:, None] * e1))

    # 5. Compute dual areas (Barycentric)
    # Area of triangle = 0.5 * ||e1 x e2||
    tri_areas = 0.5 * np.linalg.norm(_fast_cross(e1, e2), axis=1)
    vertex_areas = np.zeros(n_verts, dtype=float)
    # Distribute 1/3 of tri area to each vertex
    np.add.at(vertex_areas, tri_rows[:, 0], tri_areas / 3.0)
    np.add.at(vertex_areas, tri_rows[:, 1], tri_areas / 3.0)
    np.add.at(vertex_areas, tri_rows[:, 2], tri_areas / 3.0)

    # 6. Compute total Willmore Energy
    # E = sum_i ||K_i||^2 / (4 * A_i)
    k_sq_norms = np.einsum("ij,ij->i", k_vecs, k_vecs)

    # Mask for non-zero area (interior or closed mesh)
    area_mask = vertex_areas > 1e-12
    energy_per_vertex = k_sq_norms[area_mask] / (4.0 * vertex_areas[area_mask])

    total_energy = float(np.sum(energy_per_vertex))

    # Scale by global bending modulus if provided
    kappa = global_params.get("bending_modulus", 1.0)

    total_energy *= kappa

    # 7. Gradient (Simplified approximation)
    # The exact gradient of Willmore energy is the bi-Laplacian Delta^2 x.
    # For a first implementation, we can approximate the force as being proportional
    # to the curvature vector K_i scaled appropriately.
    # F_i = -kappa * Delta H ... this is complex.

    # TODO: Implement exact bi-Laplacian gradient.
    # For now, we return energy and a zero/placeholder gradient if not yet derived.
    # This allows 'print energy' and 'properties' to work.

    return total_energy


def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> tuple[float, Dict[int, np.ndarray]]:
    """Legacy entry point."""

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)

    energy = compute_energy_and_gradient_array(
        mesh, global_params, param_resolver,
        positions=positions, index_map=idx_map, grad_arr=grad_arr
    )

    # Convert back to dict if needed (though bending gradient is 0 for now)
    grad = {vid: grad_arr[row] for vid, row in idx_map.items() if np.any(grad_arr[row])}

    return energy, grad
