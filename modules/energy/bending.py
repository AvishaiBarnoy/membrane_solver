# modules/energy/bending.py

from typing import Dict

import numpy as np

from geometry.curvature import compute_curvature_data
from geometry.entities import Mesh
from runtime.logging_config import setup_logging

logger = setup_logging("membrane_solver.log")


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
    Fast Analytical Bending Energy and Gradient.
    Uses a two-pass vectorized Cotangent Laplacian to compute the bi-Laplacian force.
    """
    # 1. Pass 1: Compute integrated curvature vectors K and dual areas A
    k_vecs, vertex_areas, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )

    if tri_rows.size == 0:
        return 0.0

    n_verts = len(mesh.vertex_ids)

    # Filter out boundary vertices to avoid artifacts
    boundary_vids = mesh.boundary_vertex_ids
    boundary_rows = np.array(
        [index_map[vid] for vid in boundary_vids if vid in index_map], dtype=int
    )

    # 2. Compute Mean Curvature H = K / (2 * A)
    # Avoid division by zero
    safe_areas = np.maximum(vertex_areas, 1e-12)
    h_vecs = k_vecs / (2.0 * safe_areas[:, None])

    # Energy E = sum ||K||^2 / 4A = sum H^2 * A
    h_sq_norms = np.einsum("ij,ij->i", h_vecs, h_vecs)

    # Ignore boundary contributions
    if boundary_rows.size > 0:
        h_sq_norms[boundary_rows] = 0.0

    total_energy = float(np.sum(h_sq_norms * vertex_areas))
    kappa = global_params.get("bending_modulus", 1.0)
    total_energy *= kappa

    # 3. Pass 2: Compute Laplacian of Curvature Signal (bi-Laplacian approximation)
    # L(H)_i = sum_j w_ij (H_i - H_j)
    # This represents the 'bending force' that smoothes curvature variation.

    # We use the cotangent weights from pass 1
    c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]

    v0_h = h_vecs[tri_rows[:, 0]]
    v1_h = h_vecs[tri_rows[:, 1]]
    v2_h = h_vecs[tri_rows[:, 2]]

    # Laplacian components for each triangle
    # For edge v1-v2 (weight c0), contrib to L0 is c0 * (H0 - H1) + c0 * (H0 - H2) ... no.
    # Standard: L_i = 0.5 * sum_j (cot alpha + cot beta) * (H_i - H_j)

    lh_vecs = np.zeros((n_verts, 3), dtype=float)

    # Contributions to L_i from triangle (0,1,2)
    # L0 contrib: 0.5 * [ c1 * (H0 - H2) + c2 * (H0 - H1) ]
    np.add.at(
        lh_vecs,
        tri_rows[:, 0],
        0.5 * (c1[:, None] * (v0_h - v2_h) + c2[:, None] * (v0_h - v1_h)),
    )
    np.add.at(
        lh_vecs,
        tri_rows[:, 1],
        0.5 * (c2[:, None] * (v1_h - v0_h) + c0[:, None] * (v1_h - v2_h)),
    )
    np.add.at(
        lh_vecs,
        tri_rows[:, 2],
        0.5 * (c0[:, None] * (v2_h - v1_h) + c1[:, None] * (v2_h - v0_h)),
    )

    # 4. Final Force: F = -kappa * Laplacian(H)
    # The gradient grad_arr is dE/dx = -F = kappa * Laplacian(H)
    if boundary_rows.size > 0:
        lh_vecs[boundary_rows] = 0.0

    grad_arr += kappa * lh_vecs

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
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
    )

    grad = {vid: grad_arr[row] for vid, row in idx_map.items() if np.any(grad_arr[row])}
    return energy, grad
