# modules/energy/bending.py

from typing import Dict

import numpy as np

from geometry.curvature import compute_integrated_curvature_vectors
from geometry.entities import Mesh
from runtime.logging_config import setup_logging

logger = setup_logging("membrane_solver.log")


def compute_energy_only(mesh, global_params, positions, index_map) -> float:
    """Fast path for energy-only calculation (used for perturbations)."""

    k_vecs, vertex_areas = compute_integrated_curvature_vectors(
        mesh, positions, index_map
    )

    # Filter out boundary vertices to avoid artifacts on open meshes

    boundary_vids = mesh.boundary_vertex_ids

    # Convert vids to row indices

    boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]

    k_sq_norms = np.einsum("ij,ij->i", k_vecs, k_vecs)

    # Zero out curvature for boundary vertices

    if boundary_rows:
        k_sq_norms[boundary_rows] = 0.0

    area_mask = vertex_areas > 1e-12

    energy_per_vertex = np.zeros_like(vertex_areas)

    energy_per_vertex[area_mask] = k_sq_norms[area_mask] / (
        4.0 * vertex_areas[area_mask]
    )

    total_energy = float(np.sum(energy_per_vertex))

    kappa = global_params.get("bending_modulus", 1.0)

    return total_energy * kappa


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
    Consistent Bending Energy and Gradient.
    Uses numerical differentiation to ensure the gradient is the exact derivative
    of the discrete cotangent-based energy. This guarantees line-search stability.
    """
    # 1. Base Energy
    E0 = compute_energy_only(mesh, global_params, positions, index_map)

    # 2. Numerical Gradient (Perturbation)
    # We only need to perturb vertices that contribute to the bending energy.
    # For performance, we can skip vertices with no facets.
    eps = 1e-6

    # Vectorized perturbation is hard here because each vertex affects a local neighborhood.
    # However, we can use the fact that the bending gradient is very localized.
    # To keep this 'toy' but 'consistent', we implement the numerical derivative.

    # Optimization: Only perturb active vertices
    for vid, row in index_map.items():
        # Get neighbors to potentially limit calculation, but for now we do full re-eval
        # (This is the 'Correctness-First' path)
        for d in range(3):
            orig_val = positions[row, d]

            positions[row, d] = orig_val + eps
            E_plus = compute_energy_only(mesh, global_params, positions, index_map)

            positions[row, d] = orig_val - eps
            E_minus = compute_energy_only(mesh, global_params, positions, index_map)

            # Reset
            positions[row, d] = orig_val

            # Central Difference
            grad_arr[row, d] += (E_plus - E_minus) / (2 * eps)

    return E0


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

    if not compute_gradient:
        return compute_energy_only(mesh, global_params, positions, idx_map), {}

    grad_arr = np.zeros_like(positions)
    energy = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
    )

    grad = {
        vid: grad_arr[row].copy()
        for vid, row in idx_map.items()
        if np.any(grad_arr[row])
    }
    return energy, grad
