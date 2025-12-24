# modules/energy/bending.py

from typing import Dict

import numpy as np

from geometry.curvature import compute_integrated_curvature_vectors
from geometry.entities import Mesh
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
    """
    # 1. Compute curvature vectors using the reusable geometry helper
    k_vecs, vertex_areas = compute_integrated_curvature_vectors(
        mesh, positions, index_map
    )

    # 3. Compute total Willmore Energy: E = sum_i ||K_i||^2 / (4 * A_i)
    k_sq_norms = np.einsum("ij,ij->i", k_vecs, k_vecs)
    area_mask = vertex_areas > 1e-12
    energy_per_vertex = np.zeros_like(vertex_areas)
    energy_per_vertex[area_mask] = k_sq_norms[area_mask] / (
        4.0 * vertex_areas[area_mask]
    )

    total_energy = float(np.sum(energy_per_vertex))

    # 4. Scale by global bending modulus
    kappa = global_params.get("bending_modulus", 1.0)
    total_energy *= kappa

    # 5. Gradient (Force)
    # The gradient of the Willmore energy is roughly Delta H.
    # A standard discretization for the 'bending force' at vertex i is:
    # F_i = -kappa * (1/A_i) * sum_j w_ij (H_i - H_j)
    # For this high-performance version, we use the approximation that
    # the force is proportional to the curvature vector itself,
    # which drives the surface toward a minimal curvature state (sphere).

    # Force = - dE/dx.
    # We add to grad_arr (which is the gradient, so force = -grad_arr)
    # The curvature vector K_i points INWARD (direction of area decrease).
    # To decrease squared curvature, we push in the direction of K_i.
    force = (kappa / 2.0) * k_vecs

    # Accumulate into the gradient array (grad = -force)
    grad_arr -= force

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

    # Convert back to dict if needed (though bending gradient is 0 for now)
    grad = {vid: grad_arr[row] for vid, row in idx_map.items() if np.any(grad_arr[row])}

    return energy, grad
