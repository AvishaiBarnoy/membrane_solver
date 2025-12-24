# modules/energy/bending_evolver.py

from typing import Dict

import numpy as np

from geometry.entities import Mesh, _fast_cross
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
    Toy Bending Module using Ken Brakke's Surface Evolver logic (Variation of Area).

    Energy E = sum_i ( ||grad_i A||^2 / (4 * dual_area_i) )
    where grad_i A is the gradient of the total area with respect to vertex i.
    """
    n_verts = len(mesh.vertex_ids)

    # 1. Initialize arrays for accumulation
    area_grads = np.zeros((n_verts, 3), dtype=float)
    vertex_areas = np.zeros(n_verts, dtype=float)

    # 2. Iterate over all facets to accumulate area gradients and dual areas
    for fid, facet in mesh.facets.items():
        v_ids = mesh.facet_vertex_loops.get(fid)
        if v_ids is None or len(v_ids) < 3:
            continue

        rows = [index_map[vid] for vid in v_ids]
        v_pos = positions[rows]

        # Shoelace vector area
        v_curr = v_pos
        v_next = np.roll(v_pos, -1, axis=0)
        cross_sum = _fast_cross(v_curr, v_next).sum(axis=0)
        area_doubled = np.linalg.norm(cross_sum)

        if area_doubled < 1e-12:
            continue

        area = 0.5 * area_doubled
        n_hat = cross_sum / area_doubled

        # Area gradient at each vertex: 0.5 * (v_prev - v_next) x n_hat
        v_prev = np.roll(v_pos, 1, axis=0)
        diff = v_prev - v_next
        grads = 0.5 * _fast_cross(diff, n_hat)

        # Accumulate Area Gradients and dual areas (Barycentric)
        for i, row in enumerate(rows):
            area_grads[row] += grads[i]
            vertex_areas[row] += area / len(rows)

    # 3. Compute Energy: sum (||grad A||^2 / (4 * A_dual))
    grad_sq_norms = np.einsum("ij,ij->i", area_grads, area_grads)
    area_mask = vertex_areas > 1e-12

    total_energy = float(
        np.sum(grad_sq_norms[area_mask] / (4.0 * vertex_areas[area_mask]))
    )

    kappa = global_params.get("bending_modulus", 1.0)
    total_energy *= kappa

    # 4. Gradient (Force) - Mean Curvature Flow
    # Curvature vector K_i approx -0.5 * grad_i A / A_dual
    # We push in direction of K_i to minimize squared curvature.
    # Force = - dE/dx. We set grad_arr += dE/dx

    # Simple MC Flow force:
    force = (kappa / 2.0) * area_grads
    grad_arr -= force

    return total_energy


def compute_energy_and_gradient(mesh, global_params, param_resolver, **kwargs):
    """Legacy entry point."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    E = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
    )
    return E, {}
