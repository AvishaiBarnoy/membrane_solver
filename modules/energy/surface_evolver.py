# modules/energy/surface_evolver.py

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
    Surface Tension module using the Shoelace Vector Area formulation.
    Fully vectorized for triangle meshes, handles polygons via loop.
    """
    total_energy = 0.0
    st_default = global_params.get("surface_tension", 1.0)

    # 1. Fast path: Vectorized for triangle meshes
    # Even though it's 'Evolver' style, for triangles the math is simple.
    tri_rows, tri_facets = mesh.triangle_row_cache()
    if tri_rows is not None:
        v0 = positions[tri_rows[:, 0]]
        v1 = positions[tri_rows[:, 1]]
        v2 = positions[tri_rows[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        n = _fast_cross(e1, e2)
        area_doubled = np.linalg.norm(n, axis=1)
        mask = area_doubled > 1e-12

        areas = 0.5 * area_doubled[mask]
        n_hat = n[mask] / area_doubled[mask][:, None]

        gammas = np.array(
            [
                param_resolver.get(mesh.facets[fid], "surface_tension") or st_default
                for fid in np.array(tri_facets)[mask]
            ],
            dtype=float,
        )

        total_energy += float(np.dot(gammas, areas))

        # Gradients
        g0 = 0.5 * gammas[:, None] * _fast_cross(v1[mask] - v2[mask], n_hat)
        g1 = 0.5 * gammas[:, None] * _fast_cross(v2[mask] - v0[mask], n_hat)
        g2 = 0.5 * gammas[:, None] * _fast_cross(v0[mask] - v1[mask], n_hat)

        np.add.at(grad_arr, tri_rows[mask, 0], g0)
        np.add.at(grad_arr, tri_rows[mask, 1], g1)
        np.add.at(grad_arr, tri_rows[mask, 2], g2)

    # 2. Polygons path (only for facets that aren't in the triangle cache)
    tri_facets_set = set(tri_facets) if tri_facets is not None else set()

    for fid, facet in mesh.facets.items():
        if fid in tri_facets_set:
            continue

        gamma = param_resolver.get(facet, "surface_tension") or st_default
        if gamma == 0:
            continue

        v_ids = mesh.facet_vertex_loops.get(fid)
        if v_ids is None or len(v_ids) < 3:
            continue

        rows = [index_map[vid] for vid in v_ids]
        v_pos = positions[rows]

        v_curr = v_pos
        v_next = np.roll(v_pos, -1, axis=0)
        cross_sum = _fast_cross(v_curr, v_next).sum(axis=0)
        area_doubled = np.linalg.norm(cross_sum)

        if area_doubled < 1e-12:
            continue

        total_energy += 0.5 * gamma * area_doubled
        n_hat = cross_sum / area_doubled

        v_prev = np.roll(v_pos, 1, axis=0)
        grads = 0.5 * gamma * _fast_cross(v_prev - v_next, n_hat)

        for i, row in enumerate(rows):
            grad_arr[row] += grads[i]

    return float(total_energy)


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
    grad = {
        vid: grad_arr[row].copy()
        for vid, row in index_map.items()
        if np.any(grad_arr[row])
    }
    return E, grad
