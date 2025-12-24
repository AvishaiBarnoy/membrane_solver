# modules/energy/bending.py

from typing import Dict

import numpy as np

from geometry.curvature import compute_curvature_data
from geometry.entities import Mesh
from runtime.logging_config import setup_logging

logger = setup_logging("membrane_solver.log")


def compute_energy_array(mesh, global_params, positions, index_map) -> np.ndarray:
    """Compute energy contribution per vertex. Returns (N_verts,) array."""
    k_vecs, vertex_areas, _, _ = compute_curvature_data(mesh, positions, index_map)

    # 1. Compute Mean Curvature H = K / (2 * A)
    safe_areas = np.maximum(vertex_areas, 1e-12)
    h_vecs = k_vecs / (2.0 * safe_areas[:, None])

    # 2. Handle Intrinsic (Spontaneous) Curvature C0
    # Helfrich energy density: kappa * (H - C0)^2
    c0 = global_params.get("intrinsic_curvature", 0.0)
    h_mags = np.linalg.norm(h_vecs, axis=1)
    # Energy terms: (H - C0)^2
    h_diff_sq = (h_mags - c0) ** 2

    # 3. Strict Boundary Filtering: No energy for boundary vertices
    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        h_diff_sq[boundary_rows] = 0.0

    energy_per_vertex = h_diff_sq * vertex_areas
    kappa = global_params.get("bending_modulus", 1.0)
    return energy_per_vertex * kappa


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
    Numerically Consistent Bending Energy and Gradient with Spontaneous Curvature.
    Uses a vectorized bi-Laplacian force that drives H toward C0.
    """
    # 1. Total Energy
    energies = compute_energy_array(mesh, global_params, positions, index_map)
    total_energy = float(np.sum(energies))

    # 2. Pass 2: Vectorized bi-Laplacian (The correct force direction)
    k_vecs, vertex_areas, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )
    safe_areas = np.maximum(vertex_areas, 1e-12)
    h_vecs = k_vecs / (2.0 * safe_areas[:, None])

    c0 = global_params.get("intrinsic_curvature", 0.0)

    # Preferred curvature vector: C0 * n_hat
    n_hats = np.zeros_like(h_vecs)
    h_norms = np.linalg.norm(h_vecs, axis=1)
    mask = h_norms > 1e-12
    n_hats[mask] = h_vecs[mask] / h_norms[mask][:, None]
    target_h_vecs = c0 * n_hats

    # Signal to smooth: the deviation from preferred curvature
    diff_h_vecs = h_vecs - target_h_vecs

    c0_w, c1_w, c2_w = weights[:, 0], weights[:, 1], weights[:, 2]
    v0_dh, v1_dh, v2_dh = (
        diff_h_vecs[tri_rows[:, 0]],
        diff_h_vecs[tri_rows[:, 1]],
        diff_h_vecs[tri_rows[:, 2]],
    )

    lh_vecs = np.zeros((len(positions), 3), dtype=float)
    np.add.at(
        lh_vecs,
        tri_rows[:, 0],
        0.5 * (c1_w[:, None] * (v0_dh - v2_dh) + c2_w[:, None] * (v0_dh - v1_dh)),
    )
    np.add.at(
        lh_vecs,
        tri_rows[:, 1],
        0.5 * (c2_w[:, None] * (v1_dh - v0_dh) + c0_w[:, None] * (v1_dh - v2_dh)),
    )
    np.add.at(
        lh_vecs,
        tri_rows[:, 2],
        0.5 * (c0_w[:, None] * (v2_dh - v1_dh) + c1_w[:, None] * (v2_dh - v0_dh)),
    )

    # 3. Filter boundary forces
    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = np.array(
            [index_map[vid] for vid in boundary_vids if vid in index_map], dtype=int
        )
        lh_vecs[boundary_rows] = 0.0

    kappa = global_params.get("bending_modulus", 1.0)
    # grad = kappa * L(H - C0)
    grad_arr += kappa * lh_vecs

    return total_energy


def compute_energy_and_gradient(mesh, global_params, param_resolver, **kwargs):
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