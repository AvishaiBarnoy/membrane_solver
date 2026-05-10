"""Helfrich and Willmore bending energy modules."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from geometry.curvature import compute_curvature_data
from geometry.entities import Mesh

from .bending_diagnostics import _finite_difference_gradient
from .bending_gradient import _backpropagate_bending_shape_gradient
from .bending_math import (
    _apply_beltrami_laplacian,
)
from .bending_params import (
    _energy_model,
    _gradient_mode,
    _per_vertex_params,
)
from .bending_utils import (
    _compute_effective_areas,
    _mean_curvature_vectors,
    _vertex_normals,
)

logger = logging.getLogger("membrane_solver")


def compute_total_energy(
    mesh: Mesh, global_params, positions: np.ndarray, index_map: Dict[int, int]
) -> float:
    """Compute total bending energy for the provided positions."""
    model = _energy_model(global_params)
    kappa_arr, c0_arr = _per_vertex_params(mesh, global_params, model=model)
    if float(np.max(kappa_arr)) == 0.0:
        return 0.0

    _, H, _, weights, tri_rows = _mean_curvature_vectors(mesh, positions, index_map)
    if tri_rows.size == 0:
        return 0.0

    vertex_areas_eff, _, _, _ = _compute_effective_areas(
        mesh, positions, tri_rows, weights, index_map
    )

    if model == "helfrich":
        density = 0.5 * (2.0 * H - c0_arr) ** 2
    else:
        density = H**2

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        density[boundary_rows] = 0.0

    return float(np.sum(kappa_arr * density * vertex_areas_eff))


def compute_energy_array(mesh: Mesh, global_params, positions, index_map) -> np.ndarray:
    """Compute bending energy contribution per vertex. Returns (N_verts,) array."""
    model = _energy_model(global_params)
    kappa_arr, c0_arr = _per_vertex_params(mesh, global_params, model=model)
    if float(np.max(kappa_arr)) == 0.0:
        return np.zeros(len(mesh.vertex_ids), dtype=float)

    _, H, _, weights, tri_rows = _mean_curvature_vectors(mesh, positions, index_map)
    if tri_rows.size == 0:
        return np.zeros(len(mesh.vertex_ids), dtype=float)

    vertex_areas_eff, _, _, _ = _compute_effective_areas(
        mesh, positions, tri_rows, weights, index_map
    )

    if model == "helfrich":
        density = 0.5 * (2.0 * H - c0_arr) ** 2
    else:
        density = H**2

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        density[boundary_rows] = 0.0

    return kappa_arr * density * vertex_areas_eff


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray,
) -> float:
    """Compute bending energy and accumulate analytic or FD gradient."""
    _ = param_resolver
    mesh.build_position_cache()
    k_vecs, vertex_areas_vor, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )

    if tri_rows.size == 0:
        return 0.0

    vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh, positions, tri_rows, weights, index_map
    )
    safe_areas_vor = np.maximum(vertex_areas_vor, 1e-12)

    model = _energy_model(global_params)
    kappa_arr, c0_arr = _per_vertex_params(mesh, global_params, model=model)

    k_mag = np.linalg.norm(k_vecs, axis=1)
    H_vor = k_mag / (2.0 * safe_areas_vor)

    boundary_vids = mesh.boundary_vertex_ids
    is_interior = np.ones(len(mesh.vertex_ids), dtype=bool)
    boundary_rows = []
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        is_interior[boundary_rows] = False

    ratio = np.zeros_like(vertex_areas_eff)
    mask_vor = safe_areas_vor > 1e-15
    ratio[mask_vor] = vertex_areas_eff[mask_vor] / safe_areas_vor[mask_vor]

    if model == "helfrich":
        term = (2.0 * H_vor) - c0_arr
        term[~is_interior] = 0.0
        total_energy = float(0.5 * np.sum(kappa_arr * term**2 * vertex_areas_eff))
        scale_K = (kappa_arr * term * ratio).astype(float, copy=False)
        fA_eff = 0.5 * kappa_arr * term**2
        fA_vor = -2.0 * kappa_arr * term * ratio * H_vor
    else:
        H_vor_eff = H_vor.copy()
        H_vor_eff[~is_interior] = 0.0
        total_energy = float(np.sum(kappa_arr * H_vor_eff**2 * vertex_areas_eff))
        scale_K = (kappa_arr * H_vor_eff * ratio).astype(float, copy=False)
        fA_eff = kappa_arr * H_vor_eff**2
        fA_vor = -2.0 * kappa_arr * H_vor_eff**2 * ratio

    mode = _gradient_mode(global_params)
    if mode == "finite_difference":
        eps = float(global_params.get("bending_fd_eps", 1e-6))
        grad_arr[:] += _finite_difference_gradient(
            mesh, global_params, positions, index_map, eps=eps
        )
        return total_energy

    normals = _vertex_normals(mesh, positions, tri_rows)
    K_dir = np.zeros_like(k_vecs)
    mask_k = k_mag > 1e-15
    K_dir[mask_k] = k_vecs[mask_k] / k_mag[mask_k][:, None]
    K_dir[~mask_k] = normals[~mask_k]

    factor_K_vec = np.empty_like(K_dir, order="F")
    np.multiply(K_dir, scale_K[:, None], out=factor_K_vec)

    if mode == "approx":
        grad_arr[:] -= _apply_beltrami_laplacian(weights, tri_rows, factor_K_vec)
        if boundary_rows:
            grad_arr[boundary_rows] = 0.0
        return total_energy

    _backpropagate_bending_shape_gradient(
        mesh,
        positions=positions,
        tri_rows=tri_rows,
        weights=weights,
        is_interior=is_interior,
        fA_eff=fA_eff,
        fA_vor=fA_vor,
        factor_K_vec=factor_K_vec,
        grad_arr=grad_arr,
    )

    return total_energy


def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> tuple[float, Dict[int, np.ndarray]]:
    """Dict-based wrapper for bending energy and gradients."""
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
    if not compute_gradient:
        return float(E), {}

    grad = {
        int(vid): grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }
    return float(E), grad


__all__ = ["compute_energy_and_gradient", "compute_energy_and_gradient_array"]
