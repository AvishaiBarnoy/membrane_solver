# modules/energy/bending.py

from __future__ import annotations

import logging
from typing import Dict, Literal

import numpy as np

from geometry.bending_derivatives import (
    grad_cotan,
    grad_triangle_area,
)
from geometry.curvature import compute_curvature_data
from geometry.entities import Mesh

logger = logging.getLogger("membrane_solver")


BendingEnergyModel = Literal["willmore", "helfrich"]
BendingGradientMode = Literal["approx", "finite_difference", "analytic"]


def _energy_model(global_params) -> BendingEnergyModel:
    model = str(global_params.get("bending_energy_model", "willmore") or "willmore")
    model = model.lower().strip()
    return "helfrich" if model == "helfrich" else "willmore"


def _gradient_mode(global_params) -> BendingGradientMode:
    mode = str(global_params.get("bending_gradient_mode", "approx") or "approx")
    mode = mode.lower().strip()
    if mode in {"fd", "finite_difference"}:
        return "finite_difference"
    if mode == "analytic":
        return "analytic"
    return "approx"


def _spontaneous_curvature(global_params) -> float:
    val = global_params.get("spontaneous_curvature")
    if val is None:
        val = global_params.get("intrinsic_curvature", 0.0)
    return float(val or 0.0)


def _per_vertex_params(
    mesh: Mesh,
    global_params,
    *,
    model: BendingEnergyModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-vertex (kappa, c0) arrays in vertex-row order.

    Allows local overrides via `vertex.options`:
    - `bending_modulus`
    - `spontaneous_curvature` (alias: `intrinsic_curvature`)
    """
    n = len(mesh.vertex_ids)
    kappa_default = float(global_params.get("bending_modulus", 0.0) or 0.0)
    kappa = np.full(n, kappa_default, dtype=float)

    if model == "helfrich":
        c0_default = _spontaneous_curvature(global_params)
        c0 = np.full(n, c0_default, dtype=float)
    else:
        c0 = np.zeros(n, dtype=float)

    for row, vid in enumerate(mesh.vertex_ids):
        v = mesh.vertices.get(int(vid))
        if v is None:
            continue
        opts = getattr(v, "options", None) or {}
        if "bending_modulus" in opts:
            try:
                kappa[row] = float(opts["bending_modulus"])
            except (TypeError, ValueError):
                pass
        if model == "helfrich":
            if "spontaneous_curvature" in opts:
                try:
                    c0[row] = float(opts["spontaneous_curvature"])
                except (TypeError, ValueError):
                    pass
            elif "intrinsic_curvature" in opts:
                try:
                    c0[row] = float(opts["intrinsic_curvature"])
                except (TypeError, ValueError):
                    pass

    return kappa, c0


def _vertex_normals(
    mesh: Mesh, positions: np.ndarray, tri_rows: np.ndarray
) -> np.ndarray:
    normals = np.zeros((len(mesh.vertex_ids), 3), dtype=float)
    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]
    tri_normals = np.cross(v1 - v0, v2 - v0)
    np.add.at(normals, tri_rows[:, 0], tri_normals)
    np.add.at(normals, tri_rows[:, 1], tri_normals)
    np.add.at(normals, tri_rows[:, 2], tri_normals)
    nrm = np.linalg.norm(normals, axis=1)
    mask = nrm > 1e-15
    normals[mask] /= nrm[mask, None]
    return normals


def _mean_curvature_vectors(
    mesh: Mesh, positions: np.ndarray, index_map: Dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Hn, |Hn|, A, weights, tri_rows).

    We use the magnitude ``|Hn|`` for scalar curvature to avoid dependence on
    global facet orientation, which is not guaranteed to be consistent in all
    generated meshes.
    """
    mesh.build_position_cache()
    k_vecs, vertex_areas, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )
    if tri_rows.size == 0:
        n = len(mesh.vertex_ids)
        return (
            np.zeros((n, 3), dtype=float),
            np.zeros(n, dtype=float),
            np.zeros(n, dtype=float),
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=int),
        )

    safe_areas = np.maximum(vertex_areas, 1e-12)
    h_vecs = k_vecs / (2.0 * safe_areas[:, None])  # Hn
    h_mag = np.linalg.norm(h_vecs, axis=1)
    return h_vecs, h_mag, vertex_areas, weights, tri_rows


def compute_total_energy(
    mesh: Mesh, global_params, positions: np.ndarray, index_map: Dict[int, int]
) -> float:
    model = _energy_model(global_params)
    kappa_arr, c0_arr = _per_vertex_params(mesh, global_params, model=model)
    if float(np.max(kappa_arr)) == 0.0:
        return 0.0

    _, H, A, _, tri_rows = _mean_curvature_vectors(mesh, positions, index_map)
    if tri_rows.size == 0:
        return 0.0

    if model == "helfrich":
        density = 0.5 * (2.0 * H - c0_arr) ** 2
    else:
        density = H**2

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        density[boundary_rows] = 0.0

    return float(np.sum(kappa_arr * density * A))


def compute_energy_array(mesh, global_params, positions, index_map) -> np.ndarray:
    """Compute energy contribution per vertex. Returns (N_verts,) array."""
    model = _energy_model(global_params)
    kappa_arr, c0_arr = _per_vertex_params(mesh, global_params, model=model)
    if float(np.max(kappa_arr)) == 0.0:
        return np.zeros(len(mesh.vertex_ids), dtype=float)

    _, H, A, _, tri_rows = _mean_curvature_vectors(mesh, positions, index_map)
    if tri_rows.size == 0:
        return np.zeros(len(mesh.vertex_ids), dtype=float)

    if model == "helfrich":
        density = 0.5 * (2.0 * H - c0_arr) ** 2
    else:
        density = H**2

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        density[boundary_rows] = 0.0

    return kappa_arr * density * A


def _apply_beltrami_laplacian(
    weights: np.ndarray, tri_rows: np.ndarray, field: np.ndarray
) -> np.ndarray:
    """Apply the discrete Laplace-Beltrami operator (based on cotangent weights)."""
    c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]
    v0, v1, v2 = tri_rows[:, 0], tri_rows[:, 1], tri_rows[:, 2]
    f0, f1, f2 = field[v0], field[v1], field[v2]
    out = np.zeros_like(field)
    np.add.at(out, v0, 0.5 * (c1[:, None] * (f0 - f2) + c2[:, None] * (f0 - f1)))
    np.add.at(out, v1, 0.5 * (c2[:, None] * (f1 - f0) + c0[:, None] * (f1 - f2)))
    np.add.at(out, v2, 0.5 * (c0[:, None] * (f2 - f1) + c1[:, None] * (f2 - f0)))
    return out


def _finite_difference_gradient(
    mesh: Mesh,
    global_params,
    positions: np.ndarray,
    index_map: Dict[int, int],
    *,
    eps: float,
) -> np.ndarray:
    """Energy-consistent gradient by central differences (slow)."""
    grad = np.zeros_like(positions)
    base = positions.copy()
    mesh.build_position_cache()
    for row, vid in enumerate(mesh.vertex_ids):
        if getattr(mesh.vertices[int(vid)], "fixed", False):
            continue
        for d in range(3):
            pos_plus = base.copy()
            pos_minus = base.copy()
            pos_plus[row, d] += eps
            pos_minus[row, d] -= eps
            e_plus = compute_total_energy(mesh, global_params, pos_plus, index_map)
            e_minus = compute_total_energy(mesh, global_params, pos_minus, index_map)
            grad[row, d] = (e_plus - e_minus) / (2.0 * eps)

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        grad[boundary_rows] = 0.0
    return grad


def _laplacian_on_vertex_field(
    weights: np.ndarray, tri_rows: np.ndarray, field: np.ndarray
) -> np.ndarray:
    c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]
    v0, v1, v2 = tri_rows[:, 0], tri_rows[:, 1], tri_rows[:, 2]
    f0, f1, f2 = field[v0], field[v1], field[v2]
    out = np.zeros_like(field)
    np.add.at(out, v0, 0.5 * (c1[:, None] * (f0 - f2) + c2[:, None] * (f0 - f1)))
    np.add.at(out, v1, 0.5 * (c2[:, None] * (f1 - f0) + c0[:, None] * (f1 - f2)))
    np.add.at(out, v2, 0.5 * (c0[:, None] * (f2 - f1) + c1[:, None] * (f2 - f0)))
    return out


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
    # Compute basic curvature data
    # Note: We need to cache/retrieve this efficiently.
    # compute_curvature_data returns (K_vecs, A_mixed, weights, tri_rows)
    # K_vecs is integrated mean curvature (0.5 * sum (cot a + cot b)(xi - xj))

    mesh.build_position_cache()
    k_vecs, vertex_areas, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )

    # Calculate Energy
    safe_areas = np.maximum(vertex_areas, 1e-12)
    # H_scalar = |K| / (2A)
    # E = kappa * sum( (H_scalar)^2 * A ) = kappa * sum( |K|^2 / (4A) )
    # For Helfrich: E = kappa/2 * sum( (2H - c0)^2 * A )
    #                 = kappa/2 * sum( ( |K|/A - c0 )^2 * A )

    model = _energy_model(global_params)
    kappa_arr, c0_arr = _per_vertex_params(mesh, global_params, model=model)

    # Precompute per-vertex energy and terms for gradient
    # We need gradients of K_i and A_i w.r.t positions.

    # E_i = energy density at vertex i
    k_mag = np.linalg.norm(k_vecs, axis=1)

    if model == "helfrich":
        # E = sum E_i, E_i = 0.5 * kappa * ( k_mag/A - c0 )^2 * A
        term = (k_mag / safe_areas) - c0_arr
        total_energy = float(0.5 * np.sum(kappa_arr * term**2 * safe_areas))

        # dE/dx = sum_i dE_i/dx
        # dE_i = 0.5 * kappa * [ 2 * (k/A - c0) * d(k/A - c0) * A + (k/A - c0)^2 * dA ]
        #      = 0.5 * kappa * [ 2 * term * ( (dk * A - k * dA)/A^2 ) * A + term^2 * dA ]
        #      = 0.5 * kappa * [ 2 * term * (dk/A - k/A^2 dA) * A + term^2 * dA ]
        #      = kappa * term * dk - kappa * term * (k/A) * dA + 0.5 * kappa * term^2 * dA
        #      = kappa * term * dk + (0.5 * kappa * term^2 - kappa * term * k/A) * dA

        # d(|K|) = K/|K| . dK
        # dk = (K / |K|) . dK

        # Let factor_K = kappa * term * (K / |K|)
        # Let factor_A = 0.5 * kappa * term^2 - kappa * term * (k_mag / safe_areas)

        # Use vertex normals instead of K_unit to be robust for flat regions
        # K points inwards, so we want normals aligned with K.
        normals = _vertex_normals(mesh, positions, tri_rows)
        # Ensure normals point in the same hemisphere as k_vecs
        # (or just use normals and accept the sign convention of the mesh)
        # H_scalar in compute_total_energy uses |K|, so it's always positive.
        # We should match that by using the projection of K onto n.

        # However, for a flat sheet, which way is "inwards"?
        # The perturbation breaks symmetry. We should use the normal.

        K_dir = np.zeros_like(k_vecs)
        mask = k_mag > 1e-15
        K_dir[mask] = k_vecs[mask] / k_mag[mask][:, None]
        K_dir[~mask] = normals[~mask]  # Fallback to vertex normal

        factor_K_vec = kappa_arr[:, None] * term[:, None] * K_dir  # (N, 3)
        factor_A = 0.5 * kappa_arr * term**2 - kappa_arr * term * (k_mag / safe_areas)

    else:
        # Willmore: E_i = kappa * (H^2) * A = kappa * (|K|/2A)^2 * A = kappa * |K|^2 / (4A)
        total_energy = float(np.sum(kappa_arr * (k_mag**2 / (4.0 * safe_areas))))

        # dE_i = (kappa/4) * [ (2 |K| d|K|) / A - (|K|^2 / A^2) dA ]
        #      = (kappa/2A) * (K . dK) - (kappa |K|^2 / 4A^2) dA

        # factor_K_vec = (kappa / 2A) * K
        # factor_A = - kappa * |K|^2 / (4A^2)

        factor_K_vec = (kappa_arr / (2.0 * safe_areas))[:, None] * k_vecs
        factor_A = -kappa_arr * k_mag**2 / (4.0 * safe_areas**2)

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        # Match the energy convention used elsewhere: ignore boundary vertices.
        if boundary_rows:
            factor_K_vec[boundary_rows] = 0.0
            factor_A[boundary_rows] = 0.0
            # Also remove their energy contribution so callers see E=0 on open planar patches.
            if model == "helfrich":
                term[boundary_rows] = 0.0
                total_energy = float(0.5 * np.sum(kappa_arr * term**2 * safe_areas))
            else:
                k_mag_b = k_mag.copy()
                k_mag_b[boundary_rows] = 0.0
                total_energy = float(
                    np.sum(kappa_arr * (k_mag_b**2 / (4.0 * safe_areas)))
                )

    # Now verify if user wants FD/approx/analytic
    mode = _gradient_mode(global_params)
    if mode == "finite_difference":
        eps = float(global_params.get("bending_fd_eps", 1e-6))
        grad_arr[:] += _finite_difference_gradient(
            mesh, global_params, positions, index_map, eps=eps
        )
        return total_energy

    if mode == "approx":
        # Fast, stable approximation: apply cotan Laplacian to the mean-curvature
        # normal field (or to the deviation from target curvature in Helfrich).
        h_vecs = k_vecs / (2.0 * safe_areas[:, None])  # Hn
        if model == "helfrich":
            # Use deviation from preferred curvature magnitude along local direction.
            h_norms = np.linalg.norm(h_vecs, axis=1)
            n_hats = np.zeros_like(h_vecs)
            mask_n = h_norms > 1e-12
            n_hats[mask_n] = h_vecs[mask_n] / h_norms[mask_n][:, None]
            target_h_vecs = (c0_arr / 2.0)[:, None] * n_hats
            diff_h_vecs = h_vecs - target_h_vecs
            grad_arr[:] += _apply_beltrami_laplacian(
                weights, tri_rows, kappa_arr[:, None] * diff_h_vecs
            )
        else:
            grad_arr[:] += _apply_beltrami_laplacian(
                weights, tri_rows, kappa_arr[:, None] * h_vecs
            )
        return total_energy

    # --- Analytic gradient backpropagation ---
    # K_i = 0.5 * sum_{j in N(i)} (cot a_ij + cot b_ij) (x_i - x_j)
    # This term depends on positions x_i, x_j, and the vertices defining angles a_ij, b_ij.

    # Strategy: Iterate over all triangles to accumulate gradients.
    # Each triangle contributes to cotans of its edges and edge vectors.

    v0_idxs = tri_rows[:, 0]
    v1_idxs = tri_rows[:, 1]
    v2_idxs = tri_rows[:, 2]

    v0 = positions[v0_idxs]
    v1 = positions[v1_idxs]
    v2 = positions[v2_idxs]

    # Edges
    e0 = v2 - v1  # Opp v0
    e1 = v0 - v2  # Opp v1
    e2 = v1 - v0  # Opp v2

    # Cotangents c0, c1, c2 (at vertices 0, 1, 2)
    # We need gradients of these cotangents w.r.t v0, v1, v2

    # Re-compute cotans to get gradients
    # c0 = cot(angle at v0) = cot(-e2, e1) ? No, angle between edges emanating from v0.
    # Vectors from v0 are (v1-v0) = e2 and (v2-v0) = -e1.
    # cot0 = dot(e2, -e1) / |e2 x -e1|

    # Let's use standard edge vectors:
    # u = v1-v0, v = v2-v0.  e2=u, -e1=v.
    # cot0 = cot(u, v)

    # For triangle (v0, v1, v2):
    # cot0 depends on v0, v1, v2.

    # We need to distribute `factor_K_vec` to the vertices.
    # The sum for K_i involves neighbours.
    # It's easier to iterate triangles and add contributions to K_i terms.
    # K_i += 0.5 * cot_term * (x_i - x_j)

    # Let's verify the K formula again.
    # K_i = 0.5 * sum_j (cot_alpha_j + cot_beta_j) (x_i - x_j)
    # This can be rewritten as K_i = -0.5 * sum_j (cot...)(x_j - x_i)
    # Or in terms of the Laplacian matrix L: K = L * X.
    # K_i = sum_j L_ij x_j.
    # So d(K_i)/dx_k is L_ik * I (if L was constant) + sum_j d(L_ij)/dx_k * x_j.
    # L is NOT constant (depends on cotans).

    # Term 1: Variation assuming cotans constant (L constant)
    # grad_E += sum_i (dE/dK_i) * L_ik
    # This is equivalent to L * (factor_K_vec).
    # force_1 = _apply_beltrami_laplacian(weights, tri_rows, factor_K_vec)
    # But wait, K = L X. So X^T L^T factor. L is symmetric.
    # So yes, applying L to factor_K_vec gives the gradient contribution from the linear part.

    grad_linear = np.zeros_like(positions)
    # Apply L to factor_K_vec. L is defined by weights.
    c0 = weights[:, 0]
    c1 = weights[:, 1]
    c2 = weights[:, 2]

    # L_ij * f_j part:
    # For edge (0,1) with weight c2: terms are c2 * (f0 - f1) added to 0, c2 * (f1 - f0) added to 1.
    # This matches _apply_beltrami_laplacian logic.
    # Note: our discrete integrated curvature vector is assembled as
    # K_i += 0.5 * sum_j w_ij (x_j - x_i), i.e. K = -L X for the cotan Laplacian L.
    # Therefore the "linear" backprop term uses -L^T = -L.
    grad_linear = -_apply_beltrami_laplacian(weights, tri_rows, factor_K_vec)

    # Term 2: Variation of L (cotangents)
    # E = ... K ...
    # dE/dx = sum_i (dE/dK_i) . dK_i/dx
    # dK_i = sum_j d(w_ij)/dx * (x_i - x_j) + w_ij * d(x_i - x_j)/dx
    # The second part is the linear term we just handled.
    # The first part depends on d(cot)/dx.

    # For each triangle, we have 3 cotans: c0, c1, c2.
    # c0 is on edge (1,2) opposite v0. It contributes to L_12 (weight between v1, v2).
    # K term associated with edge (1,2) is: 0.5 * c0 * (x_1 - x_2) added to K_1, and (x_2 - x_1) to K_2?
    # No, K_i = 0.5 * sum (cot) (xi - xj).
    # For edge 1-2, weight is c0 (from this tri) + c_opp (from other tri).
    # The contribution of c0 is to the interaction between 1 and 2.
    # It adds 0.5 * c0 * (x1 - x2) to K_1
    # It adds 0.5 * c0 * (x2 - x1) to K_2

    # So variation of c0 affects K_1 and K_2.
    # dE_c0 = (dE/dK_1) . (0.5 * (x1 - x2)) + (dE/dK_2) . (0.5 * (x2 - x1))
    #       = 0.5 * (factor_K_1 - factor_K_2) . (x1 - x2)

    # Scalar sensitivity of Energy w.r.t cotan c0:
    # dE/dc0 = 0.5 * dot( factor_K_1 - factor_K_2, v1 - v2 )

    fK = factor_K_vec  # Alias

    # With K_i using (x_j - x_i), the cotan sensitivity picks up a minus sign
    # compared to the (x_i - x_j) convention.
    dE_dc0 = -0.5 * np.einsum("ij,ij->i", fK[v1_idxs] - fK[v2_idxs], v1 - v2)
    dE_dc1 = -0.5 * np.einsum("ij,ij->i", fK[v2_idxs] - fK[v0_idxs], v2 - v0)
    dE_dc2 = -0.5 * np.einsum("ij,ij->i", fK[v0_idxs] - fK[v1_idxs], v0 - v1)

    # Now we need gradients of c0, c1, c2 w.r.t v0, v1, v2.
    # c0 = cot(angle at v0).
    # Vector u = v1-v0, v = v2-v0.
    # c0 = cot(u, v).
    # grad_c0_v1, grad_c0_v2 = grad_cotan(u, v)
    # grad_c0_v0 = -(grad_c0_v1 + grad_c0_v2) (translation invariance)

    # c0
    u0 = v1 - v0
    v0_vec = v2 - v0
    g_c0_u, g_c0_v = grad_cotan(u0, v0_vec)

    # c1 (angle at v1)
    u1 = v2 - v1
    v1_vec = v0 - v1
    g_c1_u, g_c1_v = grad_cotan(u1, v1_vec)

    # c2 (angle at v2)
    u2 = v0 - v2
    v2_vec = v1 - v2
    g_c2_u, g_c2_v = grad_cotan(u2, v2_vec)

    # Accumulate gradients from cotans
    grad_cot = np.zeros_like(positions)

    # Contribution from c0 (depends on v0, v1, v2)
    # v1 term: g_c0_u
    # v2 term: g_c0_v
    # v0 term: -(g_c0_u + g_c0_v)

    # We multiply by dE_dc0 (scalar) and add to vertices
    val0 = dE_dc0[:, None]
    val1 = dE_dc1[:, None]
    val2 = dE_dc2[:, None]

    # c0 terms
    np.add.at(grad_cot, v1_idxs, val0 * g_c0_u)
    np.add.at(grad_cot, v2_idxs, val0 * g_c0_v)
    np.add.at(grad_cot, v0_idxs, val0 * -(g_c0_u + g_c0_v))

    # c1 terms
    np.add.at(grad_cot, v2_idxs, val1 * g_c1_u)
    np.add.at(grad_cot, v0_idxs, val1 * g_c1_v)
    np.add.at(grad_cot, v1_idxs, val1 * -(g_c1_u + g_c1_v))

    # c2 terms
    np.add.at(grad_cot, v0_idxs, val2 * g_c2_u)
    np.add.at(grad_cot, v1_idxs, val2 * g_c2_v)
    np.add.at(grad_cot, v2_idxs, val2 * -(g_c2_u + g_c2_v))

    # --- Area Gradients (Mixed Voronoi) ---
    fA = factor_A

    # Identify obtuse triangles
    is_obtuse_0 = c0 < 0
    is_obtuse_1 = c1 < 0
    is_obtuse_2 = c2 < 0
    is_obtuse = is_obtuse_0 | is_obtuse_1 | is_obtuse_2

    # --- Case 1: Non-obtuse (Standard Voronoi) ---
    mask_std = ~is_obtuse

    # Precompute squared lengths for the gradient terms (only needed for std)
    l0_sq = np.einsum("ij,ij->i", e0, e0)
    l1_sq = np.einsum("ij,ij->i", e1, e1)
    l2_sq = np.einsum("ij,ij->i", e2, e2)

    # factors for non-obtuse
    m_std = mask_std
    if np.any(m_std):
        # We need to filter arrays by mask
        c0_m = c0[m_std]
        c1_m = c1[m_std]
        c2_m = c2[m_std]

        l0_sq_m = l0_sq[m_std]
        l1_sq_m = l1_sq[m_std]
        l2_sq_m = l2_sq[m_std]

        fA0_m = fA[v0_idxs[m_std]]
        fA1_m = fA[v1_idxs[m_std]]
        fA2_m = fA[v2_idxs[m_std]]

        # 1. Gradients from |e|^2 variation
        # Mixed Voronoi areas (non-obtuse):
        #   A0 += (l1^2*c1 + l2^2*c2)/8
        #   A1 += (l2^2*c2 + l0^2*c0)/8
        #   A2 += (l0^2*c0 + l1^2*c1)/8
        #
        # Energy contribution uses factor_A at the owning vertex only: fA[vk] * dAk.
        grad_area_len = np.zeros_like(positions)

        # A0: l1^2*c1 term (e1 = v0-v2)
        coeff = 0.25 * c1_m * fA0_m
        np.add.at(grad_area_len, v0_idxs[m_std], coeff[:, None] * e1[m_std])
        np.add.at(grad_area_len, v2_idxs[m_std], -coeff[:, None] * e1[m_std])

        # A0: l2^2*c2 term (e2 = v1-v0)
        coeff = 0.25 * c2_m * fA0_m
        np.add.at(grad_area_len, v1_idxs[m_std], coeff[:, None] * e2[m_std])
        np.add.at(grad_area_len, v0_idxs[m_std], -coeff[:, None] * e2[m_std])

        # A1: l2^2*c2 term (e2 = v1-v0)
        coeff = 0.25 * c2_m * fA1_m
        np.add.at(grad_area_len, v1_idxs[m_std], coeff[:, None] * e2[m_std])
        np.add.at(grad_area_len, v0_idxs[m_std], -coeff[:, None] * e2[m_std])

        # A1: l0^2*c0 term (e0 = v2-v1)
        coeff = 0.25 * c0_m * fA1_m
        np.add.at(grad_area_len, v2_idxs[m_std], coeff[:, None] * e0[m_std])
        np.add.at(grad_area_len, v1_idxs[m_std], -coeff[:, None] * e0[m_std])

        # A2: l0^2*c0 term (e0 = v2-v1)
        coeff = 0.25 * c0_m * fA2_m
        np.add.at(grad_area_len, v2_idxs[m_std], coeff[:, None] * e0[m_std])
        np.add.at(grad_area_len, v1_idxs[m_std], -coeff[:, None] * e0[m_std])

        # A2: l1^2*c1 term (e1 = v0-v2)
        coeff = 0.25 * c1_m * fA2_m
        np.add.at(grad_area_len, v0_idxs[m_std], coeff[:, None] * e1[m_std])
        np.add.at(grad_area_len, v2_idxs[m_std], -coeff[:, None] * e1[m_std])

        # 2. Gradients from cotan variation
        # 2. Gradients from cotan variation.
        # Each cotan contributes to two vertex areas in the triangle:
        #   c0 appears in A1 and A2; c1 appears in A0 and A2; c2 appears in A0 and A1.
        coeff_c0 = 0.125 * l0_sq_m * (fA1_m + fA2_m)
        coeff_c1 = 0.125 * l1_sq_m * (fA0_m + fA2_m)
        coeff_c2 = 0.125 * l2_sq_m * (fA0_m + fA1_m)

        grad_area_cot = np.zeros_like(positions)

        # Need re-computation of cot gradients for masked subset?
        # Or just use precomputed and mask them.
        # u0, v0_vec were computed for ALL triangles.
        # g_c0_u, g_c0_v are full size.

        # c0 terms
        v_c0 = coeff_c0[:, None]
        np.add.at(grad_area_cot, v1_idxs[m_std], v_c0 * g_c0_u[m_std])
        np.add.at(grad_area_cot, v2_idxs[m_std], v_c0 * g_c0_v[m_std])
        np.add.at(
            grad_area_cot, v0_idxs[m_std], v_c0 * -(g_c0_u[m_std] + g_c0_v[m_std])
        )

        # c1 terms
        v_c1 = coeff_c1[:, None]
        np.add.at(grad_area_cot, v2_idxs[m_std], v_c1 * g_c1_u[m_std])
        np.add.at(grad_area_cot, v0_idxs[m_std], v_c1 * g_c1_v[m_std])
        np.add.at(
            grad_area_cot, v1_idxs[m_std], v_c1 * -(g_c1_u[m_std] + g_c1_v[m_std])
        )

        # c2 terms
        v_c2 = coeff_c2[:, None]
        np.add.at(grad_area_cot, v0_idxs[m_std], v_c2 * g_c2_u[m_std])
        np.add.at(grad_area_cot, v1_idxs[m_std], v_c2 * g_c2_v[m_std])
        np.add.at(
            grad_area_cot, v2_idxs[m_std], v_c2 * -(g_c2_u[m_std] + g_c2_v[m_std])
        )

        grad_arr[:] += grad_area_len + grad_area_cot

    # --- Case 2: Obtuse triangles ---
    if np.any(is_obtuse):
        # We need gradients of Triangle Area w.r.t v0, v1, v2
        # T depends on u=v1-v0, v=v2-v0.
        # gT_u, gT_v = grad_triangle_area(u, v)
        # gT_v1 = gT_u
        # gT_v2 = gT_v
        # gT_v0 = -(gT_u + gT_v)

        # Triangle edges (v1-v0, v2-v0) correspond to u, v in our definition earlier.
        # u = v1 - v0, v = v2 - v0
        # g_c0_u used u=v1-v0, v=v2-v0. So we can reuse those vectors.
        # We need new gradients for area though.

        u_vec = v1 - v0
        v_vec = v2 - v0
        gT_u, gT_v = grad_triangle_area(u_vec, v_vec)

        # Masking for each sub-case

        # Case 2a: Obtuse at v0 (c0 < 0)
        # A0 = T/2, A1 = T/4, A2 = T/4
        # Energy variation: fA[v0]*dA0 + fA[v1]*dA1 + fA[v2]*dA2
        # = (0.5*fA[v0] + 0.25*fA[v1] + 0.25*fA[v2]) * dT
        m0 = is_obtuse_0
        if np.any(m0):
            factor = (
                0.5 * fA[v0_idxs[m0]] + 0.25 * fA[v1_idxs[m0]] + 0.25 * fA[v2_idxs[m0]]
            )[:, None]
            np.add.at(grad_arr, v1_idxs[m0], factor * gT_u[m0])
            np.add.at(grad_arr, v2_idxs[m0], factor * gT_v[m0])
            np.add.at(grad_arr, v0_idxs[m0], factor * -(gT_u[m0] + gT_v[m0]))

        # Case 2b: Obtuse at v1 (c1 < 0)
        # A1 = T/2, A0 = T/4, A2 = T/4
        m1 = is_obtuse_1
        if np.any(m1):
            factor = (
                0.5 * fA[v1_idxs[m1]] + 0.25 * fA[v0_idxs[m1]] + 0.25 * fA[v2_idxs[m1]]
            )[:, None]
            np.add.at(grad_arr, v1_idxs[m1], factor * gT_u[m1])
            np.add.at(grad_arr, v2_idxs[m1], factor * gT_v[m1])
            np.add.at(grad_arr, v0_idxs[m1], factor * -(gT_u[m1] + gT_v[m1]))

        # Case 2c: Obtuse at v2 (c2 < 0)
        # A2 = T/2, A0 = T/4, A1 = T/4
        m2 = is_obtuse_2
        if np.any(m2):
            factor = (
                0.5 * fA[v2_idxs[m2]] + 0.25 * fA[v0_idxs[m2]] + 0.25 * fA[v1_idxs[m2]]
            )[:, None]
            np.add.at(grad_arr, v1_idxs[m2], factor * gT_u[m2])
            np.add.at(grad_arr, v2_idxs[m2], factor * gT_v[m2])
            np.add.at(grad_arr, v0_idxs[m2], factor * -(gT_u[m2] + gT_v[m2]))

    grad_arr[:] += grad_linear + grad_cot

    return total_energy


def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> tuple[float, Dict[int, np.ndarray]]:
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
