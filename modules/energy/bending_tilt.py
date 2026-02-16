"""Helfrich bending energy with Kozlov–Hamm tilt-splay coupling.

This module implements the coupled term

    E = 1/2 * ∫ κ (2H - c0 + div(t))^2 dA

where ``t`` is a 3D tilt vector field stored per vertex (tangent-projected),
and ``div(t)`` is a discrete surface divergence.

Notes
-----
- This module follows the existing bending discretization (cotan Laplacian,
  mixed Voronoi areas) for the curvature part.
- ``div(t)`` is computed from piecewise-linear (P1) elements on each triangle.
  Energy is assembled per triangle using the same effective-area weights as the
  bending module; a per-vertex effective divergence average is used for the
  approximate shape gradient.
- The returned *shape* gradient currently treats ``div(t)`` as constant with
  respect to vertex positions (i.e. it ignores the derivative of the discrete
  divergence operator w.r.t. geometry). The *tilt* gradient is exact for the
  chosen discretization.
- Combine with ``modules.energy.tilt`` to add a separate tilt-magnitude cost.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.bending_derivatives import grad_triangle_area
from geometry.curvature import compute_curvature_data
from geometry.entities import Mesh
from geometry.tilt_operators import (
    compute_divergence_from_basis,
    p1_triangle_divergence,
)

# Reuse the validated bending implementation helpers.
from modules.energy.bending import (  # noqa: PLC0415
    _apply_beltrami_laplacian,
    _cached_cotan_gradients,
    _compute_effective_areas,
    _energy_model,
    _grad_cotan,
    _gradient_mode,
    _per_vertex_params,
    _vertex_normals,
)
from modules.energy.scatter import scatter_triangle_scalar_to_vertices

USES_TILT = True


def _total_energy(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts: np.ndarray,
) -> float:
    """Energy-only helper for finite-difference debugging."""
    mesh.build_position_cache()
    k_vecs, vertex_areas_vor, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )
    if tri_rows.size == 0:
        return 0.0

    div_tri, _, _, _, _ = p1_triangle_divergence(
        positions=positions, tilts=tilts, tri_rows=tri_rows
    )

    _, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh, positions, tri_rows, weights, index_map
    )
    safe_areas_vor = np.maximum(vertex_areas_vor, 1e-12)

    model = _energy_model(global_params)
    if model != "helfrich":
        model = "helfrich"
    kappa_arr, c0_arr = _per_vertex_params(mesh, global_params, model=model)

    k_mag = np.linalg.norm(k_vecs, axis=1)
    H_vor = k_mag / (2.0 * safe_areas_vor)

    boundary_vids = mesh.boundary_vertex_ids
    is_interior = np.ones(len(mesh.vertex_ids), dtype=bool)
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        is_interior[boundary_rows] = False

    base_term = (2.0 * H_vor) - c0_arr
    base_term[~is_interior] = 0.0

    term_tri = base_term[tri_rows] + div_tri[:, None]
    va_eff = np.stack([va0_eff, va1_eff, va2_eff], axis=1)
    kappa_tri = kappa_arr[tri_rows]

    return float(0.5 * np.sum(kappa_tri * term_tri**2 * va_eff))


def _finite_difference_gradient_shape(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Energy-consistent shape gradient by central differences (slow)."""
    grad = np.zeros_like(positions)
    base = positions.copy()
    for row, vid in enumerate(mesh.vertex_ids):
        if getattr(mesh.vertices[int(vid)], "fixed", False):
            continue
        for d in range(3):
            pos_plus = base.copy()
            pos_minus = base.copy()
            pos_plus[row, d] += eps
            pos_minus[row, d] -= eps
            e_plus = _total_energy(
                mesh,
                global_params,
                positions=pos_plus,
                index_map=index_map,
                tilts=tilts,
            )
            e_minus = _total_energy(
                mesh,
                global_params,
                positions=pos_minus,
                index_map=index_map,
                tilts=tilts,
            )
            grad[row, d] = (e_plus - e_minus) / (2.0 * eps)

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        grad[boundary_rows] = 0.0
    return grad


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray | None,
    ctx=None,
    tilts: np.ndarray | None = None,
    tilt_grad_arr: np.ndarray | None = None,
) -> float:
    """Compute coupled bending+tilt energy and accumulate gradients.

    Shape gradient is returned in ``grad_arr``. When ``tilt_grad_arr`` is
    provided, the (exact) tilt gradient for this discretization is accumulated
    into that array as well.
    """
    mesh.build_position_cache()
    k_vecs, vertex_areas_vor, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )

    if tri_rows.size == 0:
        return 0.0

    if tilts is None:
        tilts = mesh.tilts_view()
    else:
        tilts = np.asarray(tilts, dtype=float)
        if tilts.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts must have shape (N_vertices, 3)")

    if ctx is not None:
        area_cache, g0_cache, g1_cache, g2_cache, tri_rows_cache = (
            ctx.geometry.p1_triangle_shape_gradients(mesh, positions)
        )
    else:
        area_cache, g0_cache, g1_cache, g2_cache, tri_rows_cache = (
            mesh.p1_triangle_shape_gradient_cache(positions)
        )
    if tri_rows_cache.size and tri_rows_cache.shape[0] == tri_rows.shape[0]:
        div_tri = compute_divergence_from_basis(
            tilts=tilts, tri_rows=tri_rows, g0=g0_cache, g1=g1_cache, g2=g2_cache
        )
        g0, g1, g2 = g0_cache, g1_cache, g2_cache
        _ = area_cache
    else:
        div_tri, _, g0, g1, g2 = p1_triangle_divergence(
            positions=positions, tilts=tilts, tri_rows=tri_rows
        )

    # Step 1: Boundary area reassignment (integration weight)
    vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh, positions, tri_rows, weights, index_map
    )
    safe_areas_vor = np.maximum(vertex_areas_vor, 1e-12)

    model = _energy_model(global_params)
    if model != "helfrich":
        # Kozlov–Hamm coupling is defined for Helfrich-like models.
        model = "helfrich"
    kappa_arr, c0_arr = _per_vertex_params(mesh, global_params, model=model)

    k_mag = np.linalg.norm(k_vecs, axis=1)
    H_vor = k_mag / (2.0 * safe_areas_vor)

    boundary_vids = mesh.boundary_vertex_ids
    is_interior = np.ones(len(mesh.vertex_ids), dtype=bool)
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        is_interior[boundary_rows] = False

    ratio = np.zeros_like(vertex_areas_eff)
    mask_vor = safe_areas_vor > 1e-15
    ratio[mask_vor] = vertex_areas_eff[mask_vor] / safe_areas_vor[mask_vor]

    base_term = (2.0 * H_vor) - c0_arr
    base_term[~is_interior] = 0.0

    term_tri = base_term[tri_rows] + div_tri[:, None]
    va_eff = np.stack([va0_eff, va1_eff, va2_eff], axis=1)
    kappa_tri = kappa_arr[tri_rows]
    total_energy = float(0.5 * np.sum(kappa_tri * term_tri**2 * va_eff))

    # Shape gradient uses a per-vertex effective divergence average (treating
    # div(t) as constant w.r.t. geometry; see module docstring).
    div_eff_num = scatter_triangle_scalar_to_vertices(
        tri_rows=tri_rows,
        w0=va0_eff * div_tri,
        w1=va1_eff * div_tri,
        w2=va2_eff * div_tri,
        n_vertices=base_term.shape[0],
    )
    div_eff = np.zeros_like(base_term)
    mask_eff = vertex_areas_eff > 1e-20
    div_eff[mask_eff] = div_eff_num[mask_eff] / vertex_areas_eff[mask_eff]

    term = base_term + div_eff
    term[~is_interior] = 0.0

    if grad_arr is None:
        if tilt_grad_arr is not None:
            tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
            if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
                raise ValueError("tilt_grad_arr must have shape (N_vertices, 3)")

            dE_ddiv = np.sum(kappa_tri * term_tri * va_eff, axis=1)
            factor = dE_ddiv[:, None]

            np.add.at(tilt_grad_arr, tri_rows[:, 0], factor * g0)
            np.add.at(tilt_grad_arr, tri_rows[:, 1], factor * g1)
            np.add.at(tilt_grad_arr, tri_rows[:, 2], factor * g2)
        return float(total_energy)

    mode = _gradient_mode(global_params)
    normals = _vertex_normals(mesh, positions, tri_rows)
    K_dir = np.zeros_like(k_vecs)
    mask_k = k_mag > 1e-15
    K_dir[mask_k] = k_vecs[mask_k] / k_mag[mask_k][:, None]
    K_dir[~mask_k] = normals[~mask_k]

    scale_K = (kappa_arr * term * ratio).astype(float, copy=False)
    factor_K_vec = np.empty_like(K_dir, order="F")
    np.multiply(K_dir, scale_K[:, None], out=factor_K_vec)

    fA_eff = 0.5 * kappa_arr * term**2
    fA_vor = -2.0 * kappa_arr * term * ratio * H_vor

    if mode == "finite_difference":  # pragma: no cover - slow debugging path
        eps = float(global_params.get("bending_fd_eps", 1e-6))
        grad_arr[:] += _finite_difference_gradient_shape(
            mesh,
            global_params,
            positions=positions,
            index_map=index_map,
            tilts=tilts,
            eps=eps,
        )
    elif mode == "approx":
        grad_arr[:] -= _apply_beltrami_laplacian(weights, tri_rows, factor_K_vec)
        if boundary_vids:
            grad_arr[boundary_rows] = 0.0
    else:
        # --- Analytic gradient backpropagation (copied from bending.py) ---
        v0_idxs, v1_idxs, v2_idxs = tri_rows[:, 0], tri_rows[:, 1], tri_rows[:, 2]
        v0, v1, v2 = positions[v0_idxs], positions[v1_idxs], positions[v2_idxs]
        e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0
        c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]

        # Term 1: Variation assuming cotans constant (L constant)
        # factor_K_vec already zeroed at boundaries
        grad_linear = -_apply_beltrami_laplacian(weights, tri_rows, factor_K_vec)

        # Term 2: Variation of L (cotangents)
        fK = factor_K_vec
        dE_dc0 = -0.5 * np.einsum("ij,ij->i", fK[v1_idxs] - fK[v2_idxs], v1 - v2)
        dE_dc1 = -0.5 * np.einsum("ij,ij->i", fK[v2_idxs] - fK[v0_idxs], v2 - v0)
        dE_dc2 = -0.5 * np.einsum("ij,ij->i", fK[v0_idxs] - fK[v1_idxs], v0 - v1)

        cached = _cached_cotan_gradients(mesh, positions=positions, tri_rows=tri_rows)
        if cached is not None:
            (
                g_c0_u,
                g_c0_v,
                g_c1_u,
                g_c1_v,
                g_c2_u,
                g_c2_v,
                gc0u,
                gc0v,
                gc1u,
                gc1v,
                gc2u,
                gc2v,
            ) = cached
        else:
            u0, v0_vec = v1 - v0, v2 - v0
            g_c0_u, g_c0_v = _grad_cotan(u0, v0_vec)
            u1, v1_vec = v2 - v1, v0 - v1
            g_c1_u, g_c1_v = _grad_cotan(u1, v1_vec)
            u2, v2_vec = v0 - v2, v1 - v2
            g_c2_u, g_c2_v = _grad_cotan(u2, v2_vec)

        grad_cot = np.zeros_like(positions)
        val0, val1, val2 = dE_dc0[:, None], dE_dc1[:, None], dE_dc2[:, None]
        np.add.at(grad_cot, v1_idxs, val0 * g_c0_u)
        np.add.at(grad_cot, v2_idxs, val0 * g_c0_v)
        np.add.at(grad_cot, v0_idxs, val0 * -(g_c0_u + g_c0_v))
        np.add.at(grad_cot, v2_idxs, val1 * g_c1_u)
        np.add.at(grad_cot, v0_idxs, val1 * g_c1_v)
        np.add.at(grad_cot, v1_idxs, val1 * -(g_c1_u + g_c1_v))
        np.add.at(grad_cot, v0_idxs, val2 * g_c2_u)
        np.add.at(grad_cot, v1_idxs, val2 * g_c2_v)
        np.add.at(grad_cot, v2_idxs, val2 * -(g_c2_u + g_c2_v))

        # --- Area Gradients (Step 2: Propagate area reassignment) ---

        # Redistribution logic applies to fA_eff
        tri_is_int = is_interior[tri_rows]
        interior_counts = np.sum(tri_is_int, axis=1)

        tri_fA_eff = fA_eff[tri_rows]
        sum_fA_eff_int = np.sum(tri_fA_eff * tri_is_int, axis=1)

        avg_fA_eff = np.zeros(len(tri_rows), dtype=float)
        mask_has_int = interior_counts > 0
        avg_fA_eff[mask_has_int] = (
            sum_fA_eff_int[mask_has_int] / interior_counts[mask_has_int]
        )

        # C_eff[t, k] = tri_fA_eff[t, k] if vertex k is interior, else avg_fA_eff[t]
        C_eff = np.where(tri_is_int, tri_fA_eff, avg_fA_eff[:, None])

        # Add contribution from A_vor direct dependency (only for interior vertices)
        # C_final = C_eff + fA_vor (where fA_vor is 0 for boundary)
        tri_fA_vor = fA_vor[tri_rows]
        # fA_vor is 0 at boundaries, so we can just add tri_fA_vor to C_eff.
        # Note: For boundary vertices in C_eff, we have avg_fA_eff. We add fA_vor[bdy] which is 0. Correct.
        C = C_eff + tri_fA_vor

        grad_area = np.zeros_like(positions)

        # Now run standard Voronoi gradient logic using C
        # This applies to ALL triangles (pure interior and boundary adjacent)

        # Check for obtuse triangles
        is_obtuse = (c0 < 0) | (c1 < 0) | (c2 < 0)
        m_std = ~is_obtuse

        # Standard non-obtuse logic
        if np.any(m_std):
            c0s, c1s, c2s = c0[m_std], c1[m_std], c2[m_std]
            C0s, C1s, C2s = C[m_std, 0], C[m_std, 1], C[m_std, 2]
            e0s, e1s, e2s = e0[m_std], e1[m_std], e2[m_std]
            v0s, v1s, v2s = v0_idxs[m_std], v1_idxs[m_std], v2_idxs[m_std]

            # Length variation
            coeff = 0.25 * c1s * C0s
            np.add.at(grad_area, v0s, coeff[:, None] * e1s)
            np.add.at(grad_area, v2s, -coeff[:, None] * e1s)
            coeff = 0.25 * c2s * C0s
            np.add.at(grad_area, v1s, coeff[:, None] * e2s)
            np.add.at(grad_area, v0s, -coeff[:, None] * e2s)
            coeff = 0.25 * c2s * C1s
            np.add.at(grad_area, v1s, coeff[:, None] * e2s)
            np.add.at(grad_area, v0s, -coeff[:, None] * e2s)
            coeff = 0.25 * c0s * C1s
            np.add.at(grad_area, v2s, coeff[:, None] * e0s)
            np.add.at(grad_area, v1s, -coeff[:, None] * e0s)
            coeff = 0.25 * c0s * C2s
            np.add.at(grad_area, v2s, coeff[:, None] * e0s)
            np.add.at(grad_area, v1s, -coeff[:, None] * e0s)
            coeff = 0.25 * c1s * C2s
            np.add.at(grad_area, v0s, coeff[:, None] * e1s)
            np.add.at(grad_area, v2s, -coeff[:, None] * e1s)

            # Cotan variation
            l0sq = np.einsum("ij,ij->i", e0s, e0s)
            l1sq = np.einsum("ij,ij->i", e1s, e1s)
            l2sq = np.einsum("ij,ij->i", e2s, e2s)

            coeff_c0 = 0.125 * l0sq * (C1s + C2s)
            coeff_c1 = 0.125 * l1sq * (C0s + C2s)
            coeff_c2 = 0.125 * l2sq * (C0s + C1s)

            if cached is None:
                gc0u, gc0v = _grad_cotan(e2s, -e1s)
                gc1u, gc1v = _grad_cotan(e0s, -e2s)
                gc2u, gc2v = _grad_cotan(e1s, -e0s)
            else:
                # Cached arrays are in full-triangle order; slice for the std mask.
                gc0u, gc0v = gc0u[m_std], gc0v[m_std]
                gc1u, gc1v = gc1u[m_std], gc1v[m_std]
                gc2u, gc2v = gc2u[m_std], gc2v[m_std]

            v_c0, v_c1, v_c2 = (
                coeff_c0[:, None],
                coeff_c1[:, None],
                coeff_c2[:, None],
            )
            np.add.at(grad_area, v1s, v_c0 * gc0u)
            np.add.at(grad_area, v2s, v_c0 * gc0v)
            np.add.at(grad_area, v0s, v_c0 * -(gc0u + gc0v))
            np.add.at(grad_area, v2s, v_c1 * gc1u)
            np.add.at(grad_area, v0s, v_c1 * gc1v)
            np.add.at(grad_area, v1s, v_c1 * -(gc1u + gc1v))
            np.add.at(grad_area, v0s, v_c2 * gc2u)
            np.add.at(grad_area, v1s, v_c2 * gc2v)
            np.add.at(grad_area, v2s, v_c2 * -(gc2u + gc2v))

        # Obtuse logic
        if np.any(is_obtuse):
            for i, m_sub in enumerate([(c0 < 0), (c1 < 0), (c2 < 0)]):
                m_do = m_sub & is_obtuse
                if np.any(m_do):
                    v0o, v1o, v2o = v0_idxs[m_do], v1_idxs[m_do], v2_idxs[m_do]
                    # Gradient of Triangle Area
                    gT_u, gT_v = grad_triangle_area(
                        positions[v1o] - positions[v0o],
                        positions[v2o] - positions[v0o],
                    )

                    C0o, C1o, C2o = C[m_do, 0], C[m_do, 1], C[m_do, 2]
                    if i == 0:  # Angle at v0 obtuse
                        # A0 = T/2, A1 = T/4, A2 = T/4
                        factor = (0.5 * C0o + 0.25 * C1o + 0.25 * C2o)[:, None]
                    elif i == 1:  # Angle at v1 obtuse
                        factor = (0.5 * C1o + 0.25 * C0o + 0.25 * C2o)[:, None]
                    else:  # Angle at v2 obtuse
                        factor = (0.5 * C2o + 0.25 * C0o + 0.25 * C1o)[:, None]

                    np.add.at(grad_area, v1o, factor * gT_u)
                    np.add.at(grad_area, v2o, factor * gT_v)
                    np.add.at(grad_area, v0o, factor * -(gT_u + gT_v))

        grad_arr[:] += grad_linear + grad_cot + grad_area

    # --- Tilt gradient (exact for discretization) ---
    if tilt_grad_arr is not None:
        tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
        if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_grad_arr must have shape (N_vertices, 3)")

        dE_ddiv = np.sum(kappa_tri * term_tri * va_eff, axis=1)
        factor = dE_ddiv[:, None]

        np.add.at(tilt_grad_arr, tri_rows[:, 0], factor * g0)
        np.add.at(tilt_grad_arr, tri_rows[:, 1], factor * g1)
        np.add.at(tilt_grad_arr, tri_rows[:, 2], factor * g2)

    return total_energy


def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> (
    tuple[float, Dict[int, np.ndarray]]
    | tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Dict-based wrapper returning (E, shape_grad[, tilt_grad])."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions) if compute_gradient else None
    E = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts=None,
        tilt_grad_arr=tilt_grad_arr,
    )
    if not compute_gradient:
        return float(E), {}

    shape_grad = {
        int(vid): grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }

    tilt_grad = {
        int(vid): tilt_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if tilt_grad_arr is not None and np.any(tilt_grad_arr[row])
    }
    return float(E), shape_grad, tilt_grad


__all__ = ["compute_energy_and_gradient", "compute_energy_and_gradient_array"]
