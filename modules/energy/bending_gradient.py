"""Analytic shape-gradient backpropagation for Helfrich/Willmore bending energy."""

from __future__ import annotations

import numpy as np

from geometry.bending_derivatives import grad_triangle_area
from geometry.entities import Mesh

from .bending_math import (
    _apply_beltrami_laplacian,
    _cached_cotan_gradients,
    _grad_cotan,
)


def _backpropagate_bending_shape_gradient(
    mesh: Mesh,
    *,
    positions: np.ndarray,
    tri_rows: np.ndarray,
    weights: np.ndarray,
    is_interior: np.ndarray,
    fA_eff: np.ndarray,
    fA_vor: np.ndarray,
    factor_K_vec: np.ndarray,
    grad_arr: np.ndarray,
) -> None:
    """Propagate energy gradients back to vertex positions."""
    v0_idxs, v1_idxs, v2_idxs = tri_rows[:, 0], tri_rows[:, 1], tri_rows[:, 2]
    v0, v1, v2 = positions[v0_idxs], positions[v1_idxs], positions[v2_idxs]
    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0
    c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]

    # Term 1: Variation assuming cotans constant (L constant)
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

    # redistribution logic applies to fA_eff
    tri_is_int = is_interior[tri_rows]
    interior_counts = np.sum(tri_is_int, axis=1)

    tri_fA_eff = fA_eff[tri_rows]
    sum_fA_eff_int = np.sum(tri_fA_eff * tri_is_int, axis=1)

    avg_fA_eff = np.zeros(len(tri_rows), dtype=float)
    mask_has_int = interior_counts > 0
    avg_fA_eff[mask_has_int] = (
        sum_fA_eff_int[mask_has_int] / interior_counts[mask_has_int]
    )

    C_eff = np.where(tri_is_int, tri_fA_eff, avg_fA_eff[:, None])
    tri_fA_vor = fA_vor[tri_rows]
    C = C_eff + tri_fA_vor

    grad_area = np.zeros_like(positions)
    is_obtuse = (c0 < 0) | (c1 < 0) | (c2 < 0)
    m_std = ~is_obtuse

    if np.any(m_std):
        c0s, c1s, c2s = c0[m_std], c1[m_std], c2[m_std]
        C0s, C1s, C2s = C[m_std, 0], C[m_std, 1], C[m_std, 2]
        e0s, e1s, e2s = e0[m_std], e1[m_std], e2[m_std]
        v0s, v1s, v2s = v0_idxs[m_std], v1_idxs[m_std], v2_idxs[m_std]

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
            gc0u, gc0v = gc0u[m_std], gc0v[m_std]
            gc1u, gc1v = gc1u[m_std], gc1v[m_std]
            gc2u, gc2v = gc2u[m_std], gc2v[m_std]

        v_c0, v_c1, v_c2 = coeff_c0[:, None], coeff_c1[:, None], coeff_c2[:, None]
        np.add.at(grad_area, v1s, v_c0 * gc0u)
        np.add.at(grad_area, v2s, v_c0 * gc0v)
        np.add.at(grad_area, v0s, v_c0 * -(gc0u + gc0v))
        np.add.at(grad_area, v2s, v_c1 * gc1u)
        np.add.at(grad_area, v0s, v_c1 * gc1v)
        np.add.at(grad_area, v1s, v_c1 * -(gc1u + gc1v))
        np.add.at(grad_area, v0s, v_c2 * gc2u)
        np.add.at(grad_area, v1s, v_c2 * gc2v)
        np.add.at(grad_area, v2s, v_c2 * -(gc2u + gc2v))

    if np.any(is_obtuse):
        for i, m_sub in enumerate([(c0 < 0), (c1 < 0), (c2 < 0)]):
            m_do = m_sub & is_obtuse
            if np.any(m_do):
                v0o, v1o, v2o = v0_idxs[m_do], v1_idxs[m_do], v2_idxs[m_do]
                gT_u, gT_v = grad_triangle_area(
                    positions[v1o] - positions[v0o], positions[v2o] - positions[v0o]
                )

                C0o, C1o, C2o = C[m_do, 0], C[m_do, 1], C[m_do, 2]
                if i == 0:
                    factor = (0.5 * C0o + 0.25 * C1o + 0.25 * C2o)[:, None]
                elif i == 1:
                    factor = (0.5 * C1o + 0.25 * C0o + 0.25 * C2o)[:, None]
                else:
                    factor = (0.5 * C2o + 0.25 * C0o + 0.25 * C1o)[:, None]

                np.add.at(grad_area, v1o, factor * gT_u)
                np.add.at(grad_area, v2o, factor * gT_v)
                np.add.at(grad_area, v0o, factor * -(gT_u + gT_v))

    grad_arr[:] += grad_linear + grad_cot + grad_area
