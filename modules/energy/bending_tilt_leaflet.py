"""Shared helpers for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.bending_derivatives import grad_triangle_area
from geometry.entities import Mesh
from geometry.tilt_operators import (
    _resolve_transport_model,
    compute_divergence_from_basis,
    p1_triangle_divergence,
)
from modules.energy.bending import (  # noqa: PLC0415
    _apply_beltrami_laplacian,
    _cached_cotan_gradients,
    _compute_effective_areas,
    _energy_model,
    _grad_cotan,
    _gradient_mode,
    _vertex_normals,
)
from modules.energy.scatter import scatter_triangle_scalar_to_vertices

from .bt_divergence import (
    _inner_bending_tilt_dE_ddiv,
    _inner_recovered_divergence,
    _inner_recovered_divergence_pullback,
)
from .bt_params import (
    _assume_J0_center_xy,
    _assume_J0_presets,
    _assume_J0_radius_max,
    _per_vertex_params_leaflet,
    _use_inner_recovered_divergence,
    _use_stage_a_inner_shape_cross_suppression,
    _use_stage_a_outer_grad_linear_transition_operator,
)
from .bt_payload import (
    _leaflet_static_tilt_payload,
    _leaflet_triangle_payload,
)
from .bt_selection import (
    _apply_inner_divergence_update_mode,
    _base_term_region_zero_rows,
    _collect_group_rows,
    _collect_preset_rows,
    _interior_mask_leaflet,
    _shared_rim_support_transition_triangle_mask,
)
from .bt_transition import (
    _apply_transition_aware_beltrami_laplacian,
)
from .bt_utils import _accumulate_leaflet_tilt_gradient


def _total_energy_leaflet(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts: np.ndarray,
    kappa_key: str,
    cache_tag: str,
    div_sign: float,
) -> float:
    """Energy-only helper for finite-difference debugging."""
    payload = _leaflet_triangle_payload(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
        cache_tag=cache_tag,
    )
    k_vecs = np.asarray(payload["k_vecs"], dtype=float)
    vertex_areas_vor = np.asarray(payload["vertex_areas_vor"], dtype=float)
    weights = np.asarray(payload["weights"], dtype=float)
    tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
    tri_area = payload.get("tri_area")
    if tri_rows.size == 0:
        return 0.0

    transport_model = _resolve_transport_model(
        global_params.get("tilt_transport_model", "ambient_v1")
        if global_params is not None
        else "ambient_v1"
    )
    g0 = payload["g0"]
    g1 = payload["g1"]
    g2 = payload["g2"]
    if g0 is not None and g1 is not None and g2 is not None:
        div_tri = compute_divergence_from_basis(
            mesh=mesh,
            tilts=tilts,
            tri_rows=tri_rows,
            g0=np.asarray(g0, dtype=float),
            g1=np.asarray(g1, dtype=float),
            g2=np.asarray(g2, dtype=float),
            positions=positions,
            transport_model=transport_model,
        )
    else:
        div_tri, _, _, _, _ = p1_triangle_divergence(
            mesh=mesh,
            positions=positions,
            tilts=tilts,
            tri_rows=tri_rows,
            transport_model=transport_model,
        )
    div_term = float(div_sign) * div_tri
    div_term = _apply_inner_divergence_update_mode(
        mesh,
        global_params,
        positions=positions,
        tri_rows=tri_rows,
        cache_tag=cache_tag,
        div_term=div_term,
    )
    use_recovered_div = _use_inner_recovered_divergence(
        global_params, cache_tag=cache_tag
    )
    if use_recovered_div:
        if tri_area is None:
            tri_area = mesh.p1_triangle_shape_gradient_cache(positions)[0]
        tri_area = np.asarray(tri_area, dtype=float)
        div_eval_tri, _, _ = _inner_recovered_divergence(
            global_params=global_params,
            cache_tag=cache_tag,
            tri_rows=tri_rows,
            tri_area=tri_area,
            div_tri=div_term,
            n_vertices=len(mesh.vertex_ids),
            scratch_tag=f"btl_{cache_tag}",
        )
    else:
        div_eval_tri = div_term

    _, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        compute_vertex_areas=False,
    )
    static_payload = _leaflet_static_tilt_payload(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
        k_vecs=k_vecs,
        vertex_areas_vor=vertex_areas_vor,
        tri_rows=tri_rows,
        kappa_key=kappa_key,
        cache_tag=cache_tag,
    )
    base_tri = np.asarray(static_payload["base_tri"], dtype=float)
    term_tri = base_tri + div_eval_tri[:, None]
    va_eff = np.stack([va0_eff, va1_eff, va2_eff], axis=1)
    kappa_tri = np.asarray(static_payload["kappa_tri"], dtype=float)

    return float(0.5 * np.sum(kappa_tri * term_tri**2 * va_eff))


def _finite_difference_gradient_shape_leaflet(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts: np.ndarray,
    kappa_key: str,
    cache_tag: str,
    div_sign: float,
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
            e_plus = _total_energy_leaflet(
                mesh,
                global_params,
                positions=pos_plus,
                index_map=index_map,
                tilts=tilts,
                kappa_key=kappa_key,
                cache_tag=cache_tag,
                div_sign=div_sign,
            )
            e_minus = _total_energy_leaflet(
                mesh,
                global_params,
                positions=pos_minus,
                index_map=index_map,
                tilts=tilts,
                kappa_key=kappa_key,
                cache_tag=cache_tag,
                div_sign=div_sign,
            )
            grad[row, d] = (e_plus - e_minus) / (2.0 * eps)
    return grad


def compute_energy_and_gradient_array_leaflet(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray | None,
    ctx=None,
    tilts: np.ndarray,
    tilt_grad_arr: np.ndarray | None,
    kappa_key: str,
    cache_tag: str,
    div_sign: float = 1.0,
) -> float:
    """Compute coupled bending+tilt energy and accumulate gradients."""
    _ = param_resolver
    payload = _leaflet_triangle_payload(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
        cache_tag=cache_tag,
        ctx=ctx,
    )
    k_vecs = np.asarray(payload["k_vecs"], dtype=float)
    vertex_areas_vor = np.asarray(payload["vertex_areas_vor"], dtype=float)
    tri_rows_full = np.asarray(payload["tri_rows_full"], dtype=np.int32)
    weights = np.asarray(payload["weights"], dtype=float)
    tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
    tri_keep = np.asarray(payload["tri_keep"], dtype=bool)
    tri_area = payload.get("tri_area")

    if tri_rows_full.size == 0 or tri_rows.size == 0:
        return 0.0

    tilts = np.asarray(tilts, dtype=float)
    if tilts.shape != (len(mesh.vertex_ids), 3):
        raise ValueError("tilts must have shape (N_vertices, 3)")

    transport_model = _resolve_transport_model(
        global_params.get("tilt_transport_model", "ambient_v1")
        if global_params is not None
        else "ambient_v1"
    )

    g0_use = payload["g0"]
    g1_use = payload["g1"]
    g2_use = payload["g2"]
    if g0_use is not None and g1_use is not None and g2_use is not None:
        div_tri = compute_divergence_from_basis(
            mesh=mesh,
            tilts=tilts,
            tri_rows=tri_rows,
            g0=np.asarray(g0_use, dtype=float),
            g1=np.asarray(g1_use, dtype=float),
            g2=np.asarray(g2_use, dtype=float),
            positions=positions,
            transport_model=transport_model,
        )
        g0 = np.asarray(g0_use, dtype=float)
        g1 = np.asarray(g1_use, dtype=float)
        g2 = np.asarray(g2_use, dtype=float)
    else:
        div_tri, _, g0, g1, g2 = p1_triangle_divergence(
            mesh=mesh,
            positions=positions,
            tilts=tilts,
            tri_rows=tri_rows,
            transport_model=transport_model,
        )
    div_term = float(div_sign) * div_tri
    div_term = _apply_inner_divergence_update_mode(
        mesh,
        global_params,
        positions=positions,
        tri_rows=tri_rows,
        cache_tag=cache_tag,
        div_term=div_term,
    )
    use_recovered_div = _use_inner_recovered_divergence(
        global_params, cache_tag=cache_tag
    )
    if use_recovered_div:
        if tri_area is None:
            tri_area = mesh.p1_triangle_shape_gradient_cache(positions)[0]
        tri_area = np.asarray(tri_area, dtype=float)
        div_eval_tri, _, div_eval_vertex_area = _inner_recovered_divergence(
            global_params=global_params,
            cache_tag=cache_tag,
            tri_rows=tri_rows,
            tri_area=tri_area,
            div_tri=div_term,
            n_vertices=len(mesh.vertex_ids),
            ctx=ctx,
            scratch_tag=f"btl_{cache_tag}",
        )
    else:
        div_eval_tri = div_term
        div_eval_vertex_area = None

    vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token=f"bending_tilt_leaflet_{cache_tag}",
        compute_vertex_areas=grad_arr is not None,
    )
    if grad_arr is None:
        static_payload = _leaflet_static_tilt_payload(
            mesh,
            global_params,
            positions=positions,
            index_map=index_map,
            k_vecs=k_vecs,
            vertex_areas_vor=vertex_areas_vor,
            tri_rows=tri_rows,
            kappa_key=kappa_key,
            cache_tag=cache_tag,
        )
        base_tri = np.asarray(static_payload["base_tri"], dtype=float)
        kappa_tri = np.asarray(static_payload["kappa_tri"], dtype=float)
        if ctx is not None:
            term_tri = ctx.scratch_array(
                f"btl_{cache_tag}_tilt_only_term_tri",
                shape=base_tri.shape,
                dtype=base_tri.dtype,
            )
            np.copyto(term_tri, base_tri)
            term_tri += div_eval_tri[:, None]
        else:
            term_tri = base_tri + div_eval_tri[:, None]
        total_energy = float(
            0.5
            * np.sum(
                (kappa_tri[:, 0] * term_tri[:, 0] ** 2 * va0_eff)
                + (kappa_tri[:, 1] * term_tri[:, 1] ** 2 * va1_eff)
                + (kappa_tri[:, 2] * term_tri[:, 2] ** 2 * va2_eff)
            )
        )

        if tilt_grad_arr is not None:
            tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
            if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
                raise ValueError("tilt_grad_arr must have shape (N_vertices, 3)")

            dE_ddiv_base, mode_stats = _inner_bending_tilt_dE_ddiv(
                mesh=mesh,
                global_params=global_params,
                cache_tag=cache_tag,
                kappa_tri=kappa_tri,
                base_tri=base_tri,
                div_term=div_eval_tri,
                va0_eff=va0_eff,
                va1_eff=va1_eff,
                va2_eff=va2_eff,
            )
            if str(cache_tag) == "in":
                setattr(mesh, "_last_bending_tilt_in_update_mode_stats", mode_stats)
            if use_recovered_div:
                dE_ddiv = _inner_recovered_divergence_pullback(
                    global_params=global_params,
                    cache_tag=cache_tag,
                    tri_rows=tri_rows,
                    tri_area=tri_area,
                    coeff_div_eval=float(div_sign) * dE_ddiv_base,
                    v_area=div_eval_vertex_area,
                    ctx=ctx,
                    scratch_tag=f"btl_{cache_tag}",
                )
            else:
                dE_ddiv = float(div_sign) * dE_ddiv_base
            factor = dE_ddiv[:, None]
            _accumulate_leaflet_tilt_gradient(
                tilt_grad_arr,
                tri_rows,
                factor,
                g0,
                g1,
                g2,
                ctx=ctx,
                scratch_tag=f"btl_{cache_tag}_tilt_only_scaled",
            )
        return float(total_energy)

    safe_areas_vor = np.maximum(vertex_areas_vor, 1e-12)

    model = _energy_model(global_params)
    if model != "helfrich":
        model = "helfrich"
    kappa_arr, c0_arr = _per_vertex_params_leaflet(
        mesh, global_params, model=model, kappa_key=kappa_key, cache_tag=cache_tag
    )

    k_mag = np.linalg.norm(k_vecs, axis=1)
    H_vor = k_mag / (2.0 * safe_areas_vor)

    is_interior = _interior_mask_leaflet(
        mesh, global_params, cache_tag=cache_tag, index_map=index_map
    )

    base_term = (2.0 * H_vor) - c0_arr
    base_term[~is_interior] = 0.0
    presets = _assume_J0_presets(global_params, cache_tag=cache_tag)
    if presets:
        radius_max = _assume_J0_radius_max(global_params, cache_tag=cache_tag)
        center_xy = _assume_J0_center_xy(global_params)
        rows = _collect_preset_rows(
            mesh,
            presets=presets,
            cache_tag=cache_tag,
            index_map=index_map,
            radius_max=radius_max,
            center_xy=center_xy,
        )
        if rows.size:
            base_term[rows] = 0.0
    region_rows = _base_term_region_zero_rows(
        mesh,
        global_params,
        cache_tag=cache_tag,
        index_map=index_map,
    )
    if region_rows.size:
        base_term[region_rows] = 0.0

    base_tri = base_term[tri_rows]
    term_tri = base_tri + div_eval_tri[:, None]
    kappa_tri = kappa_arr[tri_rows]
    total_energy = float(
        0.5
        * np.sum(
            (kappa_tri[:, 0] * term_tri[:, 0] ** 2 * va0_eff)
            + (kappa_tri[:, 1] * term_tri[:, 1] ** 2 * va1_eff)
            + (kappa_tri[:, 2] * term_tri[:, 2] ** 2 * va2_eff)
        )
    )

    if ctx is not None:
        ratio = ctx.scratch_array(
            f"btl_{cache_tag}_ratio",
            shape=vertex_areas_eff.shape,
            dtype=vertex_areas_eff.dtype,
        )
    else:
        ratio = np.zeros_like(vertex_areas_eff)
    mask_vor = safe_areas_vor > 1e-15
    ratio[mask_vor] = vertex_areas_eff[mask_vor] / safe_areas_vor[mask_vor]

    if ctx is not None:
        div_eff_num = ctx.scratch_array(
            f"btl_{cache_tag}_div_eff_num",
            shape=base_term.shape,
            dtype=base_term.dtype,
        )
    else:
        div_eff_num = np.zeros_like(base_term)
    div_eff_source = div_eval_tri if use_recovered_div else div_term
    div_eff_num = scatter_triangle_scalar_to_vertices(
        tri_rows=tri_rows,
        w0=va0_eff * div_eff_source,
        w1=va1_eff * div_eff_source,
        w2=va2_eff * div_eff_source,
        n_vertices=base_term.shape[0],
        out=div_eff_num,
    )
    if ctx is not None:
        div_eff = ctx.scratch_array(
            f"btl_{cache_tag}_div_eff",
            shape=base_term.shape,
            dtype=base_term.dtype,
        )
    else:
        div_eff = np.zeros_like(base_term)
    mask_eff = vertex_areas_eff > 1e-20
    div_eff[mask_eff] = div_eff_num[mask_eff] / vertex_areas_eff[mask_eff]

    term = base_term + div_eff
    term[~is_interior] = 0.0

    suppress_shape_cross = _use_stage_a_inner_shape_cross_suppression(
        mesh, global_params, cache_tag=cache_tag
    )
    setattr(
        mesh,
        f"_last_bending_tilt_{cache_tag}_shape_cross_stats",
        {
            "enabled": bool(suppress_shape_cross),
            "cache_tag": str(cache_tag),
            "lane": str(global_params.get("theory_parity_lane") or "")
            if global_params is not None
            else "",
        },
    )

    mode = _gradient_mode(global_params)
    normals = _vertex_normals(mesh, positions, tri_rows)
    if ctx is not None:
        K_dir = ctx.scratch_array(
            f"btl_{cache_tag}_K_dir", shape=k_vecs.shape, dtype=k_vecs.dtype
        )
    else:
        K_dir = np.zeros_like(k_vecs)
    mask_k = k_mag > 1e-15
    K_dir[mask_k] = k_vecs[mask_k] / k_mag[mask_k][:, None]
    K_dir[~mask_k] = normals[~mask_k]

    shape_term = base_term if suppress_shape_cross else term
    scale_K = (kappa_arr * shape_term * ratio).astype(float, copy=False)
    if ctx is not None:
        factor_K_vec = ctx.scratch_array(
            f"btl_{cache_tag}_factor_K_vec", shape=K_dir.shape, dtype=K_dir.dtype
        )
    else:
        factor_K_vec = np.empty_like(K_dir, order="F")
    np.multiply(K_dir, scale_K[:, None], out=factor_K_vec)

    transition_operator_enabled = _use_stage_a_outer_grad_linear_transition_operator(
        global_params, cache_tag=cache_tag
    )
    transition_operator_stats = {
        "enabled": False,
        "mode": "off",
        "cache_tag": str(cache_tag),
        "lane": str(global_params.get("theory_parity_lane") or "")
        if global_params is not None
        else "",
    }

    if suppress_shape_cross:
        fA_eff = 0.5 * kappa_arr * (base_term**2 + div_eff**2)
        fA_vor = -2.0 * kappa_arr * base_term * ratio * H_vor
    else:
        fA_eff = 0.5 * kappa_arr * term**2
        fA_vor = -2.0 * kappa_arr * term * ratio * H_vor

    if mode == "finite_difference":  # pragma: no cover - slow debugging path
        eps = float(global_params.get("bending_fd_eps", 1e-6))
        grad_arr[:] += _finite_difference_gradient_shape_leaflet(
            mesh,
            global_params,
            positions=positions,
            index_map=index_map,
            tilts=tilts,
            kappa_key=kappa_key,
            cache_tag=cache_tag,
            div_sign=div_sign,
            eps=eps,
        )
    elif mode == "approx":
        if transition_operator_enabled:
            grad_linear_raw, transition_operator_stats = (
                _apply_transition_aware_beltrami_laplacian(
                    mesh,
                    positions=positions,
                    weights=weights,
                    tri_rows=tri_rows,
                    tri_rows_full=tri_rows_full,
                    field=factor_K_vec,
                    cache_tag=cache_tag,
                )
            )
            grad_arr[:] -= grad_linear_raw
        else:
            grad_arr[:] -= _apply_beltrami_laplacian(weights, tri_rows, factor_K_vec)
        grad_arr[~is_interior] = 0.0
    else:
        # --- Analytic gradient backpropagation (copied from bending.py) ---
        v0_idxs, v1_idxs, v2_idxs = tri_rows[:, 0], tri_rows[:, 1], tri_rows[:, 2]
        v0, v1, v2 = positions[v0_idxs], positions[v1_idxs], positions[v2_idxs]
        e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0
        c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]

        cached = _cached_cotan_gradients(
            mesh,
            positions=positions,
            tri_rows=tri_rows_full if tri_keep.size else tri_rows,
        )
        if cached is not None and tri_keep.size:
            cached = tuple(arr[tri_keep] for arr in cached)

        # Term 1: Variation assuming cotans constant (L constant)
        # factor_K_vec already zeroed at boundaries
        if transition_operator_enabled:
            grad_linear_raw, transition_operator_stats = (
                _apply_transition_aware_beltrami_laplacian(
                    mesh,
                    positions=positions,
                    weights=weights,
                    tri_rows=tri_rows,
                    tri_rows_full=tri_rows_full,
                    field=factor_K_vec,
                    cache_tag=cache_tag,
                )
            )
            grad_linear = -grad_linear_raw
        else:
            grad_linear = -_apply_beltrami_laplacian(weights, tri_rows, factor_K_vec)

        # Term 2: Variation of L (cotangents)
        fK = factor_K_vec
        dE_dc0 = -0.5 * np.einsum("ij,ij->i", fK[v1_idxs] - fK[v2_idxs], v1 - v2)
        dE_dc1 = -0.5 * np.einsum("ij,ij->i", fK[v2_idxs] - fK[v0_idxs], v2 - v0)
        dE_dc2 = -0.5 * np.einsum("ij,ij->i", fK[v0_idxs] - fK[v1_idxs], v0 - v1)

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

        if ctx is not None:
            grad_cot = ctx.scratch_array(
                f"btl_{cache_tag}_grad_cot",
                shape=positions.shape,
                dtype=positions.dtype,
            )
        else:
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

        tri_is_int = is_interior[tri_rows]
        interior_counts = np.sum(tri_is_int, axis=1)

        tri_fA_eff = fA_eff[tri_rows]
        sum_fA_eff_int = np.sum(tri_fA_eff * tri_is_int, axis=1)

        if ctx is not None:
            avg_fA_eff = ctx.scratch_array(
                f"btl_{cache_tag}_avg_fA_eff",
                shape=(len(tri_rows),),
                dtype=float,
            )
        else:
            avg_fA_eff = np.zeros(len(tri_rows), dtype=float)
        mask_has_int = interior_counts > 0
        avg_fA_eff[mask_has_int] = (
            sum_fA_eff_int[mask_has_int] / interior_counts[mask_has_int]
        )

        C_eff = np.where(tri_is_int, tri_fA_eff, avg_fA_eff[:, None])
        tri_fA_vor = fA_vor[tri_rows]
        C = C_eff + tri_fA_vor

        if ctx is not None:
            grad_area = ctx.scratch_array(
                f"btl_{cache_tag}_grad_area",
                shape=positions.shape,
                dtype=positions.dtype,
            )
        else:
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

        if np.any(is_obtuse):
            for i, m_sub in enumerate([(c0 < 0), (c1 < 0), (c2 < 0)]):
                m_do = m_sub & is_obtuse
                if np.any(m_do):
                    v0o, v1o, v2o = v0_idxs[m_do], v1_idxs[m_do], v2_idxs[m_do]
                    gT_u, gT_v = grad_triangle_area(
                        positions[v1o] - positions[v0o],
                        positions[v2o] - positions[v0o],
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

        grad_arr[:] += grad_linear
        grad_arr[:] += grad_cot
        grad_arr[:] += grad_area

    setattr(
        mesh,
        f"_last_bending_tilt_{cache_tag}_grad_linear_transition_stats",
        transition_operator_stats,
    )

    if tilt_grad_arr is not None:
        tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
        if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_grad_arr must have shape (N_vertices, 3)")

        dE_ddiv_base, mode_stats = _inner_bending_tilt_dE_ddiv(
            mesh=mesh,
            global_params=global_params,
            cache_tag=cache_tag,
            kappa_tri=kappa_tri,
            base_tri=base_tri,
            div_term=div_eval_tri,
            va0_eff=va0_eff,
            va1_eff=va1_eff,
            va2_eff=va2_eff,
        )
        if str(cache_tag) == "in":
            setattr(mesh, "_last_bending_tilt_in_update_mode_stats", mode_stats)
        if use_recovered_div:
            dE_ddiv = _inner_recovered_divergence_pullback(
                global_params=global_params,
                cache_tag=cache_tag,
                tri_rows=tri_rows,
                tri_area=tri_area,
                coeff_div_eval=float(div_sign) * dE_ddiv_base,
                v_area=div_eval_vertex_area,
                ctx=ctx,
                scratch_tag=f"btl_{cache_tag}",
            )
        else:
            dE_ddiv = float(div_sign) * dE_ddiv_base
        factor = dE_ddiv[:, None]

        np.add.at(tilt_grad_arr, tri_rows[:, 0], factor * g0)
        np.add.at(tilt_grad_arr, tri_rows[:, 1], factor * g1)
        np.add.at(tilt_grad_arr, tri_rows[:, 2], factor * g2)

    return total_energy


def compute_energy_and_gradient_leaflet(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    tilts: np.ndarray,
    kappa_key: str,
    cache_tag: str,
    div_sign: float = 1.0,
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
    energy = compute_energy_and_gradient_array_leaflet(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts=tilts,
        tilt_grad_arr=tilt_grad_arr,
        kappa_key=kappa_key,
        cache_tag=cache_tag,
        div_sign=div_sign,
    )
    if not compute_gradient:
        return float(energy), {}

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
    return float(energy), shape_grad, tilt_grad


__all__ = [
    "compute_energy_and_gradient_array_leaflet",
    "compute_energy_and_gradient_leaflet",
    "_apply_inner_divergence_update_mode",
    "_base_term_region_zero_rows",
    "_collect_group_rows",
    "_collect_preset_rows",
    "_interior_mask_leaflet",
    "_shared_rim_support_transition_triangle_mask",
]
