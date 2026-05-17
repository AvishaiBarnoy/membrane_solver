"""Shared helpers for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.entities import Mesh
from geometry.tilt_operators import (
    _resolve_transport_model,
    compute_divergence_from_basis,
    p1_triangle_divergence,
)
from modules.energy.bending_params import (
    _energy_model,
    _gradient_mode,
)
from modules.energy.bending_utils import (
    _compute_effective_areas,
    _vertex_normals,
)
from modules.energy.scatter import scatter_triangle_scalar_to_vertices

from .bt_diagnostics import (
    _finite_difference_gradient_shape_leaflet,
    _total_energy_leaflet,
)
from .bt_divergence import (
    _inner_bending_tilt_dE_ddiv,
    _inner_recovered_divergence,
    _inner_recovered_divergence_pullback,
)
from .bt_gradient import (
    _backpropagate_bending_tilt_shape_gradient,
)
from .bt_params import (
    _assume_J0_center_xy,
    _assume_J0_presets,
    _assume_J0_radius_max,
    _base_term_boundary_group,
    _base_term_reference_mode,
    _per_vertex_params_leaflet,
    _resolve_bending_modulus,
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
from .bt_utils import _accumulate_leaflet_tilt_gradient


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
    if _base_term_reference_mode(global_params) == "flat_reference_zero_j0":
        base_term[:] = 0.0
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
            "lane": (
                str(global_params.get("theory_parity_lane") or "")
                if global_params is not None
                else ""
            ),
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
        factor_K_vec = np.empty_like(K_dir, order="C")
    np.multiply(K_dir, scale_K[:, None], out=factor_K_vec)

    transition_operator_enabled = _use_stage_a_outer_grad_linear_transition_operator(
        global_params, cache_tag=cache_tag
    )
    transition_operator_stats = {
        "enabled": False,
        "mode": "off",
        "cache_tag": str(cache_tag),
        "lane": (
            str(global_params.get("theory_parity_lane") or "")
            if global_params is not None
            else ""
        ),
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
    else:
        transition_operator_stats = _backpropagate_bending_tilt_shape_gradient(
            mesh,
            positions=positions,
            tri_rows=tri_rows,
            tri_rows_full=tri_rows_full,
            weights=weights,
            tri_keep=tri_keep,
            is_interior=is_interior,
            fA_eff=fA_eff,
            fA_vor=fA_vor,
            factor_K_vec=factor_K_vec,
            grad_arr=grad_arr,
            cache_tag=cache_tag,
            mode=mode,
            ctx=ctx,
            transition_operator_enabled=transition_operator_enabled,
        )

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
    "_total_energy_leaflet",
    "_finite_difference_gradient_shape_leaflet",
    "_backpropagate_bending_tilt_shape_gradient",
    "_assume_J0_center_xy",
    "_assume_J0_presets",
    "_assume_J0_radius_max",
    "_base_term_boundary_group",
    "_per_vertex_params_leaflet",
    "_resolve_bending_modulus",
    "_use_inner_recovered_divergence",
    "_use_stage_a_inner_shape_cross_suppression",
    "_use_stage_a_outer_grad_linear_transition_operator",
]
