"""Diagnostic and finite-difference helpers for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.entities import Mesh
from geometry.tilt_operators import (
    _resolve_transport_model,
    compute_divergence_from_basis,
    p1_triangle_divergence,
)
from modules.energy.bending import (
    _compute_effective_areas,
)

from .bt_divergence import (
    _inner_recovered_divergence,
)
from .bt_params import (
    _use_inner_recovered_divergence,
)
from .bt_payload import (
    _leaflet_static_tilt_payload,
    _leaflet_triangle_payload,
)
from .bt_selection import (
    _apply_inner_divergence_update_mode,
)


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
