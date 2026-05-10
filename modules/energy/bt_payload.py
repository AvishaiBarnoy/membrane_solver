"""Payload builders for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.curvature import compute_curvature_data
from geometry.entities import Mesh
from modules.energy.bending_params import (
    _energy_model,
    _spontaneous_curvature,
)
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)

from .bt_params import (
    _assume_J0_center_xy,
    _assume_J0_presets,
    _assume_J0_radius_max,
    _base_term_boundary_group,
    _base_term_region_mode,
    _base_term_region_radius,
    _per_vertex_params_leaflet,
    _resolve_bending_modulus,
)
from .bt_selection import (
    _base_term_region_zero_rows,
    _collect_preset_rows,
    _interior_mask_leaflet,
    _shared_rim_support_transition_triangle_mask,
)


def _leaflet_triangle_payload(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    cache_tag: str,
    ctx=None,
) -> dict[str, np.ndarray | None]:
    """Return cached leaflet-masked triangle geometry for fixed positions."""
    mesh.build_position_cache()
    absent_mask = None
    absent_key = None
    if cache_tag in {"in", "out"}:
        absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet=cache_tag)
        absent_key = id(absent_mask)

    use_cache = mesh._geometry_cache_active(positions)
    cache_attr = f"_bending_tilt_leaflet_triangle_payload_cache_{cache_tag}"
    cache_key = (
        int(mesh._version),
        int(mesh._facet_loops_version),
        int(mesh._vertex_ids_version),
        id(positions),
        absent_key,
        str(global_params.get("rim_slope_match_mode") or "")
        if global_params is not None
        else "",
        str(global_params.get("rim_slope_match_outer_group") or "")
        if global_params is not None
        else "",
        str(global_params.get("rim_slope_match_group") or "")
        if global_params is not None
        else "",
        str(global_params.get("rim_slope_match_disk_group") or "")
        if global_params is not None
        else "",
    )
    if use_cache:
        cached = getattr(mesh, cache_attr, None)
        if cached is not None and cached.get("key") == cache_key:
            payload = cached["value"]
            if isinstance(payload, dict) and "tri_area" not in payload:
                payload = dict(payload)
                payload.setdefault("tri_area", None)
                cached["value"] = payload
            return payload

    k_vecs, vertex_areas_vor, weights_full, tri_rows_full = compute_curvature_data(
        mesh, positions, index_map
    )
    if tri_rows_full.size == 0:
        payload = {
            "k_vecs": k_vecs,
            "vertex_areas_vor": vertex_areas_vor,
            "weights_full": weights_full,
            "tri_rows_full": tri_rows_full,
            "weights": weights_full,
            "tri_rows": tri_rows_full,
            "tri_keep": np.zeros(0, dtype=bool),
            "tri_area": None,
            "g0": None,
            "g1": None,
            "g2": None,
        }
        if use_cache:
            setattr(mesh, cache_attr, {"key": cache_key, "value": payload})
        return payload

    tri_keep = np.zeros(0, dtype=bool)
    weights = weights_full
    tri_rows = tri_rows_full
    if absent_mask is not None:
        tri_keep = leaflet_present_triangle_mask(
            mesh, tri_rows_full, absent_vertex_mask=absent_mask
        )
        if tri_keep.size:
            tri_rows = tri_rows_full[tri_keep]
            weights = weights_full[tri_keep]

    transition_mask = _shared_rim_support_transition_triangle_mask(
        mesh,
        global_params,
        tri_rows_full,
        keep_physical_outer_edge=str(cache_tag) == "out",
    )

    tri_area = None
    g0 = None
    g1 = None
    g2 = None
    if ctx is not None:
        # P1 shape gradients are only needed for divergence pullback paths.
        (
            _area_cache,
            g0_cache,
            g1_cache,
            g2_cache,
            tri_rows_cache,
        ) = mesh.p1_triangle_shape_gradient_cache(positions)
        if tri_rows_cache.size and tri_rows_cache.shape[0] == tri_rows_full.shape[0]:
            if tri_keep.size:
                tri_area = _area_cache[tri_keep]
                g0 = g0_cache[tri_keep]
                g1 = g1_cache[tri_keep]
                g2 = g2_cache[tri_keep]
            else:
                tri_area = _area_cache
                g0 = g0_cache
                g1 = g1_cache
                g2 = g2_cache

    payload = {
        "k_vecs": k_vecs,
        "vertex_areas_vor": vertex_areas_vor,
        "weights_full": weights_full,
        "tri_rows_full": tri_rows_full,
        "weights": weights,
        "tri_rows": tri_rows,
        "tri_keep": tri_keep,
        "transition_mask": transition_mask,
        "tri_area": tri_area,
        "g0": g0,
        "g1": g1,
        "g2": g2,
    }
    if use_cache:
        setattr(mesh, cache_attr, {"key": cache_key, "value": payload})
    return payload


def _leaflet_static_tilt_payload(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    k_vecs: np.ndarray,
    vertex_areas_vor: np.ndarray,
    tri_rows: np.ndarray,
    kappa_key: str,
    cache_tag: str,
) -> dict[str, np.ndarray]:
    """Return cached fixed-geometry arrays used by tilt-only leaflet coupling."""
    model = _energy_model(global_params)
    if model != "helfrich":
        model = "helfrich"
    kappa_default = _resolve_bending_modulus(global_params, kappa_key)
    c0_key = f"spontaneous_curvature_{cache_tag}"
    c0_default = global_params.get(c0_key)
    if c0_default is None:
        c0_default = (
            _spontaneous_curvature(global_params) if model == "helfrich" else 0.0
        )
    c0_default = float(c0_default or 0.0)

    presets = _assume_J0_presets(global_params, cache_tag=cache_tag)
    radius_max = _assume_J0_radius_max(global_params, cache_tag=cache_tag)
    center_xy = _assume_J0_center_xy(global_params)
    center_key = (float(center_xy[0]), float(center_xy[1]))
    region_mode = _base_term_region_mode(global_params)
    region_radius = _base_term_region_radius(global_params)
    boundary_group = _base_term_boundary_group(global_params, cache_tag=cache_tag)

    use_cache = mesh._geometry_cache_active(positions)
    cache_attr = f"_bending_tilt_leaflet_static_cache_{cache_tag}"
    cache_key = (
        int(mesh._version),
        int(mesh._facet_loops_version),
        int(mesh._vertex_ids_version),
        int(mesh._topology_version),
        id(positions),
        id(k_vecs),
        id(vertex_areas_vor),
        id(tri_rows),
        str(model),
        float(kappa_default),
        float(c0_default),
        None if boundary_group is None else str(boundary_group),
        presets,
        None if radius_max is None else float(radius_max),
        center_key,
        str(region_mode),
        None if region_radius is None else float(region_radius),
    )
    if use_cache:
        cached = getattr(mesh, cache_attr, None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["value"]

    kappa_arr, c0_arr = _per_vertex_params_leaflet(
        mesh, global_params, model=model, kappa_key=kappa_key, cache_tag=cache_tag
    )
    safe_areas_vor = np.maximum(vertex_areas_vor, 1e-12)
    k_mag = np.linalg.norm(k_vecs, axis=1)
    h_vor = k_mag / (2.0 * safe_areas_vor)
    is_interior = _interior_mask_leaflet(
        mesh, global_params, cache_tag=cache_tag, index_map=index_map
    )

    base_term = (2.0 * h_vor) - c0_arr
    base_term[~is_interior] = 0.0
    if presets:
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
    kappa_tri = kappa_arr[tri_rows]
    value = {
        "kappa_arr": kappa_arr,
        "c0_arr": c0_arr,
        "h_vor": h_vor,
        "is_interior": is_interior,
        "base_term": base_term,
        "base_tri": base_tri,
        "kappa_tri": kappa_tri,
    }
    if use_cache:
        setattr(mesh, cache_attr, {"key": cache_key, "value": value})
    return value
