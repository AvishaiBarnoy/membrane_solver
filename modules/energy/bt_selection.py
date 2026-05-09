"""Vertex and group selection helpers for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.entities import Mesh

# These must be imported from .bt_params to avoid circularity if they were in bending_tilt_leaflet.py
# However, we'll assume they are available from .bt_params as we just moved them there.
from .bt_params import (
    _assume_J0_center_xy,
    _base_term_boundary_group,
    _base_term_region_mode,
    _base_term_region_radius,
    _bending_tilt_in_update_mode,
)

_BASE_TERM_BOUNDARY_OPTION_KEYS = (
    # Most configs tag the disk interface ring via the rim-slope match group.
    "rim_slope_match_group",
    # Some configs use explicit thetaB group tags.
    "tilt_thetaB_group",
    "tilt_thetaB_group_in",
    "tilt_thetaB_group_out",
)


def _apply_inner_divergence_update_mode(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    tri_rows: np.ndarray,
    cache_tag: str,
    div_term: np.ndarray,
) -> np.ndarray:
    """Apply benchmark-only inner divergence cap beyond the rim."""
    stats_attr = "_last_bending_tilt_in_update_mode_stats"
    mode = _bending_tilt_in_update_mode(global_params)
    stats = {
        "enabled": bool(mode != "off"),
        "mode": str(mode),
        "candidate_tri_count": 0,
        "capped_tri_count": 0,
        "rim_tri_count": 0,
        "cap_magnitude": 0.0,
        "cross_term_removed": False,
    }
    if str(cache_tag) != "in":
        return div_term
    if mode == "off" or tri_rows.size == 0:
        setattr(mesh, stats_attr, stats)
        return div_term
    if mode == "radial_cross_term_off_v1":
        stats["cross_term_removed"] = True
        setattr(mesh, stats_attr, stats)
        return div_term

    radius = float(global_params.get("benchmark_disk_radius") or 0.0)
    lambda_value = float(global_params.get("benchmark_lambda_value") or 0.0)
    if radius <= 0.0 or lambda_value <= 0.0:
        setattr(mesh, stats_attr, stats)
        return div_term

    center = _assume_J0_center_xy(global_params)
    tri_xy = np.mean(positions[tri_rows, :2], axis=1)
    tri_radii = np.linalg.norm(tri_xy - center[None, :], axis=1)
    rim_w = float(lambda_value)
    near_w = 4.0 * float(lambda_value)
    rim_mask = np.abs(tri_radii - radius) <= rim_w
    outer_near_mask = (tri_radii > (radius + rim_w)) & (tri_radii <= (radius + near_w))
    stats["candidate_tri_count"] = int(np.sum(outer_near_mask))
    stats["rim_tri_count"] = int(np.sum(rim_mask))
    rim_mag = np.abs(div_term[rim_mask])
    if rim_mag.size == 0:
        setattr(mesh, stats_attr, stats)
        return div_term

    cap_magnitude = float(1.05 * np.median(rim_mag))
    stats["cap_magnitude"] = cap_magnitude
    if cap_magnitude <= 0.0 or not np.any(outer_near_mask):
        setattr(mesh, stats_attr, stats)
        return div_term

    updated = np.array(div_term, copy=True)
    hit = outer_near_mask & (np.abs(updated) > cap_magnitude)
    updated[hit] = np.sign(updated[hit]) * cap_magnitude
    stats["capped_tri_count"] = int(np.sum(hit))
    setattr(mesh, stats_attr, stats)
    return updated


def _shared_rim_support_transition_triangle_mask(
    mesh: Mesh,
    global_params,
    tri_rows: np.ndarray,
    *,
    keep_physical_outer_edge: bool = False,
) -> np.ndarray | None:
    """Mask triangles incident to the artificial shared-rim support row."""
    if global_params is None:
        return None
    mode = str(global_params.get("rim_slope_match_mode") or "").strip()
    if mode != "shared_rim_staggered_v1":
        return None

    support_group = str(global_params.get("rim_slope_match_outer_group") or "").strip()
    rim_group = str(global_params.get("rim_slope_match_group") or "").strip()
    disk_group = str(global_params.get("rim_slope_match_disk_group") or "").strip()
    if not support_group or not rim_group or not disk_group:
        return None

    support_rows = np.zeros(len(mesh.vertex_ids), dtype=bool)
    true_free_rows = np.ones(len(mesh.vertex_ids), dtype=bool)
    excluded_groups = {support_group, rim_group, disk_group}
    for row, vertex_id in enumerate(mesh.vertex_ids):
        vertex = mesh.vertices[int(vertex_id)]
        group = vertex.options.get("rim_slope_match_group")
        if group == support_group:
            support_rows[row] = True
        if group in excluded_groups:
            true_free_rows[row] = False

    if not np.any(support_rows):
        return None
    has_support = np.any(support_rows[tri_rows], axis=1)
    if not keep_physical_outer_edge:
        return has_support
    free_corner_count = np.sum(true_free_rows[tri_rows], axis=1)
    has_physical_outer_edge = has_support & (free_corner_count >= 2)
    return has_support & ~has_physical_outer_edge


def _collect_preset_rows(
    mesh: Mesh,
    *,
    presets: tuple[str, ...],
    cache_tag: str,
    index_map: Dict[int, int] | None = None,
    radius_max: float | None = None,
    center_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Return vertex-row indices whose ``preset`` option is in ``presets``."""
    if not presets:
        return np.zeros(0, dtype=int)

    mesh.build_position_cache()
    if index_map is None:
        index_map = mesh.vertex_index_to_row

    radius_key = None if radius_max is None else float(radius_max)
    center_key = None
    if center_xy is not None:
        center_arr = np.asarray(center_xy, dtype=float).reshape(-1)
        if center_arr.size >= 2:
            center_key = (float(center_arr[0]), float(center_arr[1]))
    cache_key = (mesh._vertex_ids_version, presets, radius_key, center_key)
    cache_attr = f"_bending_tilt_assume_J0_rows_cache_{cache_tag}"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["rows"]
    presets_set = set(presets)
    present: set[str] = set()
    center = (
        np.zeros(2, dtype=float)
        if center_xy is None
        else np.asarray(center_xy, dtype=float)
        .reshape(-1)[:2]
        .astype(float, copy=False)
    )
    radius_tol = 1.0e-12

    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        preset = str(opts.get("preset") or "")
        present.add(preset)
        if preset in presets_set:
            row = index_map.get(int(vid))
            if row is not None:
                if radius_max is not None:
                    pos = np.asarray(mesh.vertices[int(vid)].position, dtype=float)
                    radius = float(np.linalg.norm(pos[:2] - center))
                    if radius > float(radius_max) + radius_tol:
                        continue
                rows.append(int(row))

    unknown = presets_set - present
    if unknown:
        raise ValueError(
            "Unknown presets in bending_tilt_assume_J0_presets: "
            + ", ".join(sorted(unknown))
        )

    out = np.asarray(rows, dtype=int)
    setattr(mesh, cache_attr, {"key": cache_key, "rows": out})
    return out


def _collect_group_rows(
    mesh: Mesh, *, group: str, index_map: Dict[int, int] | None = None
) -> np.ndarray:
    """Return vertex-row indices whose options tag them as members of ``group``."""
    mesh.build_position_cache()
    if index_map is None:
        index_map = mesh.vertex_index_to_row

    cache_key = (mesh._vertex_ids_version, str(group))
    cache_attr = "_bending_tilt_group_rows_cache"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["rows"]

    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if any(opts.get(key) == group for key in _BASE_TERM_BOUNDARY_OPTION_KEYS):
            row = index_map.get(int(vid))
            if row is not None:
                rows.append(int(row))
    out = np.asarray(rows, dtype=int)
    setattr(mesh, cache_attr, {"key": cache_key, "rows": out})
    return out


def _base_term_region_zero_rows(
    mesh: Mesh,
    global_params,
    *,
    cache_tag: str,
    index_map: Dict[int, int] | None = None,
) -> np.ndarray:
    """Return extra rows zeroed by benchmark-scoped base-term region modes."""
    mode = _base_term_region_mode(global_params)
    if mode == "off":
        return np.zeros(0, dtype=int)
    radius = _base_term_region_radius(global_params)
    if radius is None:
        raise ValueError(
            "bending_tilt_base_term_region_radius is required when "
            "bending_tilt_base_term_region_mode is enabled."
        )

    mesh.build_position_cache()
    if index_map is None:
        index_map = mesh.vertex_index_to_row

    if mode in {"physical_disk_split_v1", "disk_only_base_term_v1"}:
        if mode == "physical_disk_split_v1" and str(cache_tag) != "out":
            return np.zeros(0, dtype=int)
        if mode == "disk_only_base_term_v1" and str(cache_tag) != "in":
            return np.zeros(0, dtype=int)
        center = _assume_J0_center_xy(global_params)
        cache_key = (
            mesh._vertex_ids_version,
            str(cache_tag),
            str(mode),
            float(radius),
            float(center[0]),
            float(center[1]),
        )
        cache_attr = "_bending_tilt_base_term_region_rows_cache"
        cached = getattr(mesh, cache_attr, None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["rows"]
        rows: list[int] = []
        for vid in mesh.vertex_ids:
            row = index_map.get(int(vid))
            if row is None:
                continue
            pos = np.asarray(mesh.vertices[int(vid)].position, dtype=float)
            row_radius = float(np.linalg.norm(pos[:2] - center))
            if mode == "physical_disk_split_v1":
                if row_radius <= radius + 1.0e-12:
                    rows.append(int(row))
            else:
                if row_radius > radius + 1.0e-12:
                    rows.append(int(row))
        out = np.asarray(rows, dtype=int)
        setattr(mesh, cache_attr, {"key": cache_key, "rows": out})
        return out
    raise AssertionError("unreachable")


def _interior_mask_leaflet(
    mesh: Mesh,
    global_params,
    *,
    cache_tag: str,
    index_map: Dict[int, int] | None = None,
) -> np.ndarray:
    """Return cached interior mask for base-term evaluation.

    Interior vertices are all non-boundary vertices, optionally excluding a
    configured interface group and preset list used for theory-mode matching.
    """
    group = _base_term_boundary_group(global_params, cache_tag=cache_tag)
    cache_key = (
        int(mesh._vertex_ids_version),
        int(mesh._topology_version),
        None if group is None else str(group),
    )
    cache_attr = f"_bending_tilt_interior_mask_cache_{cache_tag}"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["mask"]

    mesh.build_position_cache()
    if index_map is None:
        index_map = mesh.vertex_index_to_row

    is_interior = np.ones(len(mesh.vertex_ids), dtype=bool)

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        if boundary_rows:
            is_interior[np.asarray(boundary_rows, dtype=int)] = False

    if group:
        rows = _collect_group_rows(mesh, group=group, index_map=index_map)
        if rows.size:
            is_interior[rows] = False

    setattr(mesh, cache_attr, {"key": cache_key, "mask": is_interior})
    return is_interior
