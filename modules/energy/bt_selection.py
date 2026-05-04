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
)

_BASE_TERM_BOUNDARY_OPTION_KEYS = (
    # Most configs tag the disk interface ring via the rim-slope match group.
    "rim_slope_match_group",
    # Some configs use explicit thetaB group tags.
    "tilt_thetaB_group",
    "tilt_thetaB_group_in",
    "tilt_thetaB_group_out",
)


def _collect_preset_rows(
    mesh: Mesh,
    *,
    presets: tuple[str, ...],
    cache_tag: str,
    index_map: Dict[int, int],
    radius_max: float | None = None,
    center_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Return vertex-row indices whose ``preset`` option is in ``presets``."""
    if not presets:
        return np.zeros(0, dtype=int)
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
    mesh: Mesh, *, group: str, index_map: Dict[int, int]
) -> np.ndarray:
    """Return vertex-row indices whose options tag them as members of ``group``."""
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
    index_map: Dict[int, int],
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
    index_map: Dict[int, int],
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
