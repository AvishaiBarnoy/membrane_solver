"""Helpers for masking leaflet-specific energies on subsets of the mesh.

Some theoretical setups treat one leaflet as absent in specific regions (e.g.
protein-covered patches). This file provides lightweight utilities for energy
modules to exclude triangles that touch those regions.
"""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh
from modules.constraints.local_interface_shells import build_local_interface_shell_data


def _normalize_preset_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        val = raw.strip()
        return [val] if val else []
    if isinstance(raw, (list, tuple, set)):
        out: list[str] = []
        for item in raw:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return []


def leaflet_absent_vertex_mask(
    mesh: Mesh,
    global_params,
    *,
    leaflet: str,
) -> np.ndarray:
    """Return a boolean mask of vertices where a leaflet is absent.

    The mask is based on the per-vertex ``options.preset`` label.

    Parameters
    ----------
    mesh:
        Mesh instance with cached vertex ordering.
    global_params:
        GlobalParameters-like object with ``get``.
    leaflet:
        One of {"in", "out"}.

    Returns
    -------
    np.ndarray
        Boolean array of shape (N_vertices,) aligned with ``mesh.vertex_ids``.
    """
    mesh.build_position_cache()
    leaflet_norm = str(leaflet or "").strip().lower()
    if leaflet_norm not in {"in", "out"}:
        return np.zeros(len(mesh.vertex_ids), dtype=bool)

    key = f"leaflet_{leaflet_norm}_absent_presets"
    absent_presets = set(_normalize_preset_list(global_params.get(key)))
    absent_token = tuple(sorted(absent_presets))
    mode_token = (
        str(global_params.get("rim_slope_match_mode") or "").strip().lower(),
        str(global_params.get(f"leaflet_{leaflet_norm}_absence_mode") or "")
        .strip()
        .lower(),
    )

    cache = getattr(mesh, "_leaflet_absent_mask_cache", None)
    if cache is None:
        cache = {}
        setattr(mesh, "_leaflet_absent_mask_cache", cache)

    entry = cache.get(leaflet_norm)
    if (
        entry is not None
        and entry.get("mesh_version") == int(mesh._version)
        and entry.get("vertex_ids_version") == int(mesh._vertex_ids_version)
        and entry.get("absent_token") == absent_token
        and entry.get("mode_token") == mode_token
        and entry.get("vertex_count") == len(mesh.vertex_ids)
    ):
        return entry["mask"]

    if not absent_presets:
        mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
        cache[leaflet_norm] = {
            "mesh_version": int(mesh._version),
            "vertex_ids_version": int(mesh._vertex_ids_version),
            "absent_token": absent_token,
            "mode_token": mode_token,
            "vertex_count": len(mesh.vertex_ids),
            "mask": mask,
        }
        return mask

    mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        preset = opts.get("preset")
        if preset in absent_presets:
            mask[row] = True
    _restore_physical_edge_outer_trace_rows(
        mesh,
        mask,
        absent_presets=absent_presets,
        leaflet=leaflet_norm,
        mode_token=mode_token,
    )
    cache[leaflet_norm] = {
        "mesh_version": int(mesh._version),
        "vertex_ids_version": int(mesh._vertex_ids_version),
        "absent_token": absent_token,
        "mode_token": mode_token,
        "vertex_count": len(mesh.vertex_ids),
        "mask": mask,
    }
    return mask


def _restore_physical_edge_outer_trace_rows(
    mesh: Mesh,
    mask: np.ndarray,
    *,
    absent_presets: set[str],
    leaflet: str,
    mode_token: tuple[str, str],
) -> None:
    """Keep physical-edge continuation shell rows present for the outer leaflet."""
    if leaflet != "out" or "disk" not in absent_presets:
        return
    rim_mode, absence_mode = mode_token
    if rim_mode != "physical_edge_staggered_v1":
        return
    if absence_mode not in {"triangles", "triangle", "facets", "facet"}:
        return
    try:
        shell_data = build_local_interface_shell_data(
            mesh, positions=mesh.positions_view()
        )
    except AssertionError:
        return
    rows = np.concatenate(
        [
            np.asarray(shell_data.disk_rows, dtype=int),
            np.asarray(shell_data.rim_rows, dtype=int),
            np.asarray(shell_data.outer_rows, dtype=int),
        ]
    )
    if rows.size:
        mask[rows] = False


def leaflet_present_triangle_mask(
    mesh: Mesh,
    tri_rows: np.ndarray,
    *,
    absent_vertex_mask: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask selecting triangles that do not touch absent vertices."""
    if tri_rows is None or len(tri_rows) == 0:
        return np.zeros(0, dtype=bool)
    absent_vertex_mask = np.asarray(absent_vertex_mask, dtype=bool)
    if absent_vertex_mask.size == 0:
        return np.ones(len(tri_rows), dtype=bool)
    return ~np.any(absent_vertex_mask[tri_rows], axis=1)


__all__ = [
    "leaflet_absent_vertex_mask",
    "leaflet_present_triangle_mask",
]
