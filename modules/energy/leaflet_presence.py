"""Helpers for masking leaflet-specific energies on subsets of the mesh.

Some theoretical setups treat one leaflet as absent in specific regions (e.g.
protein-covered patches). This file provides lightweight utilities for energy
modules to exclude triangles that touch those regions.
"""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh


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
    if not absent_presets:
        return np.zeros(len(mesh.vertex_ids), dtype=bool)

    mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        preset = opts.get("preset")
        if preset in absent_presets:
            mask[row] = True
    return mask


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
