"""Shared utility logic for leaflet-specific tilt smoothness energy."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.entities import Mesh
from modules.energy import tilt_smoothness as _base
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)


def _masked_weights_and_tris(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    leaflet: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return cached leaflet-masked smoothness payload."""
    weights, tri_rows = _base._get_weights_and_tris(
        mesh, positions=positions, index_map=index_map
    )
    if tri_rows is None:
        return None, None

    # Inner leaflet currently doesn't use masking in the original code,
    # but we'll support it if a mask is present for consistency.
    absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet=leaflet)
    if absent_mask is None or not np.any(absent_mask):
        # No masking needed
        return weights, tri_rows

    absent_token = id(absent_mask)
    use_cache = mesh._geometry_cache_active(positions)
    cache_attr = f"_tilt_smoothness_{leaflet}_mask_cache"
    cache_key = (
        int(mesh._version),
        int(mesh._facet_loops_version),
        int(mesh._vertex_ids_version),
        id(positions),
        id(weights),
        id(tri_rows),
        absent_token,
    )

    if use_cache:
        cached = getattr(mesh, cache_attr, None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["weights"], cached["tri_rows"]

    tri_keep = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_mask
    )
    if tri_keep.size and not np.any(tri_keep):
        weights_use = np.zeros((0, 3), dtype=float)
        tri_rows_use = np.zeros((0, 3), dtype=np.int32)
    elif tri_keep.size:
        weights_use = weights[tri_keep]
        tri_rows_use = tri_rows[tri_keep]
    else:
        weights_use = weights
        tri_rows_use = tri_rows

    if use_cache:
        setattr(
            mesh,
            cache_attr,
            {"key": cache_key, "weights": weights_use, "tri_rows": tri_rows_use},
        )
    return weights_use, tri_rows_use


def _resolve_smoothness_rigidity(param_resolver, leaflet: str) -> float:
    """Resolve smoothness rigidity for the specified leaflet."""
    k = param_resolver.get(None, f"bending_modulus_{leaflet}")
    if k is None:
        k = param_resolver.get(None, "bending_modulus")
    return float(k or 0.0)
