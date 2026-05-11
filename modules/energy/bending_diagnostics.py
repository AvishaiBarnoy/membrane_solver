"""Diagnostic and finite-difference helpers for Helfrich/Willmore bending energy."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.entities import Mesh


def _finite_difference_gradient(
    mesh: Mesh,
    global_params,
    positions: np.ndarray,
    index_map: Dict[int, int],
    *,
    eps: float,
) -> np.ndarray:
    """Energy-consistent gradient by central differences (slow)."""
    # Import compute_total_energy locally to avoid top-level circularity if possible,
    # or rely on the orchestrator passing it. Actually, bending.py will be the orchestrator.
    from .bending import compute_total_energy

    grad = np.zeros_like(positions)
    base = positions.copy()
    mesh.build_position_cache()
    for row, vid in enumerate(mesh.vertex_ids):
        if getattr(mesh.vertices[int(vid)], "fixed", False):
            continue
        for d in range(3):
            pos_plus = base.copy()
            pos_minus = base.copy()
            pos_plus[row, d] += eps
            pos_minus[row, d] -= eps
            e_plus = compute_total_energy(mesh, global_params, pos_plus, index_map)
            e_minus = compute_total_energy(mesh, global_params, pos_minus, index_map)
            grad[row, d] = (e_plus - e_minus) / (2.0 * eps)

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        grad[boundary_rows] = 0.0
    return grad
