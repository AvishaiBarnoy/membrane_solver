"""Edge length penalty energy module.

Each edge marked with the ``edge_length_penalty`` energy contributes
    E_edge = 0.5 * k * (length - target_length)^2
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from geometry.entities import Mesh


def _edges_to_constrain(mesh: Mesh) -> Iterator[int]:
    """Yield edge IDs that have a target length (or explicit module tag)."""
    for idx, edge in mesh.edges.items():
        opts = getattr(edge, "options", None) or {}
        energy = opts.get("energy", [])
        if "edge_length_penalty" in energy or "target_length" in opts:
            yield int(idx)


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
    grad_arr: np.ndarray,
) -> float:
    """Accumulate edge-length penalty energy and gradient into ``grad_arr``."""
    k = float(global_params.get("edge_stiffness", 100.0))
    total_energy = 0.0

    for idx in _edges_to_constrain(mesh):
        edge = mesh.edges[idx]
        target_L = edge.options.get("target_length")
        if target_L is None:
            continue

        tail_row = index_map.get(edge.tail_index)
        head_row = index_map.get(edge.head_index)
        if tail_row is None or head_row is None:
            continue

        tail = positions[tail_row]
        head = positions[head_row]
        vec = head - tail
        length = np.linalg.norm(vec)

        if length < 1e-15:
            continue

        delta = length - target_L
        total_energy += 0.5 * k * delta**2

        # grad = k * (L - L0) * dL/dx
        # dL/dx_head = vec / L
        # dL/dx_tail = -vec / L
        direction = vec / length
        force = k * delta * direction

        grad_arr[head_row] += force
        grad_arr[tail_row] -= force

    return total_energy
