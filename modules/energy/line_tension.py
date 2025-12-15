"""Line-tension energy module.

Each edge marked with the ``line_tension`` energy contributes

    E_edge = gamma * |edge|

where ``gamma`` is either specified explicitly on the edge via the
``line_tension`` option or obtained from ``global_params.line_tension``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import numpy as np

from geometry.entities import Mesh
from logging_config import setup_logging

logger = setup_logging("membrane_solver.log")


def _edges_with_line_tension(mesh: Mesh) -> Iterable[int]:
    for idx, edge in mesh.edges.items():
        opts = getattr(edge, "options", {}) or {}
        energy = opts.get("energy")
        has_line = False
        if isinstance(energy, str):
            has_line = energy == "line_tension"
        elif isinstance(energy, (list, tuple)):
            has_line = "line_tension" in energy
        if has_line or "line_tension" in opts:
            yield idx


def _edge_length_and_grad(mesh: Mesh, edge_index: int) -> tuple[float, Dict[int, np.ndarray]]:
    edge = mesh.edges[edge_index]
    tail = mesh.vertices[edge.tail_index].position
    head = mesh.vertices[edge.head_index].position
    vec = head - tail
    length = float(np.linalg.norm(vec))
    if length < 1e-15:
        grad = {
            edge.tail_index: np.zeros(3, dtype=float),
            edge.head_index: np.zeros(3, dtype=float),
        }
        return 0.0, grad
    direction = vec / length
    grad = {
        edge.tail_index: -direction,
        edge.head_index: direction,
    }
    return length, grad


def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> tuple[float, Dict[int, np.ndarray]]:
    edges = list(_edges_with_line_tension(mesh))
    if not edges:
        return 0.0, {}

    default_gamma = float(global_params.get("line_tension", 0.0) or 0.0)

    if not compute_gradient:
        energy = 0.0
        for idx in edges:
            edge = mesh.edges[idx]
            gamma = edge.options.get("line_tension", default_gamma)
            if not gamma:
                continue
            length = edge.compute_length(mesh)
            energy += float(gamma) * float(length)
        logger.debug("Line-tension energy (energy-only path): %.6f", energy)
        return float(energy), {}

    grad: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=float))
    energy = 0.0

    for idx in edges:
        edge = mesh.edges[idx]
        gamma = edge.options.get("line_tension", default_gamma)
        if not gamma:
            continue
        length, edge_grad = _edge_length_and_grad(mesh, idx)
        energy += float(gamma) * length
        for vidx, vec in edge_grad.items():
            grad[vidx] += float(gamma) * vec

    if logger.isEnabledFor(10):
        logger.debug("Computed line-tension energy: %.6f", energy)

    return float(energy), dict(grad)


__all__ = ["compute_energy_and_gradient"]
