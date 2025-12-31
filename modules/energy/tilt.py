"""Tilt energy module.

This module currently models a simple per-vertex tilt penalty:

    E = 1/2 * k_t * sum_v (|t_v|^2 * A_v)

where ``t_v`` is a 2D tilt vector stored on each vertex and ``A_v`` is a
per-vertex area weight (Voronoi area).

Note: The solver's minimization state is currently vertex positions only.
Accordingly, the dense-array API accumulates only the *shape* gradient
(w.r.t. positions). The tilt gradient is returned by the legacy API for
callers/tests that treat tilt as a separate degree of freedom.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver
) -> Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return tilt energy and gradients.

    Returns
    -------
    (E, shape_grad, tilt_grad)
        ``shape_grad`` is currently zero because the Voronoi-area derivative
        w.r.t. positions is not modeled here.
    """
    k_tilt = float(param_resolver.get(None, "tilt_rigidity") or 0.0)
    if k_tilt == 0.0:
        shape_grad = {i: np.zeros(3, dtype=float) for i in mesh.vertices}
        tilt_grad = {i: np.zeros(2, dtype=float) for i in mesh.vertices}
        return 0.0, shape_grad, tilt_grad

    energy = 0.0
    shape_grad = {i: np.zeros(3, dtype=float) for i in mesh.vertices}
    tilt_grad = {i: np.zeros(2, dtype=float) for i in mesh.vertices}

    for vertex in mesh.vertices.values():
        tilt_vec = np.asarray(getattr(vertex, "tilt", np.zeros(2, dtype=float)))
        area = float(vertex.voronoi_area())
        energy += 0.5 * k_tilt * float(np.dot(tilt_vec, tilt_vec)) * area
        tilt_grad[vertex.index] += k_tilt * tilt_vec * area

    return float(energy), shape_grad, tilt_grad


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray,
) -> float:
    """Dense-array tilt energy accumulation (shape gradient only)."""
    k_tilt = float(param_resolver.get(None, "tilt_rigidity") or 0.0)
    if k_tilt == 0.0:
        return 0.0

    energy = 0.0
    for vertex in mesh.vertices.values():
        tilt_vec = np.asarray(getattr(vertex, "tilt", np.zeros(2, dtype=float)))
        area = float(vertex.voronoi_area())
        energy += 0.5 * k_tilt * float(np.dot(tilt_vec, tilt_vec)) * area
    return float(energy)


__all__ = ["compute_energy_and_gradient", "compute_energy_and_gradient_array"]
