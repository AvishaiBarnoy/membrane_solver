"""Soft body-surface-area penalty energy.

For each body with an ``area_target`` value, this module adds

    E_body = 1/2 * k * (A - A₀)²

where
    A   = total surface area of the body's facets
    A₀  = target area (``body.options['area_target']``)
    k   = stiffness (``body.options['area_stiffness']`` or
          ``global_params.area_stiffness``).

The gradient is k * (A - A₀) * ∂A/∂x, constructed by aggregating per‑facet
area gradients for all facets in the body.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np

from logging_config import setup_logging

logger = setup_logging("membrane_solver")


def compute_energy_and_gradient(
    mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> tuple[float, Dict[int, np.ndarray]]:
    """Return soft body-area penalty energy and gradient."""
    energy = 0.0
    grad: Dict[int, np.ndarray] | None = (
        defaultdict(lambda: np.zeros(3, dtype=float)) if compute_gradient else None
    )

    # Default stiffness if not overridden per body.
    default_k = float(getattr(global_params, "area_stiffness", 0.0) or 0.0)

    for body in mesh.bodies.values():
        target = body.options.get("area_target")
        if target is None:
            continue

        k = param_resolver.get(body, "area_stiffness")
        if k is None:
            k = default_k
        k = float(k)
        if k == 0.0:
            continue

        area = body.compute_surface_area(mesh)
        delta = area - float(target)
        energy += 0.5 * k * delta * delta

        if not compute_gradient:
            continue

        # Aggregate facet area gradients for this body.
        body_grad: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=float))
        for facet_idx in body.facet_indices:
            facet = mesh.facets[facet_idx]
            facet_grad = facet.compute_area_gradient(mesh)
            for vidx, vec in facet_grad.items():
                body_grad[vidx] += vec

        factor = k * delta
        for vidx, gvec in body_grad.items():
            grad[vidx] += factor * gvec

    if not compute_gradient:
        return float(energy), {}

    return float(energy), dict(grad)


__all__ = ["compute_energy_and_gradient"]

