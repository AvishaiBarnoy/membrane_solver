# modules/constraints/fix_facet_area.py
"""Hard facet-area constraints enforced via Lagrange multipliers."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger("membrane_solver")


def enforce_constraint(mesh, tol: float = 1e-12, max_iter: int = 3) -> None:
    """Adjust facets with ``target_area`` to match their specified area."""

    for facet in mesh.facets.values():
        target_area = facet.options.get("target_area")
        if target_area is None:
            continue

        for _ in range(max_iter):
            current_area = facet.compute_area(mesh)
            delta = current_area - target_area
            if abs(delta) < tol:
                break

            grad: Dict[int, np.ndarray] = facet.compute_area_gradient(mesh)
            norm_sq = sum(np.dot(vec, vec) for vec in grad.values())
            if norm_sq < 1e-18:
                logger.debug(
                    "Facet %s area constraint skipped due to near-zero gradient.",
                    facet.index,
                )
                break

            lam = delta / (norm_sq + 1e-18)
            logger.debug(
                "Applying facet area constraint on facet %s: ΔA=%.3e, λ=%.3e",
                facet.index,
                delta,
                lam,
            )

            for vidx, gvec in grad.items():
                vertex = mesh.vertices[vidx]
                if getattr(vertex, "fixed", False):
                    continue
                vertex.position -= lam * gvec


__all__ = ["enforce_constraint"]
