import logging
from typing import Dict

import numpy as np

logger = logging.getLogger("membrane_solver")


def enforce_constraint(mesh, tol: float = 1e-12, max_iter: int = 3, **_kwargs) -> None:
    """Enforce a global surface-area constraint over the entire mesh."""

    target_area = getattr(mesh.global_parameters, "target_surface_area", None)
    if target_area is None:
        target_area = mesh.global_parameters.get("target_surface_area")
    if target_area is None:
        return

    for _ in range(max_iter):
        current_area = mesh.compute_total_surface_area()
        delta = current_area - target_area
        if abs(delta) < tol:
            break

        grad: Dict[int, np.ndarray] = {vidx: np.zeros(3) for vidx in mesh.vertices}
        for facet in mesh.facets.values():
            facet_grad = facet.compute_area_gradient(mesh)
            for vidx, vec in facet_grad.items():
                grad[vidx] += vec

        norm_sq = sum(np.dot(vec, vec) for vec in grad.values())
        if norm_sq < 1e-18:
            logger.debug("Global area constraint skipped due to near-zero gradient.")
            break

        lam = delta / (norm_sq + 1e-18)
        logger.debug(
            "Applying global area constraint: ΔA=%.3e, λ=%.3e",
            delta,
            lam,
        )

        for vidx, gvec in grad.items():
            vertex = mesh.vertices[vidx]
            if getattr(vertex, "fixed", False):
                continue
            vertex.position -= lam * gvec


__all__ = ["enforce_constraint"]
