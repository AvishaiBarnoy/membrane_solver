import logging
from collections import defaultdict
from typing import Dict

import numpy as np

logger = logging.getLogger("membrane_solver")


def apply_constraint_gradient(grad: Dict[int, np.ndarray], mesh, global_params) -> None:
    """Project the gradient to preserve each body's target area."""
    for body in mesh.bodies.values():
        if body.options.get("target_area") is None:
            continue

        gA = {}
        for facet_idx in body.facet_indices:
            facet = mesh.facets[facet_idx]
            # Note: compute_area_and_gradient is defined on Facet in entities.py
            # but this module's enforce_constraint uses compute_area_gradient.
            # We'll stick to what Facet provides. Assuming compute_area_and_gradient exists.
            _, g_f = facet.compute_area_and_gradient(mesh)
            for vidx, vec in g_f.items():
                if vidx not in gA:
                    gA[vidx] = vec.copy()
                else:
                    gA[vidx] += vec

        norm_sq = sum(np.dot(v, v) for v in gA.values())
        if norm_sq < 1e-18:
            continue

        dot = 0.0
        for vidx, gvec in gA.items():
            if vidx in grad:
                dot += float(np.dot(grad[vidx], gvec))
        lam = dot / (norm_sq + 1e-18)
        for vidx, vec in gA.items():
            if vidx in grad:
                grad[vidx] -= lam * vec


def enforce_constraint(mesh, tol: float = 1e-12, max_iter: int = 20) -> None:
    """Enforce hard surface-area constraints on bodies using Lagrange multipliers.

    Each body may define ``target_area`` in ``body.options``. After each call,
    the body's surface area is nudged to match that target precisely (within
    ``tol``) by displacing vertices along the aggregated area gradient.
    """

    for body in mesh.bodies.values():
        target_area = body.options.get("target_area")
        if target_area is None:
            continue

        for _ in range(max_iter):
            current_area = body.compute_surface_area(mesh)
            delta = current_area - target_area
            if abs(delta) < tol:
                break

            grad: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3))
            for facet_idx in body.facet_indices:
                facet = mesh.facets[facet_idx]
                facet_grad = facet.compute_area_gradient(mesh)
                for vidx, vec in facet_grad.items():
                    grad[vidx] += vec

            norm_sq = sum(np.dot(vec, vec) for vec in grad.values())
            if norm_sq < 1e-18:
                logger.debug(
                    "Body %s area constraint skipped due to near-zero gradient.",
                    body.index,
                )
                break

            lam = delta / (norm_sq + 1e-18)
            logger.debug(
                "Applying body area constraint on body %s: ΔA=%.3e, λ=%.3e",
                body.index,
                delta,
                lam,
            )

            for vidx, gvec in grad.items():
                vertex = mesh.vertices[vidx]
                if getattr(vertex, "fixed", False):
                    continue
                vertex.position -= lam * gvec

            mesh.increment_version()


__all__ = ["enforce_constraint"]
