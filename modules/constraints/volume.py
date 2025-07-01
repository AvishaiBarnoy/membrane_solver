"""Volume constraint enforced via Lagrange multiplier."""

from __future__ import annotations

from typing import Dict
import numpy as np

from logging_config import setup_logging

logger = setup_logging("membrane_solver")


def enforce_constraint(mesh, tol: float = 1e-12, max_iter: int = 3) -> None:
    """Enforce hard volume constraints on all bodies in ``mesh``.

    Each body's vertices are displaced along the volume gradient so that the
    body's volume matches its target volume exactly. The displacement is
    computed using a Lagrange multiplier approach.

    Parameters
    ----------
    mesh : Mesh
        Mesh containing the bodies whose volume must be constrained.
    tol : float, optional
        Tolerance below which volume differences are ignored.
    """

    for body in mesh.bodies.values():
        V_target = body.target_volume
        if V_target is None:
            V_target = body.options.get("target_volume")
        if V_target is None:
            continue

        for _ in range(max_iter):
            V_actual = body.compute_volume(mesh)
            delta_v = V_actual - V_target
            if abs(delta_v) < tol:
                break

            grad = body.compute_volume_gradient(mesh)
            norm_sq = sum(np.dot(g, g) for g in grad.values()) + 1e-12
            lam = delta_v / norm_sq

            logger.debug(
                f"Applying volume constraint on body {body.index}: "
                f"ΔV={delta_v:.3e}, λ={lam:.3e}"
            )

            for vidx, vertex in mesh.vertices.items():
                if vidx not in grad:
                    continue
                if getattr(vertex, "fixed", False):
                    continue
                vertex.position -= lam * grad[vidx]

__all__ = ["enforce_constraint"]
