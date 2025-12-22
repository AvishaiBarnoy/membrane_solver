"""Volume constraint enforced via Lagrange multiplier."""

from __future__ import annotations

from typing import Any

import numpy as np

from runtime.logging_config import setup_logging

logger = setup_logging("membrane_solver.log")


def apply_constraint_gradient(grad: dict[int, np.ndarray], mesh, global_params) -> None:
    """Project the energy gradient to respect volume constraints.

    When ``global_parameters.volume_constraint_mode == "lagrange"``,
    project the gradient onto the subspace orthogonal to all active body
    volume gradients. This solves a small linear system in the space of
    volume constraints.
    """
    mode = global_params.get("volume_constraint_mode", "lagrange")
    if mode != "lagrange":
        return

    # Gather all bodies that have a target volume.
    constrained_bodies = []
    for body in mesh.bodies.values():
        V_target = body.target_volume
        if V_target is None:
            V_target = body.options.get("target_volume")
        if V_target is None:
            continue

        # We compute and cache the volume gradient on the body.
        # This mirrors the original Minimizer behavior which used '_last_volume_grad'
        # for both this projection and subsequent debug checks.
        vol, vol_grad = body.compute_volume_and_gradient(mesh)

        constrained_bodies.append((body, vol_grad))
    if not constrained_bodies:
        return

    k = len(constrained_bodies)
    # A_ij = <∇V_i, ∇V_j>; b_i = <∇E, ∇V_i>
    A = np.zeros((k, k), dtype=float)
    b_vec = np.zeros(k, dtype=float)

    # Build A and b by looping over vertices once per body pair.
    # We must use the projected gradients to correctly account for geometric constraints.
    # Otherwise, we overestimate the ability of constrained vertices to change volume.

    # Pre-process gradients to be projected
    projected_bodies = []
    for body, raw_vol_grad in constrained_bodies:
        proj_grad = {}
        for vidx, g in raw_vol_grad.items():
            vertex = mesh.vertices.get(vidx)
            if vertex:
                proj_grad[vidx] = vertex.project_gradient(g)
            else:
                proj_grad[vidx] = g
        projected_bodies.append((body, proj_grad))

    for i, (_, gradVi) in enumerate(projected_bodies):
        for vidx, gVi in gradVi.items():
            if vidx in grad:
                b_vec[i] += float(np.dot(grad[vidx], gVi))
            for j in range(i, k):
                gVj = projected_bodies[j][1].get(vidx)
                if gVj is None:
                    continue
                val = float(np.dot(gVi, gVj))
                A[i, j] += val
                if j != i:
                    A[j, i] += val

    # Regularise and solve A λ = b.
    eps = 1e-18
    A[np.diag_indices_from(A)] += eps
    try:
        lam = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        return

    # Apply the combined projection to the gradient.
    # We use the projected gradients here as well.
    for (_, gradVi), lam_i in zip(projected_bodies, lam):
        if lam_i == 0.0:
            continue
        for vidx, gVi in gradVi.items():
            if vidx in grad:
                grad[vidx] -= lam_i * gVi


def constraint_gradients(
    mesh, global_params
) -> list[dict[int, np.ndarray]] | None:
    """Return constraint gradients for all constrained bodies.

    This supports KKT-style projection in the constraint manager.
    """
    mode = global_params.get("volume_constraint_mode", "lagrange")
    if mode != "lagrange":
        return None

    constrained: list[dict[int, np.ndarray]] = []
    for body in mesh.bodies.values():
        V_target = body.target_volume
        if V_target is None:
            V_target = body.options.get("target_volume")
        if V_target is None:
            continue
        _, vol_grad = body.compute_volume_and_gradient(mesh)
        constrained.append(vol_grad)

    return constrained or None


def enforce_constraint(
    mesh,
    tol: float = 1e-12,
    max_iter: int = 3,
    global_params: Any | None = None,
    force_projection: bool = False,
    **kwargs,
) -> None:
    """Enforce hard volume constraints on all bodies in ``mesh``.

    Modes:
      - ``force_projection``: always apply the projection step, regardless of
        global settings. Used by the constraint manager after mesh surgery.
      - Otherwise, if ``global_params.volume_constraint_mode == "lagrange"``,
        we still project here because downstream callers expect a hard volume
        correction (e.g., after refinement/equiangulation/averaging).
      - In legacy "projection" mode (or when ``force_projection`` is True),
        bodies are displaced along the volume gradient so that their volume
        matches the target exactly, using a Lagrange‑multiplier step.
    """

    if global_params is not None:
        mode = global_params.get("volume_constraint_mode", "lagrange")
    else:
        mode = "projection"

    project = force_projection or (mode in {"lagrange", "projection"})
    if not project:
        return

    for body in mesh.bodies.values():
        V_target = body.target_volume
        if V_target is None:
            V_target = body.options.get("target_volume")
        if V_target is None:
            continue

        for _ in range(max_iter):
            # Always compute volume and its gradient fresh inside this local
            # projection loop. This keeps the caching strategy simple.
            # persistent per‑body caches are owned by the minimizer's gradient
            # pipeline rather than by the constraint module.
            V_actual, grad = body.compute_volume_and_gradient(mesh)

            delta_v = V_actual - V_target
            if abs(delta_v) < tol:
                break

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

            mesh.increment_version()


__all__ = ["enforce_constraint", "constraint_gradients"]
