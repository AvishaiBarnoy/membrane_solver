"""Volume constraint enforced via Lagrange multiplier."""

from __future__ import annotations

from typing import Any

import numpy as np

from runtime.logging_config import setup_logging

logger = setup_logging("membrane_solver.log")


def constraint_gradients(mesh, global_params) -> list[dict[int, np.ndarray]] | None:
    """Return constraint gradients for all constrained bodies.

    This supports KKT-style projection in the constraint manager.
    """
    mode = global_params.get("volume_constraint_mode", "lagrange")
    if mode != "lagrange":
        return None

    constrained: list[dict[int, np.ndarray]] = []
    positions = None
    index_map = None
    if getattr(mesh, "facet_vertex_loops", None):
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row
    positions = None
    index_map = None
    if getattr(mesh, "facet_vertex_loops", None):
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row

    positions = None
    index_map = None
    if getattr(mesh, "facet_vertex_loops", None):
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row

    for body in mesh.bodies.values():
        V_target = body.target_volume
        if V_target is None:
            V_target = body.options.get("target_volume")
        if V_target is None:
            continue
        _, vol_grad = body.compute_volume_and_gradient(
            mesh, positions=positions, index_map=index_map
        )
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

    positions = None
    index_map = None
    if getattr(mesh, "facet_vertex_loops", None):
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row

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
            V_actual, grad = body.compute_volume_and_gradient(
                mesh, positions=positions, index_map=index_map
            )

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
            if positions is not None:
                positions = mesh.positions_view()


__all__ = ["enforce_constraint", "constraint_gradients"]
