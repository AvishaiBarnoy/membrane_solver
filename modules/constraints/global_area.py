import logging

import numpy as np

logger = logging.getLogger("membrane_solver")


def enforce_constraint(mesh, tol: float = 1e-12, max_iter: int = 3, **_kwargs) -> None:
    """Enforce a global surface-area constraint over the entire mesh."""

    target_area = getattr(mesh.global_parameters, "target_surface_area", None)
    if target_area is None:
        target_area = mesh.global_parameters.get("target_surface_area")
    if target_area is None:
        return

    positions = None
    index_map = None
    if getattr(mesh, "facet_vertex_loops", None):
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row

    for _ in range(max_iter):
        current_area, grad = mesh.compute_total_area_and_gradient(
            positions=positions, index_map=index_map
        )
        delta = current_area - target_area
        if abs(delta) < tol:
            break

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

        mesh.increment_version()
        if positions is not None:
            positions = mesh.positions_view()


__all__ = ["enforce_constraint"]
