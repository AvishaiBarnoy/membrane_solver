import logging
from typing import Dict

import numpy as np

logger = logging.getLogger("membrane_solver")


def constraint_gradients(mesh, _global_params) -> list[Dict[int, np.ndarray]] | None:
    """Return constraint gradients for body target areas."""
    gradients: list[Dict[int, np.ndarray]] = []
    positions = None
    index_map = None
    if getattr(mesh, "facet_vertex_loops", None):
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row
    for body in mesh.bodies.values():
        if body.options.get("target_area") is None:
            continue
        _, gA = body.compute_area_and_gradient(
            mesh, positions=positions, index_map=index_map
        )
        gradients.append(gA)

    return gradients or None


def constraint_gradients_array(
    mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
) -> list[np.ndarray] | None:
    """Return dense constraint gradients for body target areas."""
    tri_rows, _ = mesh.triangle_row_cache()

    gradients: list[np.ndarray] = []
    for body in mesh.bodies.values():
        if body.options.get("target_area") is None:
            continue

        gA = np.zeros_like(positions)
        body_rows = body._get_triangle_rows(mesh)
        if body_rows is not None and tri_rows is not None:
            indices = tri_rows[body_rows]
            v0 = positions[indices[:, 0]]
            v1 = positions[indices[:, 1]]
            v2 = positions[indices[:, 2]]

            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            A2 = np.linalg.norm(n, axis=1)
            mask = A2 >= 1e-12
            if np.any(mask):
                n_hat = n[mask] / A2[mask][:, None]
                v0m = v0[mask]
                v1m = v1[mask]
                v2m = v2[mask]
                g0 = 0.5 * np.cross(v1m - v2m, n_hat)
                g1 = 0.5 * np.cross(v2m - v0m, n_hat)
                g2 = 0.5 * np.cross(v0m - v1m, n_hat)
                idx_masked = indices[mask]
                np.add.at(gA, idx_masked[:, 0], g0)
                np.add.at(gA, idx_masked[:, 1], g1)
                np.add.at(gA, idx_masked[:, 2], g2)
            gradients.append(gA)
            continue

        positions_fallback = positions
        for facet_idx in body.facet_indices:
            facet = mesh.facets[facet_idx]
            _, g_f = facet.compute_area_and_gradient(
                mesh, positions=positions_fallback, index_map=index_map
            )
            for vidx, vec in g_f.items():
                row = index_map.get(vidx)
                if row is None:
                    continue
                gA[row] += vec
        gradients.append(gA)

    return gradients or None


def enforce_constraint(mesh, tol: float = 1e-12, max_iter: int = 20, **_kwargs) -> None:
    """Enforce hard surface-area constraints on bodies using Lagrange multipliers.

    Each body may define ``target_area`` in ``body.options``. After each call,
    the body's surface area is nudged to match that target precisely (within
    ``tol``) by displacing vertices along the aggregated area gradient.
    """

    positions = None
    index_map = None
    if getattr(mesh, "facet_vertex_loops", None):
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row

    for body in mesh.bodies.values():
        target_area = body.options.get("target_area")
        if target_area is None:
            continue

        for _ in range(max_iter):
            current_area, grad = body.compute_area_and_gradient(
                mesh, positions=positions, index_map=index_map
            )

            delta = current_area - target_area
            if abs(delta) < tol:
                break

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
            if positions is not None:
                positions = mesh.positions_view()


__all__ = ["enforce_constraint", "constraint_gradients", "constraint_gradients_array"]
