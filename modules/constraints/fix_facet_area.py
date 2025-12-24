# modules/constraints/fix_facet_area.py
"""Hard facet-area constraints enforced via Lagrange multipliers."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("membrane_solver")


def enforce_constraint(mesh, tol: float = 1e-12, max_iter: int = 5, **_kwargs) -> None:
    """Adjust facets with ``target_area`` using a damped Lagrange step.

    This is intentionally conservative: we project along the area gradient but
    clamp the per-vertex displacement and backtrack until the area error
    decreases, to avoid the blow-ups seen when facets are badly distorted.
    """

    positions = None
    index_map = None
    if getattr(mesh, "facet_vertex_loops", None):
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row

    for facet in mesh.facets.values():
        target_area = facet.options.get("target_area")
        if target_area is None:
            continue

        # Estimate a local length scale to clamp displacements
        if (
            getattr(mesh, "facet_vertex_loops", None)
            and facet.index in mesh.facet_vertex_loops
        ):
            v_ids = mesh.facet_vertex_loops[facet.index].tolist()
        else:
            v_ids = []
            for signed_ei in facet.edge_indices:
                edge = mesh.get_edge(signed_ei)
                tail = edge.tail_index if signed_ei > 0 else edge.head_index
                if not v_ids or v_ids[-1] != tail:
                    v_ids.append(tail)
        if len(v_ids) < 3:
            continue
        if positions is not None and index_map is not None:
            coords = positions[[index_map[vid] for vid in v_ids]]
        else:
            coords = np.array([mesh.vertices[vid].position for vid in v_ids])
        diameter = np.max(
            np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        )
        max_move = 0.1 * diameter if diameter > 0 else 1e-3

        for _ in range(max_iter):
            current_area, grad = facet.compute_area_and_gradient(
                mesh, positions=positions, index_map=index_map
            )
            delta = current_area - target_area
            if abs(delta) < tol:
                break

            norm_sq = sum(np.dot(vec, vec) for vec in grad.values())
            if norm_sq < 1e-18:
                logger.debug(
                    "Facet %s area constraint skipped due to near-zero gradient.",
                    facet.index,
                )
                break

            base_positions = {
                vidx: mesh.vertices[vidx].position.copy() for vidx in grad.keys()
            }
            lam = delta / (norm_sq + 1e-18)

            success = False
            # Backtrack until the area error decreases and per-vertex move is bounded
            for _trial in range(12):
                too_far = False
                for vidx, gvec in grad.items():
                    if getattr(mesh.vertices[vidx], "fixed", False):
                        continue
                    disp = -lam * gvec
                    step_norm = np.linalg.norm(disp)
                    if step_norm > max_move:
                        too_far = True
                        break
                if too_far:
                    lam *= 0.5
                    continue

                for vidx, gvec in grad.items():
                    if getattr(mesh.vertices[vidx], "fixed", False):
                        continue
                    mesh.vertices[vidx].position = base_positions[vidx] - lam * gvec

                mesh.increment_version()
                if positions is not None:
                    positions = mesh.positions_view()

                new_area = facet.compute_area_and_gradient(
                    mesh, positions=positions, index_map=index_map
                )[0]
                if abs(new_area - target_area) < abs(delta):
                    success = True
                    break

                lam *= 0.5
                for vidx in grad.keys():
                    mesh.vertices[vidx].position = base_positions[vidx].copy()
                mesh.increment_version()
                if positions is not None:
                    positions = mesh.positions_view()

            if not success:
                # Restore best-known positions if backtracking failed
                for vidx, pos in base_positions.items():
                    mesh.vertices[vidx].position = pos
                mesh.increment_version()
                if positions is not None:
                    positions = mesh.positions_view()
                break

        logger.debug(
            "Facet %s area constraint applied: target=%.6f, final=%.6f",
            facet.index,
            target_area,
            facet.compute_area_and_gradient(
                mesh, positions=positions, index_map=index_map
            )[0],
        )

        logger.debug(
            "Facet %s area constraint applied: target=%.6f, final=%.6f",
            facet.index,
            target_area,
            facet.compute_area_and_gradient(
                mesh, positions=positions, index_map=index_map
            )[0],
        )


__all__ = ["enforce_constraint"]
