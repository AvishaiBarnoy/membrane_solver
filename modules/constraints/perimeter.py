import logging
from typing import Dict, Iterable, List

import numpy as np

logger = logging.getLogger("membrane_solver")


def _loop_gradient(mesh, edge_indices: Iterable[int]) -> Dict[int, np.ndarray]:
    grad: Dict[int, np.ndarray] = {}
    for signed_idx in edge_indices:
        edge = mesh.get_edge(signed_idx)
        tail = mesh.vertices[edge.tail_index]
        head = mesh.vertices[edge.head_index]
        vec = head.position - tail.position
        length = np.linalg.norm(vec)
        if length < 1e-12:
            continue
        direction = vec / length
        grad.setdefault(edge.tail_index, np.zeros(3))
        grad.setdefault(edge.head_index, np.zeros(3))
        grad[edge.tail_index] += -direction
        grad[edge.head_index] += direction
    return grad


def enforce_constraint(mesh, tol: float = 1e-10, max_iter: int = 3) -> None:
    """Enforce perimeter constraints defined in ``mesh.global_parameters``.

    Global parameter ``perimeter_constraints`` may be a list of dicts with keys:
    - ``edges``: list of (signed) edge indices forming a loop.
    - ``target_perimeter``: target total length for the loop.
    """
    constraints: List[Dict] = mesh.global_parameters.get("perimeter_constraints", [])
    if not constraints:
        return

    for idx, constraint in enumerate(constraints):
        edges = constraint.get("edges")
        target_perimeter = constraint.get("target_perimeter")
        if not edges or target_perimeter is None:
            continue

        for _ in range(max_iter):
            # Compute current perimeter
            perimeter = 0.0
            for signed_idx in edges:
                edge = mesh.get_edge(signed_idx)
                tail = mesh.vertices[edge.tail_index].position
                head = mesh.vertices[edge.head_index].position
                perimeter += np.linalg.norm(head - tail)

            delta = perimeter - target_perimeter
            if abs(delta) < tol:
                break

            grad = _loop_gradient(mesh, edges)
            norm_sq = sum(np.dot(vec, vec) for vec in grad.values())
            if norm_sq < 1e-18:
                logger.debug(
                    "Perimeter constraint %s skipped due to near-zero gradient.", idx
                )
                break

            lam = delta / (norm_sq + 1e-18)
            logger.debug(
                "Applying perimeter constraint %s: ΔP=%.3e, λ=%.3e", idx, delta, lam
            )

            for vidx, gvec in grad.items():
                vertex = mesh.vertices[vidx]
                if getattr(vertex, "fixed", False):
                    continue
                vertex.position -= lam * gvec


__all__ = ["enforce_constraint"]
