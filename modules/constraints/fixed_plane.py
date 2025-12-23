"""Constraint module to keep vertices on a fixed plane.

By default, vertices are projected onto the plane z = 0.

This is a geometric constraint enforced by adjusting vertex positions, so it
is compatible with both minimization steps and discrete mesh operations.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("membrane_solver")


def _project_onto_plane(
    pos: np.ndarray, normal: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """Project a point onto a plane defined by (normal, point)."""
    return pos - np.dot(pos - point, normal) * normal


def enforce_constraint(mesh, tol: float = 0.0, max_iter: int = 1, **_kwargs) -> None:
    """Project all vertices onto a fixed plane.

    The plane is given by the global parameters
    ``fixed_plane_normal`` and ``fixed_plane_point`` when present,
    otherwise defaults to the z=0 plane with normal (0, 0, 1).
    """
    gp = getattr(mesh, "global_parameters", None)

    if gp is not None and gp.get("fixed_plane_normal") is not None:
        normal = np.asarray(gp.get("fixed_plane_normal"), dtype=float)
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    norm = np.linalg.norm(normal)
    if norm < 1e-15:
        logger.warning("fixed_plane: normal is near zero; skipping projection.")
        return
    normal /= norm

    if gp is not None and gp.get("fixed_plane_point") is not None:
        point = np.asarray(gp.get("fixed_plane_point"), dtype=float)
    else:
        point = np.array([0.0, 0.0, 0.0], dtype=float)

    for vertex in mesh.vertices.values():
        if getattr(vertex, "fixed", False):
            continue
        vertex.position[:] = _project_onto_plane(vertex.position, normal, point)


__all__ = ["enforce_constraint"]
