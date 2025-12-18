"""Constraint module to pin selected vertices/edges to a plane.

This is an opt-in geometric constraint enforced by projecting vertex positions
onto a plane. It is intended for boundary conditions like "ring stays on a
plane" and is safe to run both during minimization and after discrete mesh
operations (refinement/equiangulation/averaging).

How to use (in input JSON)
--------------------------
Add ``"constraints": ["pin_to_plane"]`` to:

- a vertex options dict to project that vertex onto the plane, or
- an edge options dict to project *both endpoints* of that edge onto the plane.

The plane is configured via global parameters:

- ``pin_to_plane_normal`` (default: ``[0, 0, 1]``)
- ``pin_to_plane_point`` (default: ``[0, 0, 0]``)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("membrane_solver")


def _project_onto_plane(
    pos: np.ndarray, normal: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """Project ``pos`` onto the plane defined by (normal, point)."""
    return pos - np.dot(pos - point, normal) * normal


def _resolve_plane_from_options(
    mesh, options: dict | None
) -> tuple[np.ndarray, np.ndarray] | None:
    gp = getattr(mesh, "global_parameters", None)

    normal_raw = None
    point_raw = None
    if options:
        normal_raw = options.get("pin_to_plane_normal")
        point_raw = options.get("pin_to_plane_point")

    if normal_raw is None and gp is not None:
        normal_raw = gp.get("pin_to_plane_normal")

    if normal_raw is not None:
        normal = np.asarray(normal_raw, dtype=float)
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    norm = float(np.linalg.norm(normal))
    if norm < 1e-15:
        logger.warning("pin_to_plane: normal is near zero; skipping projection.")
        return None
    normal = normal / norm

    if point_raw is None and gp is not None:
        point_raw = gp.get("pin_to_plane_point")

    if point_raw is not None:
        point = np.asarray(point_raw, dtype=float)
    else:
        point = np.array([0.0, 0.0, 0.0], dtype=float)

    return normal, point


def _has_pin_to_plane(options: dict | None) -> bool:
    if not options:
        return False
    constraints = options.get("constraints")
    if constraints is None:
        return False
    if isinstance(constraints, str):
        return constraints == "pin_to_plane"
    if isinstance(constraints, list):
        return "pin_to_plane" in constraints
    return False


def enforce_constraint(mesh, tol: float = 0.0, max_iter: int = 1, **_kwargs) -> None:
    """Project tagged vertices (and endpoints of tagged edges) onto a plane."""
    to_project_vertices: set[int] = set()
    edges_to_project = []

    for vidx, vertex in mesh.vertices.items():
        if _has_pin_to_plane(getattr(vertex, "options", None)):
            to_project_vertices.add(int(vidx))

    for edge in mesh.edges.values():
        if _has_pin_to_plane(getattr(edge, "options", None)):
            edges_to_project.append(edge)

    # Project explicitly tagged vertices (each may optionally provide its own plane).
    for vidx in to_project_vertices:
        vertex = mesh.vertices.get(vidx)
        if vertex is None:
            continue
        resolved = _resolve_plane_from_options(mesh, getattr(vertex, "options", None))
        if resolved is None:
            continue
        normal, point = resolved
        vertex.position[:] = _project_onto_plane(vertex.position, normal, point)

    # Project endpoints of tagged edges. The edge's options (if present) define
    # the plane; otherwise we fall back to global parameters.
    for edge in edges_to_project:
        resolved = _resolve_plane_from_options(mesh, getattr(edge, "options", None))
        if resolved is None:
            continue
        normal, point = resolved
        for vidx in (int(edge.tail_index), int(edge.head_index)):
            vertex = mesh.vertices.get(vidx)
            if vertex is None:
                continue
            vertex.position[:] = _project_onto_plane(vertex.position, normal, point)


__all__ = ["enforce_constraint"]
