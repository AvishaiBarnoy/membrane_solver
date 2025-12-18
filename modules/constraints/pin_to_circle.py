"""Constraint module to pin selected entities to a circle.

The circle is defined by:

- a plane normal ``pin_to_circle_normal`` (per-entity or global, default ``[0,0,1]``),
- a center point ``pin_to_circle_point`` lying on that plane (default ``[0,0,0]``),
- a radius ``pin_to_circle_radius`` (default ``1.0``).

Attach ``"constraints": ["pin_to_circle"]`` to a vertex or an edge options dict
to have its vertices projected onto the specified circle.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("membrane_solver")


def _normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-15:
        return None
    return vec / norm


def _default_tangent(normal: np.ndarray) -> np.ndarray:
    # Pick any vector not parallel to the normal, then orthogonalize.
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(trial, normal)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    tangent = trial - np.dot(trial, normal) * normal
    tangent = _normalize(tangent)
    if tangent is None:
        tangent = np.array([1.0, 0.0, 0.0], dtype=float)
    return tangent


def _resolve_circle(mesh, options: dict | None):
    gp = getattr(mesh, "global_parameters", None)

    def pick(key: str, default):
        if options and options.get(key) is not None:
            return options.get(key)
        if gp is not None and gp.get(key) is not None:
            return gp.get(key)
        return default

    normal_raw = pick("pin_to_circle_normal", [0.0, 0.0, 1.0])
    center_raw = pick("pin_to_circle_point", [0.0, 0.0, 0.0])
    radius = pick("pin_to_circle_radius", 1.0)

    normal = np.asarray(normal_raw, dtype=float)
    normal = _normalize(normal)
    if normal is None:
        logger.warning("pin_to_circle: normal is near zero; skipping projection.")
        return None

    center = np.asarray(center_raw, dtype=float)
    radius = float(radius)
    if radius <= 0.0:
        logger.warning("pin_to_circle: radius must be positive; skipping projection.")
        return None

    return normal, center, radius


def _project_point_to_circle(
    pos: np.ndarray, normal: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
    # Project onto plane.
    pos_plane = pos - np.dot(pos - center, normal) * normal
    offset = pos_plane - center
    tangent = _normalize(offset)
    if tangent is None:
        tangent = _default_tangent(normal)
    return center + radius * tangent


def _entity_has_constraint(options: dict | None) -> bool:
    if not options:
        return False
    constraints = options.get("constraints")
    if constraints is None:
        return False
    if isinstance(constraints, str):
        return constraints == "pin_to_circle"
    if isinstance(constraints, list):
        return "pin_to_circle" in constraints
    return False


def enforce_constraint(mesh, **_kwargs):
    tagged_vertices = [
        v
        for v in mesh.vertices.values()
        if _entity_has_constraint(getattr(v, "options", None))
    ]
    tagged_edges = [
        e
        for e in mesh.edges.values()
        if _entity_has_constraint(getattr(e, "options", None))
    ]

    for vertex in tagged_vertices:
        params = _resolve_circle(mesh, getattr(vertex, "options", None))
        if params is None:
            continue
        normal, center, radius = params
        vertex.position[:] = _project_point_to_circle(
            vertex.position, normal, center, radius
        )

    for edge in tagged_edges:
        params = _resolve_circle(mesh, getattr(edge, "options", None))
        if params is None:
            continue
        normal, center, radius = params
        for vidx in (int(edge.tail_index), int(edge.head_index)):
            vertex = mesh.vertices.get(vidx)
            if vertex is None:
                continue
            vertex.position[:] = _project_point_to_circle(
                vertex.position, normal, center, radius
            )


__all__ = ["enforce_constraint"]
