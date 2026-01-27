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

Optional modes:
- ``pin_to_plane_mode``: ``"fixed"`` (default), ``"slide"``, or ``"fit"``.
  - ``"fixed"`` projects onto the specified plane.
  - ``"slide"`` keeps the normal fixed but fits the plane point to the tagged
    group’s centroid (lets the plane translate along its normal).
  - ``"fit"`` fits both normal and point from the tagged group (PCA normal).
- ``pin_to_plane_group``: group name used to fit a shared plane for ``slide``/``fit``.
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


def _mode_from_options(mesh, options: dict | None) -> str:
    gp = getattr(mesh, "global_parameters", None)
    raw = None
    if options and options.get("pin_to_plane_mode") is not None:
        raw = options.get("pin_to_plane_mode")
    elif gp is not None and gp.get("pin_to_plane_mode") is not None:
        raw = gp.get("pin_to_plane_mode")
    mode = str(raw or "fixed").lower()
    if mode == "fit":
        return "fit"
    if mode in {"slide", "normal", "normal_only", "slide_normal"}:
        return "slide"
    return "fixed"


def _group_from_options(options: dict | None) -> str:
    if not options:
        return "default"
    group = options.get("pin_to_plane_group")
    return "default" if group is None else str(group)


def _resolve_normal_from_options(mesh, options: dict | None) -> np.ndarray | None:
    gp = getattr(mesh, "global_parameters", None)
    normal_raw = None
    if options and options.get("pin_to_plane_normal") is not None:
        normal_raw = options.get("pin_to_plane_normal")
    elif gp is not None and gp.get("pin_to_plane_normal") is not None:
        normal_raw = gp.get("pin_to_plane_normal")
    if normal_raw is None:
        return None
    normal = np.asarray(normal_raw, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-15:
        return None
    return normal / norm


def _fit_plane_normal(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    centroid = np.mean(points, axis=0)
    X = points - centroid
    try:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1, :]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-15:
        return None
    return normal / norm


def enforce_constraint(mesh, tol: float = 0.0, max_iter: int = 1, **_kwargs) -> None:
    """Project tagged vertices (and endpoints of tagged edges) onto a plane."""
    _ = tol, max_iter
    to_project_vertices: list[int] = []
    edges_to_project = []
    group_vertices: dict[str, set[int]] = {}
    group_mode: dict[str, str] = {}
    group_normal: dict[str, np.ndarray | None] = {}

    def note_group(group: str, mode: str, normal: np.ndarray | None, vidx: int) -> None:
        group_vertices.setdefault(group, set()).add(int(vidx))
        if group not in group_mode:
            group_mode[group] = mode
        elif group_mode[group] != "fit" and mode == "fit":
            group_mode[group] = "fit"
        if normal is not None and group_normal.get(group) is None:
            group_normal[group] = normal

    for vidx, vertex in mesh.vertices.items():
        opts = getattr(vertex, "options", None)
        if not _has_pin_to_plane(opts):
            continue
        mode = _mode_from_options(mesh, opts)
        if mode == "fixed":
            to_project_vertices.append(int(vidx))
            continue
        group = _group_from_options(opts)
        normal = _resolve_normal_from_options(mesh, opts)
        note_group(group, mode, normal, int(vidx))

    for edge in mesh.edges.values():
        opts = getattr(edge, "options", None)
        if not _has_pin_to_plane(opts):
            continue
        mode = _mode_from_options(mesh, opts)
        if mode == "fixed":
            edges_to_project.append(edge)
            continue
        group = _group_from_options(opts)
        normal = _resolve_normal_from_options(mesh, opts)
        note_group(group, mode, normal, int(edge.tail_index))
        note_group(group, mode, normal, int(edge.head_index))

    # Fixed-mode projections (per-entity planes).
    for vidx in to_project_vertices:
        vertex = mesh.vertices.get(vidx)
        if vertex is None:
            continue
        resolved = _resolve_plane_from_options(mesh, getattr(vertex, "options", None))
        if resolved is None:
            continue
        normal, point = resolved
        vertex.position[:] = _project_onto_plane(vertex.position, normal, point)

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

    # Group-based planes for slide/fit modes.
    for group, vids in group_vertices.items():
        if not vids:
            continue
        points = np.array([mesh.vertices[v].position for v in vids], dtype=float)
        mode = group_mode.get(group, "fixed")
        normal = group_normal.get(group)
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
        if mode == "fit":
            fitted = _fit_plane_normal(points)
            if fitted is not None:
                normal = fitted
        norm = float(np.linalg.norm(normal))
        if norm < 1e-15:
            logger.warning("pin_to_plane: normal is near zero; skipping projection.")
            continue
        normal = normal / norm
        point = np.mean(points, axis=0)
        for vidx in vids:
            vertex = mesh.vertices.get(int(vidx))
            if vertex is None or getattr(vertex, "fixed", False):
                continue
            vertex.position[:] = _project_onto_plane(vertex.position, normal, point)


__all__ = ["enforce_constraint"]
