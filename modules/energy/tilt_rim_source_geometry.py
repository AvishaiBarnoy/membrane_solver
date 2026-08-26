"""Shared rim selection and circle-frame geometry for tilt source modules."""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh
from geometry.plane_ops import fit_plane_normal


def _pin_to_circle_group(options: dict | None) -> str | None:
    if not options:
        return None
    group = options.get("pin_to_circle_group")
    return "default" if group is None else str(group)


def _selected_boundary_edges(mesh: Mesh, group: str) -> list[int]:
    mesh.build_connectivity_maps()
    boundary_edges = [eid for eid, fs in mesh.edge_to_facets.items() if len(fs) < 2]
    selected: list[int] = []
    for eid in boundary_edges:
        edge = mesh.edges.get(int(eid))
        if edge is None:
            continue
        v0 = mesh.vertices[int(edge.tail_index)]
        v1 = mesh.vertices[int(edge.head_index)]
        if _pin_to_circle_group(v0.options) != group:
            continue
        if _pin_to_circle_group(v1.options) != group:
            continue
        selected.append(int(eid))
    return selected


def _resolve_edge_mode(param_resolver) -> str:
    raw = param_resolver.get(None, "tilt_rim_source_edge_mode")
    mode = str(raw or "boundary").strip().lower()
    return "all" if mode == "all" else "boundary"


def _selected_rim_edges(mesh: Mesh, group: str, *, mode: str) -> list[int]:
    """Return tagged boundary edges, or all tagged edges in ``all`` mode."""
    if mode == "all":
        selected: list[int] = []
        for eid, edge in mesh.edges.items():
            v0 = mesh.vertices[int(edge.tail_index)]
            v1 = mesh.vertices[int(edge.head_index)]
            if _pin_to_circle_group(v0.options) != group:
                continue
            if _pin_to_circle_group(v1.options) != group:
                continue
            selected.append(int(eid))
        return selected
    return _selected_boundary_edges(mesh, group)


def _resolve_center(param_resolver) -> np.ndarray:
    center = param_resolver.get(None, "tilt_rim_source_center")
    if center is None:
        center = [0.0, 0.0, 0.0]
    return np.asarray(center, dtype=float).reshape(3)


def _normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-15:
        return None
    return vec / norm


def _fit_plane_normal(points: np.ndarray) -> np.ndarray | None:
    return fit_plane_normal(points)


def _pin_to_circle_mode(mesh: Mesh, options: dict | None) -> str:
    gp = getattr(mesh, "global_parameters", None)
    raw = None
    if options and options.get("pin_to_circle_mode") is not None:
        raw = options.get("pin_to_circle_mode")
    elif gp is not None and gp.get("pin_to_circle_mode") is not None:
        raw = gp.get("pin_to_circle_mode")
    mode = str(raw or "fixed").strip().lower()
    return "fit" if mode == "fit" else "fixed"


def _pin_to_circle_normal(mesh: Mesh, options: dict | None) -> np.ndarray | None:
    gp = getattr(mesh, "global_parameters", None)
    raw = None
    if options and options.get("pin_to_circle_normal") is not None:
        raw = options.get("pin_to_circle_normal")
    elif gp is not None and gp.get("pin_to_circle_normal") is not None:
        raw = gp.get("pin_to_circle_normal")
    if raw is None:
        return None
    return _normalize(np.asarray(raw, dtype=float).reshape(3))
