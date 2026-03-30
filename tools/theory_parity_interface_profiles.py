#!/usr/bin/env python3
"""Reusable near-edge interface profiles for theory-parity fixtures."""

from __future__ import annotations

import copy
from typing import Any

SOURCE_INNER_RADIUS = 1.0
SOURCE_OUTER_RADIUS = 4.666666666666668

INTERFACE_PROFILES: dict[str, tuple[float, float] | None] = {
    "coarse": None,
    "i50": (0.8, 2.8),
    "i60": (0.76, 2.6),
    "tight": (0.6, 0.8),
    "near_edge_v1": (0.76, 2.6),
    "default_lo": (0.776, 2.68),
    "default": (0.772, 2.66),
    "default_hi": (0.771, 2.655),
    "physical_edge_family_lo": (0.78, 2.7),
    "physical_edge_primary_v1": (0.76, 2.6),
    "physical_edge_family_hi": (0.758, 2.6),
}


def _scale_ring(
    vertices: list[list[Any]], source_radius: float, target_radius: float
) -> None:
    """Scale all vertices on one source radius onto `target_radius`."""
    for vertex in vertices:
        x = float(vertex[0])
        y = float(vertex[1])
        radius = (x * x + y * y) ** 0.5
        if radius <= 0.0 or abs(radius - float(source_radius)) >= 1.0e-9:
            continue
        scale = float(target_radius) / radius
        vertex[0] = x * scale
        vertex[1] = y * scale


def _vertex_radius(vertex: list[Any]) -> float:
    """Return cylindrical radius for one YAML vertex record."""
    x = float(vertex[0])
    y = float(vertex[1])
    return float((x * x + y * y) ** 0.5)


def _find_ring_vertex_ids(vertices: list[list[Any]], radius: float) -> list[int]:
    """Return all vertex ids that lie on the given ring radius."""
    out: list[int] = []
    for vid, vertex in enumerate(vertices):
        if abs(_vertex_radius(vertex) - float(radius)) <= 1.0e-9:
            out.append(int(vid))
    return out


def _find_group_vertex_ids(vertices: list[list[Any]], group: str) -> list[int]:
    """Return all vertex ids tagged with one rim-slope match group."""
    tagged_rows: list[int] = []
    for vid, vertex in enumerate(vertices):
        opts = vertex[3] if len(vertex) > 3 and isinstance(vertex[3], dict) else {}
        preset = str(opts.get("preset") or "")
        tagged = str(opts.get("rim_slope_match_group") or "")
        if tagged == str(group):
            tagged_rows.append(int(vid))
            continue
        if str(group) == "disk" and preset == "disk":
            tagged_rows.append(int(vid))
            continue
        if str(group) == "rim" and preset == "rim":
            tagged_rows.append(int(vid))
    if not tagged_rows:
        return []
    if str(group) == "disk":
        radii = [_vertex_radius(vertices[int(vid)]) for vid in tagged_rows]
        boundary_radius = max(radii)
        return [
            int(vid)
            for vid in tagged_rows
            if abs(_vertex_radius(vertices[int(vid)]) - float(boundary_radius))
            <= 1.0e-9
        ]
    return tagged_rows


def _annulus_edges(inner_ids: list[int], outer_ids: list[int]) -> list[list[int]]:
    """Return the regular staggered annulus edge stencil between equal rings."""
    if len(inner_ids) != len(outer_ids):
        raise ValueError("annulus edge generation requires equal ring counts")
    n = len(inner_ids)
    edges: list[list[int]] = []
    for i in range(n):
        inner_i = int(inner_ids[i])
        inner_next = int(inner_ids[(i + 1) % n])
        outer_i = int(outer_ids[i])
        edges.append([inner_next, outer_i])
        edges.append([inner_i, outer_i])
    return edges


def _ring_cycle_edges(ring_ids: list[int]) -> list[list[int]]:
    """Return the cycle edges closing one ring."""
    n = len(ring_ids)
    return [[int(ring_ids[i]), int(ring_ids[(i + 1) % n])] for i in range(n)]


def _edge_token(
    edge_lookup: dict[tuple[int, int], int], start: int, end: int
) -> int | str:
    """Return the oriented edge token for one directed edge."""
    key = (int(start), int(end))
    idx = edge_lookup.get(key)
    if idx is not None:
        return int(idx)
    rev = edge_lookup.get((int(end), int(start)))
    if rev is None:
        raise ValueError(f"missing edge for directed pair ({start}, {end})")
    return f"r{int(rev)}"


def _append_edges(
    edges: list[list[int]], new_edges: list[list[int]]
) -> dict[tuple[int, int], int]:
    """Append edges and return an undirected lookup into the combined list."""
    edges.extend(new_edges)
    lookup: dict[tuple[int, int], int] = {}
    for idx, edge in enumerate(edges):
        lookup[(int(edge[0]), int(edge[1]))] = int(idx)
    return lookup


def _annulus_faces(
    *,
    inner_ids: list[int],
    outer_ids: list[int],
    edge_lookup: dict[tuple[int, int], int],
) -> list[list[int | str]]:
    """Return the regular 2-triangle-per-sector annulus face stencil."""
    if len(inner_ids) != len(outer_ids):
        raise ValueError("annulus face generation requires equal ring counts")
    n = len(inner_ids)
    faces: list[list[int | str]] = []
    for i in range(n):
        i0 = int(inner_ids[i])
        i1 = int(inner_ids[(i + 1) % n])
        o0 = int(outer_ids[i])
        o1 = int(outer_ids[(i + 1) % n])
        faces.append(
            [
                _edge_token(edge_lookup, i0, i1),
                _edge_token(edge_lookup, i1, o0),
                _edge_token(edge_lookup, o0, i0),
            ]
        )
        faces.append(
            [
                _edge_token(edge_lookup, o0, i1),
                _edge_token(edge_lookup, i1, o1),
                _edge_token(edge_lookup, o1, o0),
            ]
        )
    return faces


def _copy_scaled_ring(
    *,
    vertices: list[list[Any]],
    source_ids: list[int],
    target_radius: float,
    options_transform,
) -> list[list[Any]]:
    """Return copied ring vertices scaled to a new radius."""
    out: list[list[Any]] = []
    for vid in source_ids:
        vertex = copy.deepcopy(vertices[int(vid)])
        radius = _vertex_radius(vertex)
        if radius <= 0.0:
            raise ValueError("trace-ring source radius must be positive")
        scale = float(target_radius) / float(radius)
        vertex[0] = float(vertex[0]) * scale
        vertex[1] = float(vertex[1]) * scale
        vertex[2] = float(vertex[2])
        opts = dict(
            vertex[3] if len(vertex) > 3 and isinstance(vertex[3], dict) else {}
        )
        vertex[3] = options_transform(opts)
        out.append(vertex)
    return out


def build_scaled_fixture(
    *, base_doc: dict[str, Any], label: str, inner_radius: float, outer_radius: float
) -> dict[str, Any]:
    """Return a fixture copy with the interface rings moved inward."""
    doc = copy.deepcopy(base_doc)
    _scale_ring(doc["vertices"], SOURCE_INNER_RADIUS, float(inner_radius))
    _scale_ring(doc["vertices"], SOURCE_OUTER_RADIUS, float(outer_radius))
    gp = dict(doc.get("global_parameters") or {})
    gp["theory_parity_lane"] = str(label)
    doc["global_parameters"] = gp
    return doc


def build_profiled_fixture(
    *, base_doc: dict[str, Any], profile: str, lane: str | None = None
) -> dict[str, Any]:
    """Return a fixture copy for one named near-edge interface profile."""
    key = str(profile).strip()
    if key not in INTERFACE_PROFILES:
        raise ValueError(f"unknown interface profile: {profile}")
    radii = INTERFACE_PROFILES[key]
    label = str(lane or key)
    if radii is None:
        doc = copy.deepcopy(base_doc)
        gp = dict(doc.get("global_parameters") or {})
        gp["theory_parity_lane"] = label
        doc["global_parameters"] = gp
        return doc
    inner_radius, outer_radius = radii
    return build_scaled_fixture(
        base_doc=base_doc,
        label=label,
        inner_radius=float(inner_radius),
        outer_radius=float(outer_radius),
    )


def build_trace_ring_fixture(
    *,
    base_doc: dict[str, Any],
    label: str,
    trace_radius: float,
    planar_geometry: bool = False,
) -> dict[str, Any]:
    """Return a fixture copy with an explicit trace ring inserted at ``R + eps``."""
    doc = copy.deepcopy(base_doc)
    vertices = doc["vertices"]
    edges = [list(edge) for edge in doc["edges"]]
    faces = [list(face) for face in doc["faces"]]

    disk_ring = _find_group_vertex_ids(vertices, "disk")
    rim_ring = _find_group_vertex_ids(vertices, "rim")
    if not disk_ring or not rim_ring:
        raise ValueError(
            "trace-ring builder requires disk and rim rings in the fixture"
        )
    if len(disk_ring) != len(rim_ring):
        raise ValueError("disk/rim ring sizes must match for trace-ring insertion")
    if not (float(7.0 / 15.0) < float(trace_radius) < float(SOURCE_INNER_RADIUS)):
        raise ValueError("trace_radius must lie strictly between R and the current rim")

    def _new_ring_options(opts: dict[str, Any]) -> dict[str, Any]:
        out = dict(opts)
        out["preset"] = "rim"
        out["rim_slope_match_group"] = "rim"
        if planar_geometry:
            constraints = list(out.get("constraints") or [])
            if "pin_to_plane" not in constraints:
                constraints.append("pin_to_plane")
            out["constraints"] = constraints
        else:
            constraints = [
                c for c in list(out.get("constraints") or []) if c != "pin_to_plane"
            ]
            if constraints:
                out["constraints"] = constraints
            else:
                out.pop("constraints", None)
        return out

    def _old_rim_options(opts: dict[str, Any]) -> dict[str, Any]:
        out = dict(opts)
        if out.get("preset") == "rim":
            out.pop("preset", None)
        if out.get("rim_slope_match_group") == "rim":
            out.pop("rim_slope_match_group", None)
        return out

    new_ring = _copy_scaled_ring(
        vertices=vertices,
        source_ids=rim_ring,
        target_radius=float(trace_radius),
        options_transform=_new_ring_options,
    )
    new_ring_ids = list(range(len(vertices), len(vertices) + len(new_ring)))
    vertices.extend(new_ring)

    for vid in rim_ring:
        opts = dict(
            vertices[int(vid)][3]
            if len(vertices[int(vid)]) > 3 and isinstance(vertices[int(vid)][3], dict)
            else {}
        )
        vertices[int(vid)][3] = _old_rim_options(opts)

    new_edges: list[list[int]] = []
    new_edges.extend(_ring_cycle_edges(new_ring_ids))
    new_edges.extend(_annulus_edges(disk_ring, new_ring_ids))
    new_edges.extend(_annulus_edges(new_ring_ids, rim_ring))
    edge_lookup = _append_edges(edges, new_edges)

    disk_new_faces = _annulus_faces(
        inner_ids=disk_ring, outer_ids=new_ring_ids, edge_lookup=edge_lookup
    )
    new_rim_faces = _annulus_faces(
        inner_ids=new_ring_ids, outer_ids=rim_ring, edge_lookup=edge_lookup
    )
    disk_rim_face_start = 36
    disk_rim_face_end = 60
    updated_faces = (
        faces[:disk_rim_face_start]
        + disk_new_faces
        + new_rim_faces
        + faces[disk_rim_face_end:]
    )
    doc["edges"] = edges
    doc["faces"] = updated_faces

    gp = dict(doc.get("global_parameters") or {})
    gp["theory_parity_lane"] = str(label)
    doc["global_parameters"] = gp
    return doc


__all__ = [
    "INTERFACE_PROFILES",
    "SOURCE_INNER_RADIUS",
    "SOURCE_OUTER_RADIUS",
    "build_trace_ring_fixture",
    "build_profiled_fixture",
    "build_scaled_fixture",
]
