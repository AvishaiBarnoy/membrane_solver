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


__all__ = [
    "INTERFACE_PROFILES",
    "SOURCE_INNER_RADIUS",
    "SOURCE_OUTER_RADIUS",
    "build_profiled_fixture",
    "build_scaled_fixture",
]
