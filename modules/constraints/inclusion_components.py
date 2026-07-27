"""Disconnected inclusion-boundary discovery for multi-disc constraints.

The current rim and contact operators consume one named vertex group.  A group
may, however, contain several disconnected closed rims.  This module provides
the topology-aware split needed to give each inclusion its own local frame
without changing the existing one-disc operator behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from geometry.entities import Mesh

logger = logging.getLogger("membrane_solver")


@dataclass(frozen=True)
class InclusionBoundaryComponent:
    """One edge-connected subset of a tagged inclusion boundary."""

    vertex_ids: np.ndarray
    rows: np.ndarray
    center: np.ndarray


def _matching_vertex_ids(
    mesh: Mesh, *, group: str, option_keys: tuple[str, ...]
) -> list[int]:
    vertex_ids: list[int] = []
    for raw_vid in mesh.vertex_ids:
        vid = int(raw_vid)
        options = getattr(mesh.vertices[vid], "options", None) or {}
        if any(options.get(key) == group for key in option_keys):
            vertex_ids.append(vid)
    return vertex_ids


def _edge_connected_components(mesh: Mesh, vertex_ids: list[int]) -> list[list[int]]:
    selected = set(vertex_ids)
    adjacency = {vid: set() for vid in vertex_ids}
    mesh.build_connectivity_maps()

    for vid in vertex_ids:
        for edge_id in mesh.vertex_to_edges.get(vid, ()):
            edge = mesh.edges[int(edge_id)]
            other = (
                int(edge.head_index)
                if int(edge.tail_index) == vid
                else int(edge.tail_index)
            )
            if other in selected:
                adjacency[vid].add(other)

    components: list[list[int]] = []
    unseen = set(vertex_ids)
    while unseen:
        seed = min(unseen)
        stack = [seed]
        unseen.remove(seed)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = sorted(adjacency[current] & unseen, reverse=True)
            for neighbor in neighbors:
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def collect_group_components(
    mesh: Mesh,
    *,
    group: str,
    option_keys: tuple[str, ...] = (
        "rim_slope_match_group",
        "tilt_thetaB_group",
        "tilt_thetaB_group_in",
    ),
) -> tuple[InclusionBoundaryComponent, ...]:
    """Return stable edge-connected components for an inclusion boundary group.

    Results are ordered by the smallest vertex ID in each component.  The cache
    includes geometry, topology, and row-binding versions because component
    centers are derived from the current dense position array.
    """
    mesh.build_position_cache()
    normalized_keys = tuple(str(key) for key in option_keys)
    cache_key = (
        int(mesh._version),
        int(mesh._topology_version),
        int(mesh._vertex_ids_version),
        str(group),
        normalized_keys,
    )
    cache_attr = "_inclusion_boundary_components_cache"
    cached = getattr(mesh, cache_attr, None)
    if isinstance(cached, dict):
        value = cached.get(cache_key)
        if value is not None:
            return value
    else:
        cached = {}

    vertex_ids = _matching_vertex_ids(
        mesh, group=str(group), option_keys=normalized_keys
    )
    positions = mesh.positions_view()
    components: list[InclusionBoundaryComponent] = []
    for component_vertex_ids in _edge_connected_components(mesh, vertex_ids):
        ids = np.asarray(component_vertex_ids, dtype=int)
        rows = np.asarray(
            [mesh.vertex_index_to_row[int(vid)] for vid in ids], dtype=int
        )
        center = np.mean(positions[rows], axis=0)
        components.append(
            InclusionBoundaryComponent(
                vertex_ids=ids,
                rows=rows,
                center=np.asarray(center, dtype=float),
            )
        )

    result = tuple(components)
    cached[cache_key] = result
    if len(cached) > 16:
        cached = dict(list(cached.items())[-8:])
    setattr(mesh, cache_attr, cached)
    return result


def warn_if_disconnected_group_uses_single_frame(
    mesh: Mesh,
    *,
    group: str,
    operator: str,
    option_keys: tuple[str, ...] = (
        "rim_slope_match_group",
        "tilt_thetaB_group",
        "tilt_thetaB_group_in",
    ),
) -> tuple[InclusionBoundaryComponent, ...]:
    """Warn once when a legacy single-frame operator sees multiple rims."""
    components = collect_group_components(
        mesh,
        group=group,
        option_keys=option_keys,
    )
    if len(components) <= 1:
        return components

    warning_key = (
        int(mesh._topology_version),
        int(mesh._vertex_ids_version),
        str(operator),
        str(group),
        tuple(str(key) for key in option_keys),
    )
    cache_attr = "_multi_component_single_frame_warnings"
    warned = getattr(mesh, cache_attr, None)
    if not isinstance(warned, set):
        warned = set()
    if warning_key not in warned:
        logger.warning(
            "%s received group %r with %d disconnected components; "
            "the legacy operator uses one global center/frame, so this "
            "multi-inclusion result is not physically interpretable.",
            operator,
            group,
            len(components),
        )
        warned.add(warning_key)
        setattr(mesh, cache_attr, warned)
    return components


__all__ = [
    "InclusionBoundaryComponent",
    "collect_group_components",
    "warn_if_disconnected_group_uses_single_frame",
]
