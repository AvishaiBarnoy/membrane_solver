"""Patch-boundary helpers for multi-region meshes.

This module supports workflows where facet options tag regions such as
``disk_patch: top`` / ``disk_patch: bottom``. Patch boundaries are then the
shared edges where adjacent facets have different patch labels.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from geometry.entities import Edge, Facet, Mesh


def _facet_patch_label(
    facet: Facet, *, patch_key: str, patch_values: set[str] | None
) -> str | None:
    label = facet.options.get(patch_key)
    if label is None:
        return None
    if not isinstance(label, str):
        return None
    if patch_values is not None and label not in patch_values:
        return None
    return label


def patch_boundary_edges(
    mesh: Mesh,
    *,
    patch_key: str = "disk_patch",
    patch_values: Iterable[str] | None = None,
    include_mesh_boundary: bool = False,
) -> dict[str, list[Edge]]:
    """Return patch-boundary edges grouped by patch label.

    A patch boundary is an edge shared by facets with different labels under
    ``facet.options[patch_key]``. By default, edges on the mesh boundary
    (only one incident facet) are excluded so “holes” can be handled separately
    via the standard boundary-loop utilities.

    Parameters
    ----------
    mesh:
        Mesh with populated connectivity maps.
    patch_key:
        Facet option key storing patch labels (e.g. ``"disk_patch"``).
    patch_values:
        Optional whitelist of labels to include (others are treated as unlabeled).
    include_mesh_boundary:
        When ``True``, edges on the mesh boundary are included if they are
        incident to a labeled facet.

    Returns
    -------
    dict[str, list[Edge]]
        Mapping ``label -> boundary edges``. If an edge separates two labeled
        patches, it is included in both label groups.
    """
    if not mesh.facets or not mesh.edges:
        return {}

    mesh.build_connectivity_maps()
    allowed = set(patch_values) if patch_values is not None else None

    facet_labels: dict[int, str | None] = {
        fid: _facet_patch_label(facet, patch_key=patch_key, patch_values=allowed)
        for fid, facet in mesh.facets.items()
    }

    grouped: dict[str, list[Edge]] = {}
    for eid, incident_facets in mesh.edge_to_facets.items():
        if not include_mesh_boundary and len(incident_facets) < 2:
            continue

        labels = [facet_labels.get(fid) for fid in incident_facets]
        labels_set = set(labels)
        non_null = {lab for lab in labels_set if lab is not None}
        if not non_null:
            continue

        is_boundary = len(incident_facets) < 2 or len(labels_set) > 1
        if not is_boundary:
            continue

        edge = mesh.edges[eid]
        for lab in non_null:
            grouped.setdefault(lab, []).append(edge)

    return grouped


def patch_boundary_lengths(
    mesh: Mesh,
    *,
    patch_key: str = "disk_patch",
    patch_values: Iterable[str] | None = None,
    include_mesh_boundary: bool = False,
) -> dict[str, float]:
    """Return total patch-boundary length per patch label."""
    groups = patch_boundary_edges(
        mesh,
        patch_key=patch_key,
        patch_values=patch_values,
        include_mesh_boundary=include_mesh_boundary,
    )
    if not groups:
        return {}

    mesh.build_position_cache()
    pos = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row

    lengths: dict[str, float] = {}
    for label, edges in groups.items():
        if not edges:
            continue
        tail_rows = np.array([idx_map[e.tail_index] for e in edges], dtype=int)
        head_rows = np.array([idx_map[e.head_index] for e in edges], dtype=int)
        seg = pos[head_rows] - pos[tail_rows]
        lengths[label] = float(np.linalg.norm(seg, axis=1).sum())
    return lengths
