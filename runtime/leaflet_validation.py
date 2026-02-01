"""Validation helpers for leaflet-presence modeling assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry.entities import Mesh
from modules.energy.leaflet_presence import leaflet_absent_vertex_mask


@dataclass(frozen=True)
class LeafletBoundaryIssue:
    """A single triangle that straddles an absent/present leaflet boundary."""

    tri_index: int
    rows: tuple[int, int, int]
    presets: tuple[str, str, str]


def validate_leaflet_absence_topology(mesh: Mesh, global_params) -> None:
    """Validate mesh topology for leaflet absence modeling.

    When a leaflet is marked absent on certain vertex presets (e.g. disk),
    the discrete model can operate in two modes:

    - strict (default): the absent region is treated as a separate patch that
      should not share triangles with present vertices. In this mode, any
      triangle containing a mix of absent/present vertices is rejected.
    - triangles: triangles that touch an absent vertex are treated as belonging
      to the absent region and are masked out of the corresponding leaflet
      energies. This allows shared boundary vertices (rim) without requiring
      duplicate vertex rings.

    Raises
    ------
    ValueError
        If a straddling triangle is detected.
    """
    mode = str(global_params.get("leaflet_out_absence_mode", "strict") or "strict")
    mode = mode.strip().lower()
    if mode in {"triangles", "facet", "facets", "triangle"}:
        return

    mesh.build_position_cache()
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return

    absent_out = leaflet_absent_vertex_mask(mesh, global_params, leaflet="out")
    if not np.any(absent_out):
        return

    tri_abs = absent_out[tri_rows]
    tri_has_abs = np.any(tri_abs, axis=1)
    tri_has_pres = np.any(~tri_abs, axis=1)
    bad = tri_has_abs & tri_has_pres
    if not np.any(bad):
        return

    bad_idxs = np.nonzero(bad)[0]
    examples: list[LeafletBoundaryIssue] = []
    for idx in bad_idxs[:5]:
        rows = tuple(int(x) for x in tri_rows[idx])
        presets: list[str] = []
        for row in rows:
            vid = int(mesh.vertex_ids[row])
            opts = getattr(mesh.vertices[vid], "options", None) or {}
            presets.append(str(opts.get("preset") or ""))
        examples.append(
            LeafletBoundaryIssue(
                tri_index=int(idx),
                rows=rows,
                presets=(presets[0], presets[1], presets[2]),
            )
        )

    msg = (
        "Leaflet absence topology invalid: outer leaflet marked absent on some presets "
        "but mesh contains triangles that straddle absent/present vertices. "
        f"bad_triangles={int(bad_idxs.size)} examples={examples}"
    )
    raise ValueError(msg)


__all__ = ["validate_leaflet_absence_topology", "LeafletBoundaryIssue"]
