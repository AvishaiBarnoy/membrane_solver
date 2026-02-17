"""Helpers for building triangle-row caches from facet loops."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def triangle_facets_from_loops(
    facet_vertex_loops: Mapping[int, np.ndarray],
) -> list[int]:
    """Return sorted facet ids whose loops are triangles."""
    tri_facets: list[int] = []
    for fid in sorted(facet_vertex_loops):
        loop = facet_vertex_loops[fid]
        if len(loop) == 3:
            tri_facets.append(fid)
    return tri_facets


def triangle_rows_from_loops(
    *,
    tri_facets: list[int],
    facet_vertex_loops: Mapping[int, np.ndarray],
    vertex_index_to_row: Mapping[int, int],
) -> np.ndarray:
    """Build Fortran-contiguous triangle row indices for triangle facets."""
    tri_rows = np.empty((len(tri_facets), 3), dtype=np.int32, order="F")
    for idx, fid in enumerate(tri_facets):
        loop = facet_vertex_loops[fid]
        tri_rows[idx, 0] = vertex_index_to_row[int(loop[0])]
        tri_rows[idx, 1] = vertex_index_to_row[int(loop[1])]
        tri_rows[idx, 2] = vertex_index_to_row[int(loop[2])]
    return tri_rows
