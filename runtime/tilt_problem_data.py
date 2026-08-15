"""Immutable dense-array inputs for leaflet tilt relaxation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry.entities import _fast_cross
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)


@dataclass(frozen=True)
class LeafletTiltProblemData:
    fixed_mask_in: np.ndarray
    fixed_mask_out: np.ndarray
    fixed_values_in: np.ndarray | None
    fixed_values_out: np.ndarray | None
    vertex_areas_in: np.ndarray
    vertex_areas_out: np.ndarray


def _tilt_vertex_areas_from_triangles(
    *, n_vertices: int, tri_rows: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Return barycentric vertex areas for the supplied triangle rows."""
    tri_pos = positions[tri_rows]
    triangle_areas = 0.5 * np.linalg.norm(
        _fast_cross(tri_pos[:, 1] - tri_pos[:, 0], tri_pos[:, 2] - tri_pos[:, 0]),
        axis=1,
    )
    vertex_areas = np.zeros(n_vertices, dtype=float)
    area_thirds = triangle_areas / 3.0
    np.add.at(vertex_areas, tri_rows[:, 0], area_thirds)
    np.add.at(vertex_areas, tri_rows[:, 1], area_thirds)
    np.add.at(vertex_areas, tri_rows[:, 2], area_thirds)
    return vertex_areas


def build_leaflet_tilt_problem_data(
    *,
    mesh,
    global_params,
    positions: np.ndarray,
    tri_rows: np.ndarray,
    fixed_mask_in: np.ndarray,
    fixed_mask_out: np.ndarray,
    tilts_in: np.ndarray,
    tilts_out: np.ndarray,
) -> LeafletTiltProblemData:
    """Assemble read-only leaflet inputs without mutating mesh state."""
    areas_in = mesh.barycentric_vertex_areas(positions=positions)
    absent_out = leaflet_absent_vertex_mask(mesh, global_params, leaflet="out")
    if not np.any(absent_out):
        areas_out = areas_in
    else:
        present_triangles = leaflet_present_triangle_mask(
            mesh, tri_rows, absent_vertex_mask=absent_out
        )
        outer_rows = tri_rows[present_triangles] if present_triangles.size else tri_rows
        areas_out = (
            np.zeros(len(mesh.vertex_ids), dtype=float)
            if outer_rows.size == 0
            else _tilt_vertex_areas_from_triangles(
                n_vertices=len(mesh.vertex_ids),
                tri_rows=outer_rows,
                positions=positions,
            )
        )

    fixed_values_in = tilts_in[fixed_mask_in].copy() if np.any(fixed_mask_in) else None
    fixed_values_out = (
        tilts_out[fixed_mask_out].copy() if np.any(fixed_mask_out) else None
    )
    return LeafletTiltProblemData(
        fixed_mask_in,
        fixed_mask_out,
        fixed_values_in,
        fixed_values_out,
        areas_in,
        areas_out,
    )
