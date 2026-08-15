from __future__ import annotations

import numpy as np

from runtime.tilt_problem_data import _tilt_vertex_areas_from_triangles


def test_triangle_vertex_areas_are_barycentric_and_accumulate_shared_rows() -> None:
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    triangle_rows = np.asarray([[0, 1, 2], [1, 3, 2]])

    areas = _tilt_vertex_areas_from_triangles(
        n_vertices=4,
        tri_rows=triangle_rows,
        positions=positions,
    )

    assert np.allclose(areas, [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
