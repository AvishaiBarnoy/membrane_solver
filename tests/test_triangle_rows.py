import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from geometry.triangle_rows import triangle_facets_from_loops, triangle_rows_from_loops


def _mesh():
    mesh = parse_geometry(
        {
            "vertices": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            "edges": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
            "faces": [[0, 1, "r4"], [4, 2, 3]],
        }
    )
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def test_triangle_row_helpers_match_mesh_triangle_cache():
    mesh = _mesh()
    tri_rows_ref, tri_facets_ref = mesh.triangle_row_cache()
    assert tri_rows_ref is not None

    tri_facets = triangle_facets_from_loops(mesh.facet_vertex_loops)
    tri_rows = triangle_rows_from_loops(
        tri_facets=tri_facets,
        facet_vertex_loops=mesh.facet_vertex_loops,
        vertex_index_to_row=mesh.vertex_index_to_row,
    )

    assert tri_facets == tri_facets_ref
    assert np.array_equal(tri_rows, tri_rows_ref)
    assert tri_rows.flags["F_CONTIGUOUS"]
