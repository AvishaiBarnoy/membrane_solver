import numpy as np

from geometry.triangle_rows import triangle_facets_from_loops, triangle_rows_from_loops
from tests.sample_meshes import parsed_two_triangle_square_mesh as _mesh


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
