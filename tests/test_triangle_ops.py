import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Mesh
from geometry.geom_io import parse_geometry
from geometry.triangle_ops import (
    barycentric_vertex_areas_from_triangles,
    p1_triangle_shape_gradients,
    triangle_normals,
    triangle_normals_and_areas,
    vertex_unit_normals_from_triangles,
)


def _mesh() -> Mesh:
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


def test_triangle_ops_match_mesh_geometry_arrays():
    mesh = _mesh()
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    assert tri_rows is not None

    normals_ref = mesh.triangle_normals(positions)
    areas_ref = mesh.triangle_areas(positions)
    normals, areas = triangle_normals_and_areas(positions, tri_rows)
    assert np.allclose(normals, normals_ref)
    assert np.allclose(areas, areas_ref)
    assert np.allclose(triangle_normals(positions, tri_rows), normals_ref)

    bary_ref = mesh.barycentric_vertex_areas(positions)
    bary = barycentric_vertex_areas_from_triangles(
        n_verts=len(mesh.vertex_ids), tri_rows=tri_rows, areas=areas
    )
    assert np.allclose(bary, bary_ref)

    vnorm_ref = mesh.vertex_normals(positions)
    vnorm = vertex_unit_normals_from_triangles(
        n_verts=len(mesh.vertex_ids), tri_rows=tri_rows, tri_normals=normals
    )
    assert np.allclose(vnorm, vnorm_ref)


def test_triangle_ops_p1_gradients_match_mesh_cache():
    mesh = _mesh()
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    assert tri_rows is not None

    area_ref, g0_ref, g1_ref, g2_ref, tri_ref = mesh.p1_triangle_shape_gradient_cache(
        positions
    )
    area, g0, g1, g2 = p1_triangle_shape_gradients(positions, tri_rows)

    assert np.array_equal(tri_rows, tri_ref)
    assert np.allclose(area, area_ref)
    assert np.allclose(g0, g0_ref)
    assert np.allclose(g1, g1_ref)
    assert np.allclose(g2, g2_ref)
