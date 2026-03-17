import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex
from geometry.tangent_transport import (
    edge_transport_pairs,
    minimal_rotation_transport,
    transport_vectors,
    transport_vertex_tilts_to_triangle_planes,
    triangle_plane_transport_data,
)


def _build_single_triangle_mesh() -> Mesh:
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([0.0, 1.0, 0.0])),
    }
    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 0),
    }
    mesh.facets = {0: Facet(0, edge_indices=[1, 2, 3])}
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def test_minimal_rotation_transport_is_identity_for_equal_normals() -> None:
    src = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    rot = minimal_rotation_transport(src, src)
    vec = np.array([[1.0, 2.0, 0.0], [1.0, 0.0, 3.0]])
    got = transport_vectors(vec, rot)
    assert np.allclose(got, vec, atol=1e-12, rtol=1e-12)


def test_minimal_rotation_transport_maps_source_normal_to_destination() -> None:
    src = np.array([[0.0, 0.0, 1.0]])
    dst = np.array([[0.0, 1.0, 0.0]])
    rot = minimal_rotation_transport(src, dst)
    mapped = transport_vectors(src, rot)
    assert np.allclose(mapped, dst, atol=1e-12, rtol=1e-12)


def test_minimal_rotation_transport_handles_antiparallel_normals_stably() -> None:
    src = np.array([[0.0, 0.0, 1.0]])
    dst = np.array([[0.0, 0.0, -1.0]])
    vec = np.array([[1.0, 0.0, 0.0]])
    rot = minimal_rotation_transport(src, dst)
    mapped_normal = transport_vectors(src, rot)
    mapped_vec = transport_vectors(vec, rot)
    assert np.allclose(mapped_normal, dst, atol=1e-12, rtol=1e-12)
    assert np.isclose(np.linalg.norm(mapped_vec[0]), 1.0, atol=1e-12)
    assert np.isclose(np.dot(mapped_vec[0], dst[0]), 0.0, atol=1e-12)


def test_transport_vertex_tilts_to_triangle_planes_matches_planar_identity() -> None:
    mesh = _build_single_triangle_mesh()
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    normals = mesh.vertex_normals(positions)
    tilts = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    t0, t1, t2, tri_normals = transport_vertex_tilts_to_triangle_planes(
        positions, tri_rows, normals, tilts
    )
    assert np.allclose(t0[0], tilts[0], atol=1e-12, rtol=1e-12)
    assert np.allclose(t1[0], tilts[1], atol=1e-12, rtol=1e-12)
    assert np.allclose(t2[0], tilts[2], atol=1e-12, rtol=1e-12)
    assert np.allclose(tri_normals[0], np.array([0.0, 0.0, 1.0]), atol=1e-12)


def test_transport_vectors_preserves_norm_and_tangency_to_destination_plane() -> None:
    src = np.array([[0.0, 0.0, 1.0]])
    dst = np.array([[0.0, 1.0, 0.0]])
    vec = np.array([[1.0, 0.0, 0.0]])
    rot = minimal_rotation_transport(src, dst)
    got = transport_vectors(vec, rot)
    assert np.isclose(np.linalg.norm(got[0]), np.linalg.norm(vec[0]), atol=1e-12)
    assert np.isclose(np.dot(got[0], dst[0]), 0.0, atol=1e-12)


def test_edge_transport_pairs_returns_sorted_unique_row_pairs() -> None:
    mesh = _build_single_triangle_mesh()
    got = edge_transport_pairs(mesh)
    want = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)
    assert np.array_equal(got, want)


def test_triangle_plane_transport_data_reuses_cache_only_for_live_mesh_geometry() -> (
    None
):
    mesh = _build_single_triangle_mesh()
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()

    cached_a = triangle_plane_transport_data(
        mesh, positions, tri_rows, cache_tag="unit"
    )
    cached_b = triangle_plane_transport_data(
        mesh, positions, tri_rows, cache_tag="unit"
    )
    assert cached_a is cached_b

    positions_copy = positions.copy()
    uncached = triangle_plane_transport_data(
        mesh, positions_copy, tri_rows, cache_tag="unit"
    )
    assert uncached is not cached_a
    assert np.allclose(uncached["tri_normals"], cached_a["tri_normals"], atol=1e-12)
