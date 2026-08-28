from __future__ import annotations

import numpy as np

from geometry.curvature import compute_curvature_data
from geometry.entities import Edge, Facet, Mesh, Vertex
from modules.energy.bending_utils import _compute_effective_areas


def _closed_tetra_mesh() -> Mesh:
    mesh = Mesh()
    pts = np.array(
        [
            [0.1, 0.2, 0.05],
            [1.1, -0.1, 0.3],
            [0.4, 1.2, -0.2],
            [0.5, 0.4, 1.5],
        ],
        dtype=float,
    )
    for i, p in enumerate(pts):
        mesh.vertices[i] = Vertex(i, p)

    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    edge_map: dict[tuple[int, int], int] = {}
    next_eid = 1
    for fid, (a, b, c) in enumerate(faces):
        e_ids = []
        for tail, head in ((a, b), (b, c), (c, a)):
            key = (min(tail, head), max(tail, head))
            if key not in edge_map:
                edge_map[key] = next_eid
                mesh.edges[next_eid] = Edge(next_eid, tail, head)
                next_eid += 1
            eid = edge_map[key]
            edge = mesh.edges[eid]
            e_ids.append(eid if edge.tail_index == tail else -eid)
        mesh.facets[fid] = Facet(fid, e_ids)

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def _accumulate_vertex_areas(n_verts: int, tri_rows: np.ndarray, va_eff: np.ndarray):
    out = np.zeros(n_verts, dtype=float)
    np.add.at(out, tri_rows[:, 0], va_eff[:, 0])
    np.add.at(out, tri_rows[:, 1], va_eff[:, 1])
    np.add.at(out, tri_rows[:, 2], va_eff[:, 2])
    return out


def test_effective_areas_do_not_trust_curvature_kernel_corner_cache() -> None:
    mesh = _closed_tetra_mesh()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    _k, _a, weights, tri_rows = compute_curvature_data(mesh, positions, index_map)
    _, va0_ref, va1_ref, va2_ref = _compute_effective_areas(
        mesh,
        positions.copy(),
        tri_rows,
        weights,
        index_map,
        cache_token="detached_reference",
    )

    nf = tri_rows.shape[0]
    va0_raw = np.linspace(0.11, 0.14, nf)
    va1_raw = np.linspace(0.21, 0.24, nf)
    va2_raw = np.linspace(0.31, 0.34, nf)
    mesh._curvature_cache["va0_raw"] = va0_raw
    mesh._curvature_cache["va1_raw"] = va1_raw
    mesh._curvature_cache["va2_raw"] = va2_raw

    vertex_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token="raw_cache_reuse",
    )

    assert np.allclose(va0_eff, va0_ref)
    assert np.allclose(va1_eff, va1_ref)
    assert np.allclose(va2_eff, va2_ref)

    va_eff = np.stack([va0_ref, va1_ref, va2_ref], axis=1)
    expected_vertex = _accumulate_vertex_areas(len(mesh.vertex_ids), tri_rows, va_eff)
    assert np.allclose(vertex_eff, expected_vertex)


def test_effective_area_cache_is_invalidated_by_geometry_version() -> None:
    mesh = _closed_tetra_mesh()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    _k, _a, weights, tri_rows = compute_curvature_data(mesh, positions, index_map)
    token = "geometry_version"

    _compute_effective_areas(
        mesh, positions, tri_rows, weights, index_map, cache_token=token
    )

    mesh.vertices[0].position[0] += 0.2
    mesh.increment_version()
    positions = mesh.positions_view()
    _k, _a, weights, tri_rows = compute_curvature_data(mesh, positions, index_map)
    _, va0_active, va1_active, va2_active = _compute_effective_areas(
        mesh, positions, tri_rows, weights, index_map, cache_token=token
    )
    _, va0_ref, va1_ref, va2_ref = _compute_effective_areas(
        mesh,
        positions.copy(),
        tri_rows,
        weights,
        index_map,
        cache_token="geometry_version_reference",
    )

    assert np.allclose(va0_active, va0_ref)
    assert np.allclose(va1_active, va1_ref)
    assert np.allclose(va2_active, va2_ref)


def test_effective_areas_ignore_mismatched_raw_cache_shape() -> None:
    mesh = _closed_tetra_mesh()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    _k, _a, weights, tri_rows = compute_curvature_data(mesh, positions, index_map)

    vertex_ref, va0_ref, va1_ref, va2_ref = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token="baseline_no_raw",
    )

    mesh._curvature_cache["va0_raw"] = np.array([99.0, 99.0], dtype=float)
    mesh._curvature_cache["va1_raw"] = np.array([98.0, 98.0], dtype=float)
    mesh._curvature_cache["va2_raw"] = np.array([97.0, 97.0], dtype=float)

    vertex_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token="mismatch_raw",
    )

    assert np.allclose(vertex_eff, vertex_ref)
    assert np.allclose(va0_eff, va0_ref)
    assert np.allclose(va1_eff, va1_ref)
    assert np.allclose(va2_eff, va2_ref)


def test_effective_areas_triangle_only_mode_matches_full_outputs() -> None:
    mesh = _closed_tetra_mesh()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    _k, _a, weights, tri_rows = compute_curvature_data(mesh, positions, index_map)
    token = "triangle_only_mode"

    vertex_full, va0_full, va1_full, va2_full = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token=token,
    )
    mesh._curvature_cache.pop(f"vertex_areas_eff::{token}", None)

    vertex_none, va0_tri, va1_tri, va2_tri = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token=token,
        compute_vertex_areas=False,
    )

    assert vertex_none is None
    assert np.allclose(va0_tri, va0_full)
    assert np.allclose(va1_tri, va1_full)
    assert np.allclose(va2_tri, va2_full)
    assert f"vertex_areas_eff::{token}" not in mesh._curvature_cache

    vertex_after, va0_after, va1_after, va2_after = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token=token,
    )

    assert np.allclose(vertex_after, vertex_full)
    assert np.allclose(va0_after, va0_full)
    assert np.allclose(va1_after, va1_full)
    assert np.allclose(va2_after, va2_full)
