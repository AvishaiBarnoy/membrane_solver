from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.curvature import compute_curvature_data
from geometry.entities import Edge, Facet, Mesh, Vertex
from modules.energy.bending import _compute_effective_areas


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


def test_effective_areas_reuse_cached_raw_triangle_areas() -> None:
    mesh = _closed_tetra_mesh()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    _k, _a, weights, tri_rows = compute_curvature_data(mesh, positions, index_map)
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

    assert np.allclose(va0_eff, va0_raw)
    assert np.allclose(va1_eff, va1_raw)
    assert np.allclose(va2_eff, va2_raw)

    va_eff = np.stack([va0_raw, va1_raw, va2_raw], axis=1)
    expected_vertex = _accumulate_vertex_areas(len(mesh.vertex_ids), tri_rows, va_eff)
    assert np.allclose(vertex_eff, expected_vertex)


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
