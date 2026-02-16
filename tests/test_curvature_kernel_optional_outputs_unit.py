from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fortran_kernels.loader as loader
from geometry.curvature import compute_curvature_data
from geometry.entities import Edge, Facet, Mesh, Vertex


def _single_triangle_mesh() -> Mesh:
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
    mesh.facets = {0: Facet(0, [1, 2, 3])}
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def test_curvature_kernel_optional_area_outputs_are_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _single_triangle_mesh()
    pos = mesh.positions_view()
    idx = mesh.vertex_index_to_row

    def _kernel(
        pos_in, tri_in, k_vecs, vertex_areas, weights, zero_based, va0, va1, va2
    ):
        assert zero_based == 1
        assert pos_in.flags["F_CONTIGUOUS"]
        assert tri_in.flags["F_CONTIGUOUS"]
        k_vecs[:] = 7.0
        vertex_areas[:] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        weights[:] = 0.5
        va0[:] = 0.11
        va1[:] = 0.22
        va2[:] = 0.33

    monkeypatch.setattr(
        loader,
        "get_tilt_curvature_kernel",
        lambda: SimpleNamespace(func=_kernel, expects_transpose=False),
    )

    k_vecs, vertex_areas, weights, tri_rows = compute_curvature_data(mesh, pos, idx)

    assert tri_rows.shape == (1, 3)
    assert np.allclose(k_vecs, 7.0)
    assert np.allclose(vertex_areas, [1.0, 2.0, 3.0])
    assert np.allclose(weights, 0.5)
    assert np.allclose(mesh._curvature_cache.get("va0_raw"), [0.11])
    assert np.allclose(mesh._curvature_cache.get("va1_raw"), [0.22])
    assert np.allclose(mesh._curvature_cache.get("va2_raw"), [0.33])


def test_curvature_kernel_legacy_signature_still_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _single_triangle_mesh()
    pos = mesh.positions_view()
    idx = mesh.vertex_index_to_row
    mesh._curvature_cache["va0_raw"] = np.array([9.0], dtype=np.float64)
    mesh._curvature_cache["va1_raw"] = np.array([8.0], dtype=np.float64)
    mesh._curvature_cache["va2_raw"] = np.array([7.0], dtype=np.float64)

    def _kernel(pos_in, tri_in, k_vecs, vertex_areas, weights, zero_based):
        assert zero_based == 1
        k_vecs[:] = 2.0
        vertex_areas[:] = 4.0
        weights[:] = 6.0

    monkeypatch.setattr(
        loader,
        "get_tilt_curvature_kernel",
        lambda: SimpleNamespace(func=_kernel, expects_transpose=False),
    )

    k_vecs, vertex_areas, weights, _tri_rows = compute_curvature_data(mesh, pos, idx)

    assert np.allclose(k_vecs, 2.0)
    assert np.allclose(vertex_areas, 4.0)
    assert np.allclose(weights, 6.0)
    assert "va0_raw" not in mesh._curvature_cache
    assert "va1_raw" not in mesh._curvature_cache
    assert "va2_raw" not in mesh._curvature_cache


def test_curvature_numpy_fallback_populates_raw_area_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _single_triangle_mesh()
    pos = mesh.positions_view()
    idx = mesh.vertex_index_to_row

    monkeypatch.setattr(loader, "get_tilt_curvature_kernel", lambda: None)

    _k_vecs, vertex_areas, _weights, tri_rows = compute_curvature_data(mesh, pos, idx)

    va0 = np.asarray(mesh._curvature_cache["va0_raw"])
    va1 = np.asarray(mesh._curvature_cache["va1_raw"])
    va2 = np.asarray(mesh._curvature_cache["va2_raw"])
    assert va0.shape == (tri_rows.shape[0],)
    assert va1.shape == (tri_rows.shape[0],)
    assert va2.shape == (tri_rows.shape[0],)

    areas_from_raw = np.zeros(len(mesh.vertex_ids), dtype=float)
    np.add.at(areas_from_raw, tri_rows[:, 0], va0)
    np.add.at(areas_from_raw, tri_rows[:, 1], va1)
    np.add.at(areas_from_raw, tri_rows[:, 2], va2)
    assert np.allclose(areas_from_raw, vertex_areas, atol=1e-12, rtol=1e-12)
