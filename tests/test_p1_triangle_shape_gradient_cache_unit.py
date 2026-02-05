import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex
from geometry.tilt_operators import (
    p1_triangle_divergence,
    p1_triangle_divergence_from_shape_gradients,
)
from modules.energy import bending_tilt_leaflet as bt_leaflet


def _build_two_triangle_mesh() -> Mesh:
    # Square split into two triangles.
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([1.0, 1.0, 0.0])),
        3: Vertex(3, np.array([0.0, 1.0, 0.0])),
    }
    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 3),
        4: Edge(4, 3, 0),
        5: Edge(5, 0, 2),  # diagonal
    }
    mesh.facets = {
        0: Facet(0, edge_indices=[1, 2, -5]),
        1: Facet(1, edge_indices=[5, 3, 4]),
    }
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def test_p1_triangle_divergence_matches_cached_shape_gradients() -> None:
    mesh = _build_two_triangle_mesh()
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    assert tri_rows is not None and tri_rows.size > 0

    rng = np.random.default_rng(0)
    tilts = rng.normal(size=(len(mesh.vertex_ids), 3))

    div_ref, _area, _g0, _g1, _g2 = p1_triangle_divergence(
        positions=positions, tilts=tilts, tri_rows=tri_rows
    )
    _area_c, g0, g1, g2, tri_rows_c = mesh.p1_triangle_shape_gradient_cache(positions)
    assert tri_rows_c.shape == tri_rows.shape

    div_cached = p1_triangle_divergence_from_shape_gradients(
        tilts=tilts, tri_rows=tri_rows, g0=g0, g1=g1, g2=g2
    )
    assert np.allclose(div_cached, div_ref, atol=1e-12, rtol=1e-12)


def test_bending_tilt_leaflet_uses_cached_shape_gradients_without_changing_results(
    monkeypatch,
) -> None:
    mesh = _build_two_triangle_mesh()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    n = len(mesh.vertex_ids)

    rng = np.random.default_rng(1)
    tilts = rng.normal(size=(n, 3))

    class _GP(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    gp = _GP(
        {
            "bending_modulus_in": 2.0,
            "spontaneous_curvature_in": 0.0,
        }
    )

    grad_a = np.zeros_like(positions)
    tilt_grad_a = np.zeros_like(tilts)
    E_a = bt_leaflet.compute_energy_and_gradient_array_leaflet(
        mesh,
        gp,
        None,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_a,
        tilts=tilts,
        tilt_grad_arr=tilt_grad_a,
        kappa_key="bending_modulus_in",
        cache_tag="in",
        div_sign=1.0,
    )

    # Force fallback to the original divergence path by making the cache helper
    # return an empty triangle set.
    def _empty_cache(_pos=None):
        zeros1 = np.zeros(0, dtype=float)
        zeros3 = np.zeros((0, 3), dtype=float)
        tri_empty = np.zeros((0, 3), dtype=np.int32)
        return zeros1, zeros3, zeros3, zeros3, tri_empty

    monkeypatch.setattr(mesh, "p1_triangle_shape_gradient_cache", _empty_cache)

    grad_b = np.zeros_like(positions)
    tilt_grad_b = np.zeros_like(tilts)
    E_b = bt_leaflet.compute_energy_and_gradient_array_leaflet(
        mesh,
        gp,
        None,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_b,
        tilts=tilts,
        tilt_grad_arr=tilt_grad_b,
        kappa_key="bending_modulus_in",
        cache_tag="in",
        div_sign=1.0,
    )

    assert float(E_a) == pytest.approx(float(E_b), rel=1e-12, abs=1e-12)
    assert np.allclose(grad_a, grad_b, atol=1e-10, rtol=1e-10)
    assert np.allclose(tilt_grad_a, tilt_grad_b, atol=1e-10, rtol=1e-10)
