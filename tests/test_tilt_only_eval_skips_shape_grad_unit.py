import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex
from modules.energy import bending_tilt_leaflet as bt_leaflet
from modules.energy import tilt_in as tilt_in_mod


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


class _GP(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _Resolver:
    def __init__(self, d):
        self._d = dict(d)

    def get(self, _entity, key, default=None):
        return self._d.get(key, default)


def test_bending_tilt_leaflet_tilt_grad_parity_when_shape_grad_skipped() -> None:
    mesh = _build_two_triangle_mesh()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    n = len(mesh.vertex_ids)

    rng = np.random.default_rng(0)
    tilts = rng.normal(size=(n, 3))

    gp = _GP({"bending_modulus_in": 2.0, "spontaneous_curvature_in": 0.0})
    resolver = _Resolver({"bending_modulus_in": 2.0})

    grad_full = np.zeros_like(positions)
    tgrad_full = np.zeros_like(tilts)
    E_full = bt_leaflet.compute_energy_and_gradient_array_leaflet(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_full,
        tilts=tilts,
        tilt_grad_arr=tgrad_full,
        kappa_key="bending_modulus_in",
        cache_tag="in",
        div_sign=1.0,
    )

    tgrad_skip = np.zeros_like(tilts)
    E_skip = bt_leaflet.compute_energy_and_gradient_array_leaflet(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=None,
        tilts=tilts,
        tilt_grad_arr=tgrad_skip,
        kappa_key="bending_modulus_in",
        cache_tag="in",
        div_sign=1.0,
    )

    assert float(E_skip) == pytest.approx(float(E_full), rel=1e-12, abs=1e-12)
    assert np.allclose(tgrad_skip, tgrad_full, atol=1e-10, rtol=1e-10)


def test_tilt_in_tilt_grad_parity_when_shape_grad_skipped() -> None:
    mesh = _build_two_triangle_mesh()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    n = len(mesh.vertex_ids)

    rng = np.random.default_rng(1)
    tilts_in = rng.normal(size=(n, 3))

    gp = _GP({})
    resolver = _Resolver({"tilt_modulus_in": 3.0})

    grad_full = np.zeros_like(positions)
    tgrad_full = np.zeros_like(tilts_in)
    E_full = tilt_in_mod.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_full,
        tilts_in=tilts_in,
        tilt_in_grad_arr=tgrad_full,
    )

    tgrad_skip = np.zeros_like(tilts_in)
    E_skip = tilt_in_mod.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=None,
        tilts_in=tilts_in,
        tilt_in_grad_arr=tgrad_skip,
    )

    assert float(E_skip) == pytest.approx(float(E_full), rel=1e-12, abs=1e-12)
    assert np.allclose(tgrad_skip, tgrad_full, atol=1e-10, rtol=1e-10)
