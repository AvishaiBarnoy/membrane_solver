import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex
from modules.energy import bending_tilt_leaflet as bt_leaflet


def _build_two_triangle_mesh() -> Mesh:
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
        5: Edge(5, 0, 2),
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


def _eval(
    mesh: Mesh, gp: _GP, tilts: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad = np.zeros_like(positions)
    tilt_grad = np.zeros_like(tilts)
    E = bt_leaflet.compute_energy_and_gradient_array_leaflet(
        mesh,
        gp,
        None,
        positions=positions,
        index_map=index_map,
        grad_arr=grad,
        tilts=tilts,
        tilt_grad_arr=tilt_grad,
        kappa_key="bending_modulus_in",
        cache_tag="in",
        div_sign=1.0,
    )
    return float(E), grad, tilt_grad


def test_cotan_gradient_cache_does_not_change_bending_tilt_leaflet_outputs(
    monkeypatch,
) -> None:
    mesh = _build_two_triangle_mesh()
    n = len(mesh.vertex_ids)
    rng = np.random.default_rng(0)
    tilts = rng.normal(size=(n, 3))

    gp = _GP(
        {
            "bending_modulus_in": 2.0,
            "spontaneous_curvature_in": 0.0,
            # Ensure we hit the analytic path where cotan gradients matter.
            "bending_gradient_mode": "analytic",
        }
    )

    # Cached path (positions_view activates geometry caching).
    E_a, grad_a, tgrad_a = _eval(mesh, gp, tilts)

    # Force the cache helper to be inactive so modules fall back to per-call
    # cotan gradient computation.
    monkeypatch.setattr(mesh, "_geometry_cache_active", lambda _pos: False)

    E_b, grad_b, tgrad_b = _eval(mesh, gp, tilts)

    assert E_a == pytest.approx(E_b, rel=1e-12, abs=1e-12)
    assert np.allclose(grad_a, grad_b, atol=1e-10, rtol=1e-10)
    assert np.allclose(tgrad_a, tgrad_b, atol=1e-10, rtol=1e-10)
