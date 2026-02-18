import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Edge, Facet, Mesh, Vertex
from modules.energy import tilt_splay_twist_in


def _build_two_triangle_square_mesh() -> Mesh:
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
        3: Edge(3, 2, 0),
        4: Edge(4, 2, 3),
        5: Edge(5, 3, 0),
    }
    mesh.facets = {
        0: Facet(0, edge_indices=[1, 2, 3]),
        1: Facet(1, edge_indices=[-3, 4, 5]),
    }
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def _set_inner_tilt_from_xy(mesh: Mesh, fn) -> None:
    for v in mesh.vertices.values():
        x, y, _z = v.position
        v.tilt_in = np.asarray(fn(float(x), float(y)), dtype=float)
    mesh.touch_tilts_in()


def test_tilt_splay_twist_in_default_zero_twist_does_not_penalize_pure_curl() -> None:
    mesh = _build_two_triangle_square_mesh()
    _set_inner_tilt_from_xy(mesh, lambda x, y: np.array([-y, x, 0.0]))

    gp = GlobalParameters({"bending_modulus_in": 1.0})
    resolver = ParameterResolver(gp)

    energy, _shape_grad, tilt_grad = tilt_splay_twist_in.compute_energy_and_gradient(
        mesh, gp, resolver
    )

    assert float(energy) == pytest.approx(0.0, abs=1e-12)
    assert all(np.allclose(g, 0.0, atol=1e-12) for g in tilt_grad.values())


def test_tilt_splay_twist_in_gradient_matches_directional_derivative() -> None:
    mesh = _build_two_triangle_square_mesh()
    gp = GlobalParameters({"tilt_splay_modulus_in": 0.7, "tilt_twist_modulus_in": 0.4})
    resolver = ParameterResolver(gp)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    rng = np.random.default_rng(123)
    tilts = 1e-2 * rng.standard_normal(size=(len(mesh.vertex_ids), 3))
    tilts[:, 2] = 0.0
    direction = rng.standard_normal(size=tilts.shape)
    direction[:, 2] = 0.0

    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions)
    energy = tilt_splay_twist_in.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts_in=tilts,
        tilt_in_grad_arr=tilt_grad_arr,
    )
    assert float(energy) >= 0.0

    eps = 1e-7
    e_plus = tilt_splay_twist_in.compute_energy_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts + eps * direction,
    )
    e_minus = tilt_splay_twist_in.compute_energy_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts - eps * direction,
    )
    fd = float(e_plus - e_minus) / (2.0 * eps)
    analytic = float(np.sum(tilt_grad_arr * direction))

    assert analytic == pytest.approx(fd, rel=1e-5, abs=1e-8)
