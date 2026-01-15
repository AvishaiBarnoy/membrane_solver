import importlib
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Edge, Facet, Mesh, Vertex


def _build_single_triangle_mesh() -> Mesh:
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([0.0, 1.0, 0.0])),
    }
    mesh.edges = {1: Edge(1, 0, 1), 2: Edge(2, 1, 2), 3: Edge(3, 2, 0)}
    mesh.facets = {0: Facet(0, edge_indices=[1, 2, 3])}
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def _set_leaflet_tilts(mesh: Mesh, tilts_in: np.ndarray, tilts_out: np.ndarray) -> None:
    for vid in mesh.vertices:
        mesh.vertices[vid].tilt_in = tilts_in[vid].copy()
        mesh.vertices[vid].tilt_out = tilts_out[vid].copy()
    mesh.touch_tilts_in()
    mesh.touch_tilts_out()


def test_tilt_coupling_difference_zero_when_equal() -> None:
    module = importlib.import_module("modules.energy.tilt_coupling")
    mesh = _build_single_triangle_mesh()

    tilts = np.array([[0.3, 0.1, 0.0], [0.3, 0.1, 0.0], [0.3, 0.1, 0.0]], dtype=float)
    _set_leaflet_tilts(mesh, tilts, tilts)

    gp = GlobalParameters(
        {"tilt_coupling_modulus": 2.0, "tilt_coupling_mode": "difference"}
    )
    resolver = ParameterResolver(gp)

    energy, shape_grad, tilt_grad = module.compute_energy_and_gradient(
        mesh, gp, resolver
    )

    assert float(energy) == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert all(np.allclose(g, 0.0, atol=1e-12) for g in shape_grad.values())
    assert all(np.allclose(g, 0.0, atol=1e-12) for g in tilt_grad.values())


@pytest.mark.parametrize("mode,expected_sign", [("difference", -1.0), ("sum", 1.0)])
def test_tilt_coupling_energy_single_triangle_matches_closed_form(
    mode: str, expected_sign: float
) -> None:
    module = importlib.import_module("modules.energy.tilt_coupling")
    mesh = _build_single_triangle_mesh()

    tilts_in = np.array(
        [[0.5, -0.2, 0.0], [0.1, 0.3, 0.0], [-0.4, 0.2, 0.0]], dtype=float
    )
    tilts_out = np.array(
        [[0.25, 0.1, 0.0], [-0.2, 0.15, 0.0], [0.05, -0.1, 0.0]], dtype=float
    )
    _set_leaflet_tilts(mesh, tilts_in, tilts_out)

    k_c = 1.7
    gp = GlobalParameters({"tilt_coupling_modulus": k_c, "tilt_coupling_mode": mode})
    resolver = ParameterResolver(gp)

    energy, shape_grad, tilt_grad = module.compute_energy_and_gradient(
        mesh, gp, resolver
    )

    area = 0.5
    diff = tilts_out + expected_sign * tilts_in
    diff_sq_sum = sum(float(np.dot(vec, vec)) for vec in diff)
    expected_energy = 0.5 * k_c * (area / 3.0) * diff_sq_sum

    assert float(energy) == pytest.approx(expected_energy, rel=1e-12, abs=1e-12)
    assert shape_grad
    assert tilt_grad


def test_tilt_coupling_gradient_matches_directional_derivative() -> None:
    module = importlib.import_module("modules.energy.tilt_coupling")
    mesh = _build_single_triangle_mesh()

    rng = np.random.default_rng(123)
    tilts_in = rng.normal(size=(3, 3))
    tilts_out = rng.normal(size=(3, 3))
    tilts_in[:, 2] = 0.0
    tilts_out[:, 2] = 0.0
    _set_leaflet_tilts(mesh, tilts_in, tilts_out)

    gp = GlobalParameters(
        {"tilt_coupling_modulus": 0.9, "tilt_coupling_mode": "difference"}
    )
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view().copy()
    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_in_grad = np.zeros_like(positions)
    tilt_out_grad = np.zeros_like(positions)

    energy0 = module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
        tilts_in=tilts_in,
        tilts_out=tilts_out,
        tilt_in_grad_arr=tilt_in_grad,
        tilt_out_grad_arr=tilt_out_grad,
    )

    direction_in = rng.normal(size=tilts_in.shape)
    direction_out = rng.normal(size=tilts_out.shape)
    scale = float(
        np.sqrt(
            np.sum(direction_in**2) + np.sum(direction_out**2),
        )
    )
    direction_in /= scale
    direction_out /= scale

    analytic = float(
        np.sum(tilt_in_grad * direction_in) + np.sum(tilt_out_grad * direction_out)
    )

    eps = 1e-6
    e_plus = module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=np.zeros_like(positions),
        tilts_in=tilts_in + eps * direction_in,
        tilts_out=tilts_out + eps * direction_out,
    )
    e_minus = module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=np.zeros_like(positions),
        tilts_in=tilts_in - eps * direction_in,
        tilts_out=tilts_out - eps * direction_out,
    )
    numeric = (float(e_plus) - float(e_minus)) / (2.0 * eps)

    scale = max(1.0, abs(analytic), abs(numeric))
    assert abs(analytic - numeric) / scale < 3e-5
    assert float(energy0) >= 0.0
