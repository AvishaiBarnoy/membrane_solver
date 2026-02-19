import importlib
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Edge, Facet, Mesh, Vertex

LEAFLET_CASES = {
    "in": {
        "module": "modules.energy.tilt_in",
        "param": "tilt_modulus_in",
        "field": "tilt_in",
        "touch": "touch_tilts_in",
    },
    "out": {
        "module": "modules.energy.tilt_out",
        "param": "tilt_modulus_out",
        "field": "tilt_out",
        "touch": "touch_tilts_out",
    },
}


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


def _set_mesh_positions(mesh: Mesh, positions: np.ndarray) -> None:
    mesh.build_position_cache()
    if positions.shape != (len(mesh.vertex_ids), 3):
        raise ValueError("positions must have shape (N_vertices, 3)")

    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = positions[row]
    mesh.increment_version()


@pytest.mark.parametrize("leaflet", ["in", "out"])
def test_tilt_leaflet_energy_single_triangle_matches_closed_form(leaflet: str) -> None:
    case = LEAFLET_CASES[leaflet]
    module = importlib.import_module(case["module"])
    mesh = _build_single_triangle_mesh()

    tilts = [
        np.array([1.0, -2.0, 0.0], dtype=float),
        np.array([0.5, 0.25, 0.0], dtype=float),
        np.array([-1.5, 0.0, 0.0], dtype=float),
    ]
    for vid, vec in enumerate(tilts):
        setattr(mesh.vertices[vid], case["field"], vec)
    getattr(mesh, case["touch"])()

    k_tilt = 2.0
    gp = GlobalParameters({case["param"]: k_tilt})
    resolver = ParameterResolver(gp)

    energy, shape_grad, tilt_grad = module.compute_energy_and_gradient(
        mesh, gp, resolver
    )

    area = 0.5
    tilt_sq_sum = sum(float(np.dot(vec, vec)) for vec in tilts)
    expected_energy = (k_tilt * area / 6.0) * tilt_sq_sum
    assert float(energy) == pytest.approx(expected_energy, rel=1e-12, abs=1e-12)

    expected_vertex_area = area / 3.0
    for vid, vec in enumerate(tilts):
        assert tilt_grad[vid] == pytest.approx(
            k_tilt * vec * expected_vertex_area,
            rel=1e-12,
            abs=1e-12,
        )

    assert any(np.linalg.norm(g) > 0.0 for g in shape_grad.values())


@pytest.mark.parametrize("leaflet", ["in", "out"])
def test_tilt_leaflet_shape_gradient_matches_directional_derivative(
    leaflet: str,
) -> None:
    case = LEAFLET_CASES[leaflet]
    module = importlib.import_module(case["module"])
    mesh = _build_single_triangle_mesh()

    for v in mesh.vertices.values():
        setattr(v, case["field"], np.array([0.2, -0.1, 0.0], dtype=float))
    getattr(mesh, case["touch"])()

    gp = GlobalParameters({case["param"]: 1.7})
    resolver = ParameterResolver(gp)

    x0 = mesh.positions_view().copy()
    rng = np.random.default_rng(0)
    direction = rng.normal(size=x0.shape)
    direction /= float(np.linalg.norm(direction))

    _set_mesh_positions(mesh, x0)
    energy0, grad_dict, _tilt_grad = module.compute_energy_and_gradient(
        mesh, gp, resolver
    )

    idx_map = mesh.vertex_index_to_row
    analytic = 0.0
    for vid, gvec in grad_dict.items():
        row = idx_map.get(vid)
        if row is None:
            continue
        analytic += float(np.dot(gvec, direction[row]))

    eps = 1e-6
    _set_mesh_positions(mesh, x0 + eps * direction)
    e_plus, *_ = module.compute_energy_and_gradient(mesh, gp, resolver)
    _set_mesh_positions(mesh, x0 - eps * direction)
    e_minus, *_ = module.compute_energy_and_gradient(mesh, gp, resolver)
    numeric = (float(e_plus) - float(e_minus)) / (2.0 * eps)

    scale = max(1.0, abs(analytic), abs(numeric))
    assert abs(analytic - numeric) / scale < 3e-5


@pytest.mark.parametrize("leaflet", ["in", "out"])
def test_tilt_leaflet_array_matches_dict_energy_and_gradient(leaflet: str) -> None:
    case = LEAFLET_CASES[leaflet]
    module = importlib.import_module(case["module"])
    mesh = _build_single_triangle_mesh()

    tilts = [
        np.array([0.3, 0.1, 0.0], dtype=float),
        np.array([-0.2, 0.05, 0.0], dtype=float),
        np.array([0.0, -0.4, 0.0], dtype=float),
    ]
    for vid, vec in enumerate(tilts):
        setattr(mesh.vertices[vid], case["field"], vec)
    getattr(mesh, case["touch"])()

    gp = GlobalParameters({case["param"]: 0.9})
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view().copy()
    idx_map = mesh.vertex_index_to_row

    e_dict, grad_dict, tilt_grad = module.compute_energy_and_gradient(
        mesh, gp, resolver
    )

    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions)
    kwargs = {
        "tilts_in": mesh.tilts_in_view(),
        "tilt_in_grad_arr": tilt_grad_arr,
    }
    if leaflet == "out":
        kwargs = {
            "tilts_out": mesh.tilts_out_view(),
            "tilt_out_grad_arr": tilt_grad_arr,
        }

    e_arr = module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
        **kwargs,
    )

    assert float(e_arr) == pytest.approx(float(e_dict), rel=1e-12, abs=1e-12)
    for vid, gvec in grad_dict.items():
        row = idx_map[vid]
        assert grad_arr[row] == pytest.approx(gvec, rel=1e-12, abs=1e-12)
    for vid, gvec in tilt_grad.items():
        row = idx_map[vid]
        assert tilt_grad_arr[row] == pytest.approx(gvec, rel=1e-12, abs=1e-12)


def test_tilt_in_consistent_mass_single_triangle_closed_form() -> None:
    module = importlib.import_module("modules.energy.tilt_in")
    mesh = _build_single_triangle_mesh()

    t0 = np.array([1.0, -2.0, 0.0], dtype=float)
    t1 = np.array([0.5, 0.25, 0.0], dtype=float)
    t2 = np.array([-1.5, 0.0, 0.0], dtype=float)
    mesh.vertices[0].tilt_in = t0
    mesh.vertices[1].tilt_in = t1
    mesh.vertices[2].tilt_in = t2
    mesh.touch_tilts_in()

    k_tilt = 2.0
    gp = GlobalParameters(
        {"tilt_modulus_in": k_tilt, "tilt_mass_mode_in": "consistent"}
    )
    resolver = ParameterResolver(gp)

    energy, _shape_grad, tilt_grad = module.compute_energy_and_gradient(
        mesh, gp, resolver
    )

    area = 0.5
    s = (
        float(np.dot(t0, t0))
        + float(np.dot(t1, t1))
        + float(np.dot(t2, t2))
        + float(np.dot(t0, t1))
        + float(np.dot(t1, t2))
        + float(np.dot(t2, t0))
    )
    expected_energy = float((k_tilt * area / 12.0) * s)
    assert float(energy) == pytest.approx(expected_energy, rel=1e-12, abs=1e-12)

    scale = float(k_tilt * area / 12.0)
    assert tilt_grad[0] == pytest.approx(scale * ((2.0 * t0) + t1 + t2), abs=1e-12)
    assert tilt_grad[1] == pytest.approx(scale * ((2.0 * t1) + t2 + t0), abs=1e-12)
    assert tilt_grad[2] == pytest.approx(scale * ((2.0 * t2) + t0 + t1), abs=1e-12)


def test_tilt_in_mass_mode_guard_rejects_invalid_mode() -> None:
    module = importlib.import_module("modules.energy.tilt_in")
    mesh = _build_single_triangle_mesh()
    mesh.touch_tilts_in()
    gp = GlobalParameters({"tilt_modulus_in": 1.0, "tilt_mass_mode_in": "invalid"})
    resolver = ParameterResolver(gp)

    with pytest.raises(ValueError, match="tilt_mass_mode_in must be"):
        module.compute_energy_and_gradient(mesh, gp, resolver)
