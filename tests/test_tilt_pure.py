import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Edge, Facet, Mesh, Vertex
from modules.energy import tilt


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


def test_tilt_energy_single_triangle_matches_closed_form():
    mesh = _build_single_triangle_mesh()
    mesh.vertices[0].tilt = np.array([1.0, -2.0], dtype=float)
    mesh.vertices[1].tilt = np.array([0.5, 0.25], dtype=float)
    mesh.vertices[2].tilt = np.array([-1.5, 0.0], dtype=float)

    k_tilt = 2.0
    gp = GlobalParameters({"tilt_rigidity": k_tilt})
    resolver = ParameterResolver(gp)

    energy, shape_grad, tilt_grad = tilt.compute_energy_and_gradient(mesh, gp, resolver)

    area = 0.5
    tilt_sq_sum = (
        float(np.dot(mesh.vertices[0].tilt, mesh.vertices[0].tilt))
        + float(np.dot(mesh.vertices[1].tilt, mesh.vertices[1].tilt))
        + float(np.dot(mesh.vertices[2].tilt, mesh.vertices[2].tilt))
    )
    expected_energy = (k_tilt * area / 6.0) * tilt_sq_sum
    assert float(energy) == pytest.approx(expected_energy, rel=1e-12, abs=1e-12)

    # Per-vertex barycentric area is area/3 for a single triangle.
    expected_vertex_area = area / 3.0
    for vid in (0, 1, 2):
        assert tilt_grad[vid] == pytest.approx(
            k_tilt * mesh.vertices[vid].tilt * expected_vertex_area,
            rel=1e-12,
            abs=1e-12,
        )

    # Shape gradient should be populated (tilt couples to geometry via the area weight).
    assert any(np.linalg.norm(g) > 0.0 for g in shape_grad.values())


def test_tilt_shape_gradient_matches_directional_derivative():
    mesh = _build_single_triangle_mesh()
    for v in mesh.vertices.values():
        v.tilt = np.array([0.2, -0.1], dtype=float)

    gp = GlobalParameters({"tilt_rigidity": 1.7})
    resolver = ParameterResolver(gp)

    x0 = mesh.positions_view().copy()
    rng = np.random.default_rng(0)
    direction = rng.normal(size=x0.shape)
    direction /= float(np.linalg.norm(direction))

    _set_mesh_positions(mesh, x0)
    energy0, grad_dict, _tilt_grad = tilt.compute_energy_and_gradient(
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
    e_plus, *_ = tilt.compute_energy_and_gradient(mesh, gp, resolver)
    _set_mesh_positions(mesh, x0 - eps * direction)
    e_minus, *_ = tilt.compute_energy_and_gradient(mesh, gp, resolver)
    numeric = (float(e_plus) - float(e_minus)) / (2.0 * eps)

    scale = max(1.0, abs(analytic), abs(numeric))
    assert abs(analytic - numeric) / scale < 3e-5


def test_tilt_array_matches_dict_energy_and_gradient():
    mesh = _build_single_triangle_mesh()
    mesh.vertices[0].tilt = np.array([0.3, 0.1], dtype=float)
    mesh.vertices[1].tilt = np.array([-0.2, 0.05], dtype=float)
    mesh.vertices[2].tilt = np.array([0.0, -0.4], dtype=float)

    gp = GlobalParameters({"tilt_rigidity": 0.9})
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view().copy()
    idx_map = mesh.vertex_index_to_row

    e_dict, grad_dict, _tilt_grad = tilt.compute_energy_and_gradient(mesh, gp, resolver)

    grad_arr = np.zeros_like(positions)
    e_arr = tilt.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
    )

    assert float(e_arr) == pytest.approx(float(e_dict), rel=1e-12, abs=1e-12)
    for vid, gvec in grad_dict.items():
        row = idx_map[vid]
        assert grad_arr[row] == pytest.approx(gvec, rel=1e-12, abs=1e-12)
