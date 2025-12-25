import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex
from modules.energy import bending
from parameters.global_parameters import GlobalParameters
from parameters.resolver import ParameterResolver


def _tetra_mesh() -> Mesh:
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
        for tail, head in [(a, b), (b, c), (c, a)]:
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


def _central_diff_grad(mesh: Mesh, gp: GlobalParameters, eps: float) -> np.ndarray:
    positions = mesh.positions_view().copy()
    idx_map = mesh.vertex_index_to_row
    grad = np.zeros_like(positions)
    for i in range(len(positions)):
        for d in range(3):
            pos_plus = positions.copy()
            pos_minus = positions.copy()
            pos_plus[i, d] += eps
            pos_minus[i, d] -= eps
            e_plus = bending.compute_total_energy(mesh, gp, pos_plus, idx_map)
            e_minus = bending.compute_total_energy(mesh, gp, pos_minus, idx_map)
            grad[i, d] = (e_plus - e_minus) / (2.0 * eps)
    return grad


def test_bending_finite_difference_gradient_matches_energy_derivative():
    mesh = _tetra_mesh()
    gp = GlobalParameters(
        {
            "bending_modulus": 1.0,
            "bending_energy_model": "willmore",
            "bending_gradient_mode": "finite_difference",
            "bending_fd_eps": 1e-6,
        }
    )
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    _ = bending.compute_energy_and_gradient_array(
        mesh, gp, resolver, positions=positions, index_map=idx_map, grad_arr=grad_arr
    )

    grad_num = _central_diff_grad(mesh, gp, eps=1e-6)
    diff = np.abs(grad_arr - grad_num)
    denom = np.maximum(np.abs(grad_num), 1.0)
    rel = diff / denom
    assert float(np.max(rel)) < 5e-4


def test_bending_analytic_gradient_matches_energy_derivative():
    mesh = _tetra_mesh()
    gp = GlobalParameters(
        {
            "bending_modulus": 1.0,
            "bending_energy_model": "willmore",
            "bending_gradient_mode": "analytic",
        }
    )
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    _ = bending.compute_energy_and_gradient_array(
        mesh, gp, resolver, positions=positions, index_map=idx_map, grad_arr=grad_arr
    )

    grad_num = _central_diff_grad(mesh, gp, eps=1e-6)
    diff = np.abs(grad_arr - grad_num)
    denom = np.maximum(np.abs(grad_num), 1.0)
    rel = diff / denom
    max_rel = float(np.max(rel))
    assert max_rel < 5e-4, f"max_rel={max_rel}"


def test_helfrich_energy_reduces_when_C0_matches_sphere():
    from runtime.refinement import refine_triangle_mesh

    mesh = Mesh()
    points = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    for i, p in enumerate(points):
        mesh.vertices[i] = Vertex(i, p)

    tri_indices = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]

    edge_map: dict[tuple[int, int], int] = {}
    next_eid = 1
    for fid, (a, b, c) in enumerate(tri_indices):
        e_ids = []
        for tail, head in [(a, b), (b, c), (c, a)]:
            key = (min(tail, head), max(tail, head))
            if key not in edge_map:
                edge_map[key] = next_eid
                mesh.edges[next_eid] = Edge(next_eid, tail, head)
                next_eid += 1
            eid = edge_map[key]
            edge = mesh.edges[eid]
            e_ids.append(eid if edge.tail_index == tail else -eid)
        mesh.facets[fid] = Facet(fid, e_ids)

    for _ in range(2):
        for v in mesh.vertices.values():
            v.position = v.position / np.linalg.norm(v.position)
        mesh = refine_triangle_mesh(mesh)
    for v in mesh.vertices.values():
        v.position = v.position / np.linalg.norm(v.position)

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row

    gp0 = GlobalParameters({"bending_modulus": 1.0, "bending_energy_model": "helfrich"})
    E0 = bending.compute_total_energy(mesh, gp0, positions, idx_map)

    gp_match = GlobalParameters(
        {
            "bending_modulus": 1.0,
            "bending_energy_model": "helfrich",
            "spontaneous_curvature": 2.0,
        }
    )
    E_match = bending.compute_total_energy(mesh, gp_match, positions, idx_map)

    assert E_match < E0 * 0.3
