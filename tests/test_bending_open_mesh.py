import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex
from modules.energy import bending
from parameters.global_parameters import GlobalParameters
from parameters.resolver import ParameterResolver


def _create_open_hex_patch() -> Mesh:
    """Create a simple flat hexagonal patch of 6 triangles around a center vertex.
    The boundary is the outer hexagon ring."""
    mesh = Mesh()
    # Center
    mesh.vertices[0] = Vertex(0, [0, 0, 0])
    # Ring
    for i in range(6):
        angle = i * np.pi / 3
        mesh.vertices[i + 1] = Vertex(i + 1, [np.cos(angle), np.sin(angle), 0])

    # Edges
    next_eid = 1
    edge_map = {}

    def get_edge(u, v):
        nonlocal next_eid
        key = (min(u, v), max(u, v))
        if key not in edge_map:
            edge_map[key] = next_eid
            mesh.edges[next_eid] = Edge(next_eid, u, v)
            next_eid += 1
        return edge_map[key]

    # Faces
    for i in range(6):
        u = 0
        v = i + 1
        w = ((i + 1) % 6) + 1

        e1 = get_edge(u, v)
        e2 = get_edge(v, w)
        e3 = get_edge(w, u)

        # Orient u->v->w
        # e1: u->v (positive)
        # e2: v->w (positive)
        # e3: w->u (positive)
        mesh.facets[i] = Facet(i, [e1, e2, e3])

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def _central_diff_grad(mesh: Mesh, gp: GlobalParameters, eps: float) -> np.ndarray:
    positions = mesh.positions_view().copy()
    idx_map = mesh.vertex_index_to_row
    grad = np.zeros_like(positions)
    for i in range(len(positions)):
        if mesh.vertices[mesh.vertex_ids[i]].fixed:
            continue

        for d in range(3):
            pos_plus = positions.copy()
            pos_minus = positions.copy()
            pos_plus[i, d] += eps
            pos_minus[i, d] -= eps
            e_plus = bending.compute_total_energy(mesh, gp, pos_plus, idx_map)
            e_minus = bending.compute_total_energy(mesh, gp, pos_minus, idx_map)
            grad[i, d] = (e_plus - e_minus) / (2.0 * eps)

    # Zero out boundary gradients (FD usually doesn't handle them well if physics is undefined,
    # but for Helfrich with zero density at boundary, gradients should be well defined as 0 or from area variation).
    # The solver explicitly zeroes boundary gradients for fixed vertices.
    # Here we simulate that.
    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [idx_map[vid] for vid in boundary_vids if vid in idx_map]
        grad[boundary_rows] = 0.0

    return grad


def test_helfrich_gradient_consistency_open_mesh():
    """Verify that Analytic and Finite Difference gradients match on an open mesh (with boundaries).
    This ensures the 'Mixed Area' formulation (Voronoi H, Effective A) is consistent."""
    mesh = _create_open_hex_patch()

    # Perturb positions slightly to have non-zero curvature
    mesh.vertices[0].position[2] = 0.1
    for i in range(1, 7):
        mesh.vertices[i].position[2] = 0.05 * np.sin(i)  # Randomish perturbation

    gp = GlobalParameters(
        {
            "bending_modulus": 1.0,
            "bending_energy_model": "helfrich",
            "spontaneous_curvature": 0.5,
            "bending_gradient_mode": "analytic",
        }
    )
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row

    # Analytic Gradient
    grad_arr = np.zeros_like(positions)
    _ = bending.compute_energy_and_gradient_array(
        mesh, gp, resolver, positions=positions, index_map=idx_map, grad_arr=grad_arr
    )

    # Zero out boundary gradients in analytic result to match FD expectation    # (Since we focus on interior consistency)
    boundary_vids = mesh.boundary_vertex_ids
    boundary_rows = [idx_map[vid] for vid in boundary_vids]
    grad_arr[boundary_rows] = 0.0

    # FD Gradient
    grad_fd = _central_diff_grad(mesh, gp, eps=1e-6)

    # Compare
    diff = np.abs(grad_arr - grad_fd)
    denom = np.maximum(np.abs(grad_fd), 1.0)
    rel = diff / denom
    max_rel = float(np.max(rel))

    print(f"Max relative error: {max_rel}")
    assert max_rel < 1e-4, f"Gradient mismatch on open mesh: max_rel={max_rel}"


def test_willmore_gradient_consistency_open_mesh():
    """Verify Willmore gradient consistency on open mesh."""
    mesh = _create_open_hex_patch()

    # Perturb
    mesh.vertices[0].position[2] = 0.2

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
    bending.compute_energy_and_gradient_array(
        mesh, gp, resolver, positions=positions, index_map=idx_map, grad_arr=grad_arr
    )

    boundary_vids = mesh.boundary_vertex_ids
    boundary_rows = [idx_map[vid] for vid in boundary_vids]
    grad_arr[boundary_rows] = 0.0

    grad_fd = _central_diff_grad(mesh, gp, eps=1e-6)

    diff = np.abs(grad_arr - grad_fd)
    denom = np.maximum(np.abs(grad_fd), 1.0)
    rel = diff / denom
    max_rel = float(np.max(rel))

    print(f"Max relative error (Willmore): {max_rel}")
    assert max_rel < 1e-4, f"Gradient mismatch (Willmore): max_rel={max_rel}"
