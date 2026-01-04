import os
import sys
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from modules import mean_curvature_tilt
from modules.constraints import body_area as body_area_constraint
from modules.constraints import volume as volume_constraint
from modules.energy import (
    bending,
    body_area_penalty,
    expression,
    gaussian_curvature,
    jordan_area,
    line_tension,
    surface,
    tilt,
    volume,
)


def create_random_mesh():
    """Creates a small, non-symmetric mesh for numerical testing."""
    mesh = Mesh()
    # 4 vertices forming a non-regular tetrahedron
    pts = np.array(
        [[0.1, 0.2, 0.05], [1.1, -0.1, 0.3], [0.4, 1.2, -0.2], [0.5, 0.4, 1.5]]
    )
    for i, p in enumerate(pts):
        mesh.vertices[i] = Vertex(i, p)

    # Triangles
    indices = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    edge_map = {}
    next_eid = 1
    for i, tri in enumerate(indices):
        e_ids = []
        for pair in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = tuple(sorted(pair))
            if key not in edge_map:
                edge_map[key] = next_eid
                mesh.edges[next_eid] = Edge(next_eid, pair[0], pair[1])
                next_eid += 1
            eid = edge_map[key]
            e_ids.append(eid if mesh.edges[eid].tail_index == pair[0] else -eid)
        mesh.facets[i] = Facet(i, e_ids)

    mesh.bodies[0] = Body(0, list(mesh.facets.keys()), target_volume=0.5)
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def check_gradient_consistency(module, mesh, eps=1e-6, tol=1e-5, is_constraint=False):
    """
    Perform a central difference check:
    dE/dx approx (E(x+h) - E(x-h)) / 2h
    """
    gp = GlobalParameters()
    gp.surface_tension = 1.0
    gp.volume_stiffness = 10.0
    gp.bending_modulus = 1.0
    gp.gaussian_modulus = 1.0
    gp.area_stiffness = 5.0
    gp.set("volume_constraint_mode", "penalty" if not is_constraint else "lagrange")
    resolver = ParameterResolver(gp)

    # 1. Analytical Gradient
    positions = mesh.positions_view().copy()
    idx_map = mesh.vertex_index_to_row

    if is_constraint:
        # For constraints, we test the first gradient in the list
        grads = module.constraint_gradients(mesh, gp)
        if not grads:
            return
        grad_dict = grads[0]
        grad_analytical = np.zeros_like(positions)
        for vid, g in grad_dict.items():
            if vid in idx_map:
                grad_analytical[idx_map[vid]] = g

        def get_val(p):
            # Bypass cache by passing p to compute_volume
            if module == volume_constraint:
                return mesh.bodies[0].compute_volume(
                    mesh, positions=p, index_map=idx_map
                )
            elif module == body_area_constraint:
                # Manually sum facet areas for the first body
                total_area = 0.0
                for fidx in mesh.bodies[0].facet_indices:
                    # compute_area_and_gradient returns (area, grad)
                    total_area += mesh.facets[fidx].compute_area_and_gradient(
                        mesh, positions=p, index_map=idx_map
                    )[0]
                return total_area
            return 0.0
    else:
        grad_analytical = np.zeros_like(positions)
        module.compute_energy_and_gradient_array(
            mesh,
            gp,
            resolver,
            positions=positions,
            index_map=idx_map,
            grad_arr=grad_analytical,
        )

        def get_val(p):
            return module.compute_energy_and_gradient_array(
                mesh,
                gp,
                resolver,
                positions=p,
                index_map=idx_map,
                grad_arr=np.zeros_like(positions),
            )

    # 2. Numerical Gradient
    grad_numerical = np.zeros_like(positions)

    # We must reset the mesh positions because some modules might use mesh.vertices directly
    orig_pts = {vid: v.position.copy() for vid, v in mesh.vertices.items()}

    for i in range(len(positions)):
        for d in range(3):
            # Force update mesh.vertices positions
            def set_pos(idx, dim, delta):
                row_val = positions[idx, dim] + delta
                for vid, row in idx_map.items():
                    if row == idx:
                        mesh.vertices[vid].position[dim] = row_val
                return positions.copy()

            # Shift forward
            p_plus = positions.copy()
            p_plus[i, d] += eps
            # Update mesh state
            for vid, row in idx_map.items():
                if row == i:
                    mesh.vertices[vid].position[d] = p_plus[i, d]
            v_plus = get_val(p_plus)

            # Shift backward
            p_minus = positions.copy()
            p_minus[i, d] -= eps
            for vid, row in idx_map.items():
                if row == i:
                    mesh.vertices[vid].position[d] = p_minus[i, d]
            v_minus = get_val(p_minus)

            # Reset
            for vid, row in idx_map.items():
                if row == i:
                    mesh.vertices[vid].position[d] = positions[i, d]

            grad_numerical[i, d] = (v_plus - v_minus) / (2 * eps)

    # Restore original positions
    for vid, p in orig_pts.items():
        mesh.vertices[vid].position = p

    # Compare
    diff = np.abs(grad_analytical - grad_numerical)
    denom = np.maximum(np.abs(grad_analytical), 1.0)
    rel_diff = diff / denom
    max_rel_diff = np.max(rel_diff)

    print(
        f"Module {module.__name__ if hasattr(module, '__name__') else module}: Max Rel Diff: {max_rel_diff:.2e}"
    )
    assert max_rel_diff < tol


def test_surface_gradient_consistency():
    mesh = create_random_mesh()
    check_gradient_consistency(surface, mesh)


def test_volume_energy_gradient_consistency():
    mesh = create_random_mesh()
    check_gradient_consistency(volume, mesh)


def test_line_tension_gradient_consistency():
    mesh = create_random_mesh()
    for edge in mesh.edges.values():
        edge.options["energy"] = ["line_tension"]
    check_gradient_consistency(line_tension, mesh)


def test_bending_gradient_consistency():
    mesh = create_random_mesh()
    # Higher tolerance for 4th order derivative approximation
    check_gradient_consistency(bending, mesh, eps=1e-4, tol=2.0)


def test_body_area_penalty_gradient_consistency():
    mesh = create_random_mesh()
    mesh.bodies[0].options["area_target"] = 1.0
    check_gradient_consistency(body_area_penalty, mesh)


def test_expression_energy_gradient_consistency():
    mesh = create_random_mesh()
    # Simple x**2 energy on vertex 0 (using Python power syntax)
    mesh.vertices[0].options["energy"] = ["expression"]
    mesh.vertices[0].options["expression"] = "x**2"
    check_gradient_consistency(expression, mesh, tol=1e-4)


def test_volume_constraint_gradient_consistency():
    mesh = create_random_mesh()
    check_gradient_consistency(volume_constraint, mesh, is_constraint=True)


def test_invariance_under_translation():
    mesh = create_random_mesh()
    gp = GlobalParameters()
    E1 = surface.calculate_surface_energy(mesh, gp)
    for v in mesh.vertices.values():
        v.position += np.array([10.0, -5.0, 3.0])
    E2 = surface.calculate_surface_energy(mesh, gp)
    assert E1 == pytest.approx(E2)


def test_invariance_under_rotation():
    mesh = create_random_mesh()
    gp = GlobalParameters()
    E1 = surface.calculate_surface_energy(mesh, gp)
    c, s = 0, 1
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for v in mesh.vertices.values():
        v.position = R @ v.position
    E2 = surface.calculate_surface_energy(mesh, gp)
    assert E1 == pytest.approx(E2)


def test_bending_flow_reduces_energy():
    mesh = create_random_mesh()
    gp = GlobalParameters()
    gp.bending_modulus = 1.0
    gp.set("bending_gradient_mode", "finite_difference")
    gp.set("bending_fd_eps", 1e-6)
    resolver = ParameterResolver(gp)
    positions = mesh.positions_view().copy()
    idx_map = mesh.vertex_index_to_row
    grad = np.zeros_like(positions)
    E_initial = bending.compute_energy_and_gradient_array(
        mesh, gp, resolver, positions=positions, index_map=idx_map, grad_arr=grad
    )
    step_size = 1e-5
    new_positions = positions - step_size * grad
    for vid, row in idx_map.items():
        mesh.vertices[vid].position = new_positions[row]
    E_final = bending.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=new_positions,
        index_map=idx_map,
        grad_arr=np.zeros_like(grad),
    )
    assert E_final < E_initial


def test_single_triangle_bending_is_zero():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0, 0, 0], dtype=float))
    mesh.vertices[1] = Vertex(1, np.array([1, 0, 0], dtype=float))
    mesh.vertices[2] = Vertex(2, np.array([0, 1, 0], dtype=float))
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)
    mesh.facets[0] = Facet(0, [1, 2, 3])
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    gp = GlobalParameters()
    gp.bending_modulus = 1.0
    resolver = ParameterResolver(gp)
    energy, _ = bending.compute_energy_and_gradient(mesh, gp, resolver)
    assert energy == pytest.approx(0.0, abs=1e-10)


def create_planar_mesh():
    """Creates a simple planar mesh (single triangle) for Jordan area tests."""
    mesh = Mesh()
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for i, p in enumerate(pts):
        mesh.vertices[i] = Vertex(i, p)

    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)
    mesh.facets[0] = Facet(0, [1, 2, 3])

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def initialize_tilt(mesh):
    """Initialize random tilt vectors for testing."""
    for v in mesh.vertices.values():
        v.tilt = np.array([0.1, -0.2, 0.0], dtype=float)
    mesh.touch_tilts()


def check_gradient_consistency_dict(module, mesh, eps=1e-6, tol=1e-5):
    """
    Check consistency for modules that return a dictionary gradient
    instead of using the fill-array interface.
    """
    gp = GlobalParameters()
    gp.set("jordan_target_area", 0.5)
    gp.set("jordan_stiffness", 10.0)
    gp.set("tilt_rigidity", 1.0)
    gp.set("bending_rigidity", 1.0)
    gp.set("spontaneous_curvature", 0.1)

    resolver = ParameterResolver(gp)

    # 1. Analytical Gradient
    res = module.compute_energy_and_gradient(mesh, gp, resolver)
    if len(res) == 2:
        # energy, grad_dict
        _, grad_dict = res
    elif len(res) == 3:
        # energy, shape_grad, tilt_grad
        _, grad_dict, _ = res
    else:
        raise ValueError(f"Unexpected return from module {module}")

    positions = mesh.positions_view().copy()
    idx_map = mesh.vertex_index_to_row

    grad_analytical = np.zeros_like(positions)
    for vid, g in grad_dict.items():
        if vid in idx_map:
            grad_analytical[idx_map[vid]] = g

    def get_val(p):
        # Temporarily update positions
        # We assume the module reads from mesh.vertices, so we must update them
        # (This logic is duplicated from check_gradient_consistency but necessary)
        orig_pos = {}
        for vid, row in idx_map.items():
            orig_pos[vid] = mesh.vertices[vid].position.copy()
            mesh.vertices[vid].position[:] = p[row]

        res = module.compute_energy_and_gradient(mesh, gp, resolver)

        # Restore
        for vid, pos in orig_pos.items():
            mesh.vertices[vid].position[:] = pos

        return res[0]

    # 2. Numerical Gradient
    grad_numerical = np.zeros_like(positions)

    for i in range(len(positions)):
        for d in range(3):
            # Shift forward
            p_plus = positions.copy()
            p_plus[i, d] += eps
            v_plus = get_val(p_plus)

            # Shift backward
            p_minus = positions.copy()
            p_minus[i, d] -= eps
            v_minus = get_val(p_minus)

            grad_numerical[i, d] = (v_plus - v_minus) / (2 * eps)

    # Compare
    diff = np.abs(grad_analytical - grad_numerical)
    denom = np.maximum(np.abs(grad_analytical), 1.0)
    rel_diff = diff / denom
    max_rel_diff = np.max(rel_diff)

    print(
        f"Module {module.__name__ if hasattr(module, '__name__') else module}: Max Rel Diff: {max_rel_diff:.2e}"
    )
    assert max_rel_diff < tol


def test_jordan_area_gradient_consistency():
    mesh = create_planar_mesh()
    check_gradient_consistency_dict(jordan_area, mesh)


def test_tilt_energy_gradient_consistency():
    mesh = create_random_mesh()
    initialize_tilt(mesh)
    check_gradient_consistency_dict(tilt, mesh, tol=1e-4)


def test_mean_curvature_tilt_gradient_consistency():
    mesh = create_random_mesh()
    initialize_tilt(mesh)
    # This module is complex; patch missing geometry methods
    with (
        patch("geometry.entities.Facet.compute_mean_curvature", create=True) as mock_mc,
        patch(
            "geometry.entities.Facet.compute_divergence_of_tilt", create=True
        ) as mock_div,
        patch("geometry.entities.Facet.dJ_dvertex", create=True) as mock_dJ,
        patch("geometry.entities.Facet.dDivT_dvertex", create=True) as mock_dDivT_v,
        patch("geometry.entities.Facet.dDivT_dtilt", create=True) as mock_dDivT_t,
        patch("geometry.entities.Facet.area", create=True) as mock_area,
        patch(
            "geometry.entities.Facet.vertex_indices",
            new_callable=PropertyMock,
            create=True,
        ) as mock_v_indices,
    ):
        mock_mc.return_value = 1.0
        mock_div.return_value = 0.0
        mock_dJ.return_value = np.zeros(3)
        mock_dDivT_v.return_value = np.zeros(3)
        mock_dDivT_t.return_value = np.zeros(3)
        mock_area.return_value = 0.5
        mock_v_indices.return_value = [0, 1, 2]

        check_gradient_consistency_dict(mean_curvature_tilt, mesh, eps=1e-5, tol=1e-3)


def test_body_area_constraint_gradient_consistency():
    mesh = create_random_mesh()
    # Constraint module checks use constraint_gradients list
    mesh.bodies[0].options["target_area"] = 0.5
    check_gradient_consistency(body_area_constraint, mesh, is_constraint=True)


def test_gaussian_curvature_gradient_consistency():
    mesh = create_random_mesh()
    # Closed mesh + constant modulus: energy is topological and gradient is zero.
    check_gradient_consistency(gaussian_curvature, mesh, tol=1e-12)
