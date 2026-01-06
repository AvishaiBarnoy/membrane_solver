import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.curvature import compute_angle_defects
from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from modules.energy import gaussian_curvature
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def create_tetrahedron_mesh() -> Mesh:
    """Create a closed tetrahedron mesh (topological sphere)."""
    mesh = Mesh()
    pts = np.array(
        [[0.1, 0.2, 0.05], [1.1, -0.1, 0.3], [0.4, 1.2, -0.2], [0.5, 0.4, 1.5]]
    )
    for i, p in enumerate(pts):
        mesh.vertices[i] = Vertex(i, p)

    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    edge_map: dict[tuple[int, int], int] = {}
    next_eid = 1
    for fidx, tri in enumerate(faces):
        e_ids: list[int] = []
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = tuple(sorted((a, b)))
            if key not in edge_map:
                edge_map[key] = next_eid
                mesh.edges[next_eid] = Edge(next_eid, a, b)
                next_eid += 1
            eid = edge_map[key]
            e_ids.append(eid if mesh.edges[eid].tail_index == a else -eid)
        mesh.facets[fidx] = Facet(fidx, e_ids)

    mesh.bodies[0] = Body(0, list(mesh.facets.keys()), target_volume=0.5)
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def test_angle_defects_sum_to_gauss_bonnet():
    mesh = create_tetrahedron_mesh()
    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    defects = compute_angle_defects(mesh, positions, idx_map)
    assert np.isclose(float(np.sum(defects)), 4.0 * np.pi, atol=1e-6)


def test_gaussian_energy_is_topological_constant_and_zero_gradient():
    mesh = create_tetrahedron_mesh()
    mesh.energy_modules = ["gaussian_curvature"]

    gp = GlobalParameters()
    gp.set("gaussian_modulus", 2.5)
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)

    E = gaussian_curvature.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
    )

    assert np.isclose(float(E), 4.0 * np.pi * 2.5, atol=1e-12)
    assert float(np.max(np.abs(grad_arr))) == 0.0


def test_gaussian_term_offsets_energy_but_not_gradient():
    mesh = create_tetrahedron_mesh()
    mesh.constraint_modules = []

    gp = GlobalParameters()
    gp.surface_tension = 1.0
    gp.gaussian_modulus = 3.0

    stepper = GradientDescent()
    constraint_manager = ConstraintModuleManager([])

    energy_manager_surface = EnergyModuleManager(["surface"])
    minim_surface = Minimizer(
        mesh,
        gp,
        stepper,
        energy_manager_surface,
        constraint_manager,
        energy_modules=["surface"],
        constraint_modules=[],
    )
    e0, g0 = minim_surface.compute_energy_and_gradient_array()

    energy_manager_full = EnergyModuleManager(["surface", "gaussian_curvature"])
    minim_full = Minimizer(
        mesh,
        gp,
        stepper,
        energy_manager_full,
        constraint_manager,
        energy_modules=["surface", "gaussian_curvature"],
        constraint_modules=[],
    )
    e1, g1 = minim_full.compute_energy_and_gradient_array()

    assert np.allclose(g0, g1, atol=0.0)
    assert np.isclose(float(e1 - e0), 4.0 * np.pi * 3.0, atol=1e-12)


def test_gaussian_energy_open_surface_uses_boundary_terms():
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
        0: Facet(0, [1, 2, -5]),
        1: Facet(1, [5, 3, 4]),
    }
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()

    gp = GlobalParameters()
    gp.gaussian_modulus = 1.0
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)

    energy = gaussian_curvature.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
    )

    assert np.isclose(float(energy), 2.0 * np.pi, atol=1e-6)
    assert float(np.max(np.abs(grad_arr))) == 0.0


def test_gaussian_energy_annulus_cancels_boundary_loops():
    from sample_meshes import square_annulus_mesh

    mesh = square_annulus_mesh()

    gp = GlobalParameters()
    gp.gaussian_modulus = 1.0
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)

    energy = gaussian_curvature.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
    )

    assert np.isclose(float(energy), 0.0, atol=1e-6)
    assert float(np.max(np.abs(grad_arr))) == 0.0
