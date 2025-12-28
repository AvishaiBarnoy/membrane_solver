import logging
import math

import numpy as np

from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.constraint_manager import ConstraintModuleManager
from runtime.diagnostics.gauss_bonnet import (
    GaussBonnetMonitor,
    corner_angle,
    extract_boundary_loops,
    find_boundary_edges,
    gauss_bonnet_invariant,
)
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.refinement import refine_triangle_mesh
from runtime.steppers.gradient_descent import GradientDescent


def _build_square_mesh() -> Mesh:
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
        1: Facet(1, [1, 2, -5]),
        2: Facet(2, [5, 3, 4]),
    }
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def _build_square_mesh_with_center(z_offset: float = 0.0) -> Mesh:
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([1.0, 1.0, 0.0])),
        3: Vertex(3, np.array([0.0, 1.0, 0.0])),
        4: Vertex(4, np.array([0.5, 0.5, z_offset])),
    }
    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 3),
        4: Edge(4, 3, 0),
        5: Edge(5, 0, 4),
        6: Edge(6, 1, 4),
        7: Edge(7, 2, 4),
        8: Edge(8, 3, 4),
    }
    mesh.facets = {
        1: Facet(1, [1, 6, -5]),
        2: Facet(2, [2, 7, -6]),
        3: Facet(3, [3, 8, -7]),
        4: Facet(4, [4, 5, -8]),
    }
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def test_boundary_loop_extraction_square_mesh():
    mesh = _build_square_mesh()
    boundary_edges = find_boundary_edges(mesh)
    loops = extract_boundary_loops(mesh, boundary_edges)

    assert len(loops) == 1
    loop = loops[0]
    loop_ids = [v.index for v in loop]
    assert len(loop_ids) == 4
    assert set(loop_ids) == {0, 1, 2, 3}

    edge_pairs = {
        tuple(sorted((edge.tail_index, edge.head_index))) for edge in boundary_edges
    }
    for i in range(len(loop_ids)):
        a = loop_ids[i]
        b = loop_ids[(i + 1) % len(loop_ids)]
        assert tuple(sorted((a, b))) in edge_pairs


def test_corner_angle_sum_triangle():
    mesh = _build_square_mesh_with_center()
    face = mesh.facets[1]
    angles = [corner_angle(mesh, face, vid) for vid in (0, 1, 4)]
    assert math.isclose(sum(angles), math.pi, rel_tol=1e-6, abs_tol=1e-6)


def test_gauss_bonnet_invariant_disk():
    mesh = _build_square_mesh()
    g_total, k_int_total, b_total, per_loop = gauss_bonnet_invariant(mesh)

    assert math.isclose(g_total, 2.0 * math.pi, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(k_int_total, 0.0, abs_tol=1e-6)
    assert math.isclose(b_total, 2.0 * math.pi, rel_tol=1e-6, abs_tol=1e-6)
    assert len(per_loop) == 1


def test_gauss_bonnet_refinement_stability():
    mesh = _build_square_mesh()
    g0, _, _, per_loop0 = gauss_bonnet_invariant(mesh)

    refined = refine_triangle_mesh(mesh)
    g1, _, _, per_loop1 = gauss_bonnet_invariant(refined)

    assert abs(g1 - g0) < 1e-5
    assert abs(per_loop1[0] - per_loop0[0]) < 1e-5


def test_gauss_bonnet_minimization_drift():
    mesh = _build_square_mesh_with_center(z_offset=0.2)
    for vid in range(4):
        mesh.vertices[vid].fixed = True
    mesh.energy_modules = ["surface"]
    mesh.constraint_modules = []
    mesh.global_parameters.set("surface_tension", 1.0)

    g0, _, _, per_loop0 = gauss_bonnet_invariant(mesh)

    minimizer = Minimizer(
        mesh=mesh,
        global_params=mesh.global_parameters,
        stepper=GradientDescent(max_iter=5),
        energy_manager=EnergyModuleManager(mesh.energy_modules),
        constraint_manager=ConstraintModuleManager(mesh.constraint_modules),
        step_size=1e-2,
        tol=-1.0,
        quiet=True,
    )
    minimizer.minimize(n_steps=10)

    g1, _, _, per_loop1 = gauss_bonnet_invariant(mesh)
    tol = 1e-4
    assert abs(g1 - g0) < tol
    assert per_loop1


def test_gauss_bonnet_excludes_facets():
    mesh = _build_square_mesh()
    mesh.facets[2].options["gauss_bonnet_exclude"] = True

    g_total, k_int_total, b_total, per_loop = gauss_bonnet_invariant(mesh)
    assert math.isclose(k_int_total, 0.0, abs_tol=1e-6)
    assert math.isclose(b_total, 2.0 * math.pi, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(g_total, 2.0 * math.pi, rel_tol=1e-6, abs_tol=1e-6)
    assert len(per_loop) == 1


def test_gauss_bonnet_logging_includes_loops(caplog):
    mesh = _build_square_mesh()
    monitor = GaussBonnetMonitor.from_mesh(mesh, eps_angle=1e-6)

    with caplog.at_level(logging.DEBUG, logger="membrane_solver"):
        monitor.evaluate(mesh)

    assert "Gauss-Bonnet loop" in caplog.text
