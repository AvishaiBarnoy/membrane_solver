import math
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Mesh, Vertex
from geometry.geom_io import parse_geometry
from modules.energy import line_tension as line_module
from parameters.resolver import ParameterResolver
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.refinement import refine_triangle_mesh
from runtime.steppers.conjugate_gradient import ConjugateGradient
from runtime.vertex_average import vertex_average


def _build_simple_mesh():
    mesh = Mesh()
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.vertices = {0: v0, 1: v1}
    mesh.edges = {
        1: Edge(
            1,
            0,
            1,
            options={"energy": ["line_tension"], "line_tension": 2.0},
        )
    }
    mesh.energy_modules = ["line_tension"]
    mesh.global_parameters.set("line_tension", 2.0)
    mesh.build_connectivity_maps()
    return mesh


def test_line_tension_energy_and_gradient_linear_with_length():
    mesh = _build_simple_mesh()
    resolver = ParameterResolver(mesh.global_parameters)
    energy, grad = line_module.compute_energy_and_gradient(
        mesh, mesh.global_parameters, resolver
    )
    length = mesh.edges[1].compute_length(mesh)
    assert np.isclose(energy, 2.0 * length)
    assert set(grad.keys()) == {0, 1}
    g0 = grad[0]
    g1 = grad[1]
    assert np.allclose(g0 + g1, np.zeros(3))
    assert g0[0] < 0.0
    assert g1[0] > 0.0


def test_line_tension_shrinks_segment_during_minimization():
    mesh = _build_simple_mesh()
    gp = mesh.global_parameters
    em = EnergyModuleManager(mesh.energy_modules)
    cm = ConstraintModuleManager(mesh.constraint_modules)
    minimizer = Minimizer(
        mesh,
        gp,
        stepper=ConjugateGradient(),
        energy_manager=em,
        constraint_manager=cm,
        quiet=True,
    )
    initial_length = mesh.edges[1].compute_length(mesh)
    minimizer.step_size = 5e-2
    minimizer.minimize(n_steps=50)
    final_length = mesh.edges[1].compute_length(mesh)
    assert final_length < initial_length * 0.2


def test_line_tension_flags_are_preserved_by_triangle_refinement():
    mesh = parse_geometry(
        {
            "vertices": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            "edges": [
                [0, 1, {"energy": ["line_tension"], "line_tension": 1.0}],
                [1, 2],
                [2, 0],
            ],
            "faces": [[0, 1, 2]],
            "global_parameters": {"line_tension": 1.0},
        }
    )
    mesh = refine_triangle_mesh(mesh)
    lt_edges = [
        e
        for e in mesh.edges.values()
        if isinstance(e.options.get("energy"), list)
        and "line_tension" in e.options["energy"]
    ]
    assert lt_edges, "Refined mesh should still contain line-tension edges."


def _lt_perimeter(mesh: Mesh) -> float:
    mesh.build_connectivity_maps()
    P = 0.0
    for edge in mesh.edges.values():
        opts = getattr(edge, "options", {}) or {}
        energy = opts.get("energy")
        tagged = False
        if isinstance(energy, str):
            tagged = energy == "line_tension"
        elif isinstance(energy, (list, tuple)):
            tagged = "line_tension" in energy
        if tagged:
            tail = mesh.vertices[edge.tail_index].position
            head = mesh.vertices[edge.head_index].position
            P += float(np.linalg.norm(head - tail))
    return P


def test_square_line_tension_rounds_toward_circle():
    data = {
        "vertices": [
            [0.0, 0.0, 0.0, {"constraints": ["pin_to_plane"]}],
            [1.0, 0.0, 0.0, {"constraints": ["pin_to_plane"]}],
            [1.0, 1.0, 0.0, {"constraints": ["pin_to_plane"]}],
            [0.0, 1.0, 0.0, {"constraints": ["pin_to_plane"]}],
        ],
        "edges": [
            [0, 1, {"energy": ["line_tension"]}],
            [1, 2, {"energy": ["line_tension"]}],
            [2, 3, {"energy": ["line_tension"]}],
            [3, 0, {"energy": ["line_tension"]}],
        ],
        "faces": [[0, 1, 2, 3]],
        "bodies": {
            "faces": [[0]],
            "energy": [{"constraints": ["body_area"], "target_area": 1.0}],
        },
        "global_parameters": {
            "line_tension": 0.5,
            "surface_tension": 0.0,
            "volume_constraint_mode": "lagrange",
            "volume_projection_during_minimization": False,
        },
    }
    mesh = parse_geometry(data)
    gp = mesh.global_parameters
    gp.set("step_size", 5e-3)
    em = EnergyModuleManager(mesh.energy_modules)
    cm = ConstraintModuleManager(mesh.constraint_modules)
    minim = Minimizer(mesh, gp, ConjugateGradient(), em, cm, quiet=True)

    initial_perimeter = _lt_perimeter(mesh)
    minim.minimize(n_steps=40)
    mesh = refine_triangle_mesh(mesh)
    minim.mesh = mesh
    minim.enforce_constraints_after_mesh_ops(mesh)
    minim.minimize(n_steps=80)
    vertex_average(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)
    minim.minimize(n_steps=80)

    body = next(iter(mesh.bodies.values()))
    final_area = body.compute_surface_area(mesh)
    final_perimeter = _lt_perimeter(mesh)
    circle_perimeter = 2 * math.sqrt(math.pi * final_area)

    assert math.isclose(final_area, 1.0, rel_tol=1e-4, abs_tol=1e-4)
    assert final_perimeter < initial_perimeter * 0.93
    assert final_perimeter < circle_perimeter * 1.1
