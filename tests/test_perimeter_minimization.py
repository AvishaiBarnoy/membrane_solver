import math

import numpy as np

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.equiangulation import equiangulate_mesh
from runtime.minimizer import Minimizer
from runtime.refinement import refine_polygonal_facets
from runtime.steppers.conjugate_gradient import ConjugateGradient
from sample_meshes import square_perimeter_input


def _build_minimizer(data: dict, step_size: float = 1e-2) -> Minimizer:
    mesh = parse_geometry(data)
    mesh.global_parameters.set("step_size", step_size)
    minimizer = Minimizer(
        mesh=mesh,
        global_params=mesh.global_parameters,
        stepper=ConjugateGradient(),
        energy_manager=EnergyModuleManager(mesh.energy_modules),
        constraint_manager=ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    return minimizer


def _loop_perimeter(mesh):
    constraints = mesh.global_parameters.get("perimeter_constraints", [])
    assert constraints, "Test mesh should include perimeter constraints."
    spec = constraints[0]
    edges = spec["edges"]
    perimeter = 0.0
    for signed_idx in edges:
        edge = mesh.get_edge(signed_idx)
        tail = mesh.vertices[edge.tail_index].position
        head = mesh.vertices[edge.head_index].position
        perimeter += float(np.linalg.norm(head - tail))
    return perimeter, float(spec["target_perimeter"])


def test_perimeter_minimization_returns_to_target_length():
    data = square_perimeter_input()
    minim = _build_minimizer(data, step_size=1e-2)

    # Distort the loop so perimeter & area deviate from the target.
    minim.mesh.vertices[1].position += np.array([0.25, -0.10, 0.05])
    minim.mesh.vertices[2].position += np.array([0.15, 0.20, -0.05])

    initial_energy = minim.compute_energy()
    initial_perimeter, target = _loop_perimeter(minim.mesh)
    initial_area = minim.mesh.compute_total_surface_area()
    assert not math.isclose(initial_perimeter, target, rel_tol=1e-3)

    result = minim.minimize(n_steps=40)
    assert result["iterations"] > 0

    final_energy = minim.compute_energy()
    final_perimeter, target = _loop_perimeter(minim.mesh)

    # Energy should not grow significantly.
    assert final_energy <= initial_energy + 1e-3

    # Perimeter should be driven back very close to the target, and certainly
    # closer than in the distorted configuration.
    assert abs(final_perimeter - target) <= abs(initial_perimeter - target) + 1e-9
    assert math.isclose(final_perimeter, target, rel_tol=1e-2, abs_tol=1e-2)

    # The body-level area constraint should keep the square near unit area
    # and improve it relative to the distorted configuration.
    final_area = minim.mesh.compute_total_surface_area()
    assert abs(final_area - 1.0) <= abs(initial_area - 1.0) + 1e-6
    assert math.isclose(final_area, 1.0, rel_tol=5e-2, abs_tol=5e-2)


def test_perimeter_constraint_survives_refinement_and_equiangulation():
    data = square_perimeter_input()
    minim = _build_minimizer(data, step_size=5e-3)

    # Perturb vertices then refine/equiangulate to exercise mesh ops.
    minim.mesh.vertices[0].position += np.array([-0.15, 0.1, 0.05])
    minim.mesh.vertices[2].position += np.array([0.2, 0.1, -0.05])

    mesh = refine_polygonal_facets(minim.mesh)
    mesh = equiangulate_mesh(mesh)
    minim.mesh = mesh

    initial_energy = minim.compute_energy()
    perimeter, target = _loop_perimeter(minim.mesh)
    initial_perimeter = perimeter
    initial_area = minim.mesh.compute_total_surface_area()
    assert not math.isclose(perimeter, target, rel_tol=1e-3)

    res = minim.minimize(n_steps=40)
    assert res["iterations"] > 0

    final_energy = minim.compute_energy()
    perimeter, target = _loop_perimeter(minim.mesh)
    assert final_energy <= initial_energy + 1e-3
    # Refinement/equiangulation introduce small geometric changes, so accept
    # a slightly looser tolerance here while still requiring a substantial
    # improvement relative to the distorted configuration.
    assert abs(perimeter - target) <= abs(initial_perimeter - target)
    assert math.isclose(perimeter, target, rel_tol=1e-2, abs_tol=1e-2)
    assert minim.mesh.validate_edge_indices()

    final_area = minim.mesh.compute_total_surface_area()
    assert abs(final_area - 1.0) <= abs(initial_area - 1.0) + 1e-6
    assert math.isclose(final_area, 1.0, rel_tol=5e-2, abs_tol=5e-2)
