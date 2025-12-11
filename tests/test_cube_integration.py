import math

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.equiangulation import equiangulate_mesh
from runtime.minimizer import Minimizer
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh
from runtime.steppers.conjugate_gradient import ConjugateGradient
from sample_meshes import cube_soft_volume_input


def _build_minimizer(data: dict) -> Minimizer:
    mesh = parse_geometry(data)
    mesh.global_parameters.set("step_size", 1e-2)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        ConjugateGradient(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    return minim


def test_cube_penalty_minimization_reduces_energy():
    minim = _build_minimizer(cube_soft_volume_input(volume_mode="penalty"))

    initial_energy = minim.compute_energy()
    res = minim.minimize(n_steps=40)
    assert res["iterations"] > 0
    final_energy = minim.compute_energy()

    assert final_energy <= initial_energy + 1e-3

    target_volume = minim.mesh.bodies[0].options.get("target_volume", 0.0)
    final_volume = minim.mesh.compute_total_volume()
    assert math.isclose(final_volume, target_volume, rel_tol=5e-2, abs_tol=5e-2)
    assert minim.mesh.validate_edge_indices()


def test_cube_refine_and_equiangulate_remains_valid():
    minim = _build_minimizer(cube_soft_volume_input(volume_mode="penalty"))
    minim.minimize(n_steps=20)

    mesh = refine_triangle_mesh(minim.mesh)
    mesh = refine_polygonal_facets(mesh)
    mesh = equiangulate_mesh(mesh)

    minim.mesh = mesh
    minim.minimize(n_steps=20)

    assert mesh.validate_edge_indices()
    final_volume = mesh.compute_total_volume()
    target_volume = mesh.bodies[0].options.get("target_volume", 0.0)
    assert math.isclose(final_volume, target_volume, rel_tol=5e-2, abs_tol=5e-2)
