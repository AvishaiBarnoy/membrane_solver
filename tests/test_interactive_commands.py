from main import parse_instructions, execute_command
from runtime.minimizer import Minimizer
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
from runtime.steppers.conjugate_gradient import ConjugateGradient
from sample_meshes import cube_soft_volume_input
from geometry.geom_io import parse_geometry


def _build_minimizer():
    mesh = parse_geometry(cube_soft_volume_input("penalty"))
    mesh.global_parameters.set("step_size", 1e-2)
    return Minimizer(
        mesh,
        mesh.global_parameters,
        ConjugateGradient(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def test_parse_instructions_accepts_refine_counts():
    assert parse_instructions('r5') == ['r5']


def test_parse_instructions_properties_alias():
    assert parse_instructions('i') == ['properties']


def test_execute_command_multiple_refine_passes():
    minim = _build_minimizer()
    mesh = minim.mesh
    facets_before = len(mesh.facets)
    mesh, _ = execute_command('r2', mesh, minim, minim.stepper)
    assert len(mesh.facets) > facets_before
