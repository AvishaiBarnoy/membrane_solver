from sample_meshes import cube_soft_volume_input

from commands.context import CommandContext
from commands.io import PropertiesCommand
from commands.mesh_ops import RefineCommand
from commands.registry import get_command
from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.conjugate_gradient import ConjugateGradient


def _build_context():
    mesh = parse_geometry(cube_soft_volume_input("penalty"))
    mesh.global_parameters.set("step_size", 1e-2)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        ConjugateGradient(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    return CommandContext(mesh, minim, minim.stepper)


def test_get_command_parsing():
    # Test 'r5' -> RefineCommand with args=['5']
    cmd, args = get_command("r5")
    assert isinstance(cmd, RefineCommand)
    assert args == ["5"]


def test_get_command_aliases():
    # Test 'i' -> PropertiesCommand
    cmd, args = get_command("i")
    assert isinstance(cmd, PropertiesCommand)

    cmd, args = get_command("props")
    assert isinstance(cmd, PropertiesCommand)


def test_execute_refine_command():
    ctx = _build_context()
    facets_before = len(ctx.mesh.facets)

    cmd, args = get_command("r2")
    # r2 returns args=['2']
    cmd.execute(ctx, args)

    assert len(ctx.mesh.facets) > facets_before
