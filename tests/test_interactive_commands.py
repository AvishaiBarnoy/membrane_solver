import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_meshes import cube_soft_volume_input

from commands.context import CommandContext
from commands.executor import execute_command_line
from commands.io import PropertiesCommand
from commands.mesh_ops import RefineCommand
from commands.meta import TiltStatsCommand
from commands.registry import get_command
from geometry.geom_io import load_data, parse_geometry
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


def test_get_command_tilt_stats():
    cmd, args = get_command("tilt_stats")
    assert isinstance(cmd, TiltStatsCommand)
    assert args == []

    cmd, args = get_command("tstats")
    assert isinstance(cmd, TiltStatsCommand)
    assert args == []

    cmd, args = get_command("tilt_stat")
    assert isinstance(cmd, TiltStatsCommand)
    assert args == []

    cmd, args = get_command("tstat")
    assert isinstance(cmd, TiltStatsCommand)
    assert args == []


def test_tilt_stats_accepts_leaflet_args(capsys):
    ctx = _build_context()
    cmd, _args = get_command("tilt_stats")

    cmd.execute(ctx, ["in"])
    out = capsys.readouterr().out
    assert "tilt_in" in out

    cmd.execute(ctx, ["out"])
    out = capsys.readouterr().out
    assert "tilt_out" in out


def test_tilt_stats_is_read_only_on_leaflet_fixture() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        ConjugateGradient(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    ctx = CommandContext(mesh, minim, minim.stepper)

    e0 = float(minim.compute_energy())
    p0 = mesh.positions_view().copy()
    tin0 = mesh.tilts_in_view().copy()
    tout0 = mesh.tilts_out_view().copy()
    versions0 = (mesh._version, mesh._tilts_in_version, mesh._tilts_out_version)

    execute_command_line(ctx, "tstat in")
    execute_command_line(ctx, "tstat out")

    e1 = float(minim.compute_energy())
    np.testing.assert_allclose(mesh.positions_view(), p0, rtol=0, atol=0)
    np.testing.assert_allclose(mesh.tilts_in_view(), tin0, rtol=0, atol=0)
    np.testing.assert_allclose(mesh.tilts_out_view(), tout0, rtol=0, atol=0)
    assert e1 == e0
    assert (mesh._version, mesh._tilts_in_version, mesh._tilts_out_version) == versions0


def test_execute_refine_command():
    ctx = _build_context()
    facets_before = len(ctx.mesh.facets)

    cmd, args = get_command("r2")
    # r2 returns args=['2']
    cmd.execute(ctx, args)

    assert len(ctx.mesh.facets) > facets_before


def test_execute_compound_commands_separated_by_semicolon():
    ctx = _build_context()
    execute_command_line(ctx, "t1e-3; tf")
    assert ctx.mesh.global_parameters.get("step_size_mode") == "adaptive"
