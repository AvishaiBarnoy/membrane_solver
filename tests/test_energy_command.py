import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.context import CommandContext
from commands.executor import execute_command_line
from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _build_context(data: dict) -> CommandContext:
    mesh = parse_geometry(data)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    return CommandContext(mesh, minim, minim.stepper)


def test_energy_command_prints_breakdown(capsys):
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "energy_modules": ["surface"],
        "global_parameters": {"surface_tension": 1.0},
        "instructions": [],
    }
    ctx = _build_context(data)

    execute_command_line(ctx, "energy")
    out = capsys.readouterr().out
    assert "Current Total Energy" in out
    assert "surface" in out
