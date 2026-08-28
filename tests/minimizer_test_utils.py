"""Shared minimizer construction for focused tests."""

from commands.context import CommandContext
from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def build_minimizer(mesh, *, tol: float | None = None) -> Minimizer:
    """Return the standard quiet minimizer used by focused test scenarios."""
    kwargs = {} if tol is None else {"tol": float(tol)}
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        **kwargs,
    )


def build_command_context(data: dict) -> CommandContext:
    """Parse geometry data and attach the standard quiet minimizer."""
    mesh = parse_geometry(data)
    minimizer = build_minimizer(mesh)
    return CommandContext(mesh, minimizer, minimizer.stepper)
