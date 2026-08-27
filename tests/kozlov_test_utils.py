"""Shared construction helpers for Kozlov regression and E2E tests."""

from pathlib import Path

from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

FIXTURE_DIR = Path(__file__).with_name("fixtures")


def fixture_path(name: str) -> str:
    """Return an absolute path to a named test fixture."""
    return str(FIXTURE_DIR / name)


def build_minimizer(mesh, *, tol: float | None = None) -> Minimizer:
    """Return the standard quiet minimizer used by Kozlov scenarios."""
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
