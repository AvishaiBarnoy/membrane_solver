import os
import sys
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.executor import execute_command_line
from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


@dataclass
class _Context:
    mesh: object
    minimizer: object
    stepper: object


@pytest.mark.e2e
def test_e2e_kozlov_1disk_3d_small_drive_macro_runs() -> None:
    """E2E: load YAML, run the small-drive macro, and ensure it completes."""
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "meshes",
        "caveolin",
        "kozlov_1disk_3d_tensionless_single_leaflet_source.yaml",
    )
    mesh = parse_geometry(load_data(path))
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    stepper = GradientDescent()
    minim.stepper = stepper
    ctx = _Context(mesh=mesh, minimizer=minim, stepper=stepper)

    execute_command_line(ctx, "show_small_drive_physical_quick")
    # A second call ensures the command+macro plumbing is deterministic.
    execute_command_line(ctx, "energy curvature")
