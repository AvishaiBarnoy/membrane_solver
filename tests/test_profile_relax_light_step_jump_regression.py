import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.context import CommandContext  # noqa: E402
from commands.executor import execute_command_line  # noqa: E402
from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


@pytest.mark.regression
def test_profile_relax_light_state_energy_matches_breakdown_after_g1() -> None:
    """Regression: stale caches must not desync scalar vs breakdown energy.

    This reproduces the profile_relax_light sequence up to the first `g1`
    after `t2e-3`. At this point, scalar total energy and energy-breakdown
    total must be evaluated on the same state.
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "meshes",
        "caveolin",
        "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml",
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
    ctx = CommandContext(mesh=mesh, minimizer=minim, stepper=minim.stepper)

    for command in ("g10", "r", "V2", "t5e-3", "g8", "t2e-3", "g1"):
        execute_command_line(ctx, command)

    scalar_energy = float(minim.compute_energy())
    breakdown_total = float(sum(minim.compute_energy_breakdown().values()))

    assert scalar_energy == pytest.approx(breakdown_total, rel=1e-9, abs=1e-9)
