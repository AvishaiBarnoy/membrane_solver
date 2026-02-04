import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer


class FakeStepper:
    """Stepper that always tries to adapt step size upward."""

    def step(
        self,
        mesh,
        grad,
        step_size,
        energy_fn,
        constraint_enforcer=None,
    ):
        return True, float(step_size) * 2.0, float(energy_fn())


def test_fixed_step_size_mode_disables_cross_iteration_adaptation():
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "energy_modules": ["surface"],
        "global_parameters": {
            "surface_tension": 1.0,
            "step_size": 1e-3,
            "step_size_mode": "fixed",
        },
        "instructions": [],
    }
    mesh = parse_geometry(data)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        FakeStepper(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        step_size=1e-3,
    )
    minim.minimize(n_steps=2)
    assert minim.step_size == 1e-3
