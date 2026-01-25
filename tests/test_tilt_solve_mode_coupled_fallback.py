import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _single_triangle_data() -> dict:
    """Return a minimal mesh used to exercise tilt-only relaxation settings."""
    vertices = [
        [0.0, 0.0, 0.0, {"tilt_in": [1.0, 0.0], "tilt_out": [0.0, 1.0]}],
        [1.0, 0.0, 0.0, {"tilt_in": [1.0, 0.0], "tilt_out": [0.0, 1.0]}],
        [0.0, 1.0, 0.0, {"tilt_in": [1.0, 0.0], "tilt_out": [0.0, 1.0]}],
    ]
    edges = [[0, 1], [1, 2], [2, 0]]
    faces = [[0, 1, 2]]
    return {
        "global_parameters": {
            "tilt_modulus_in": 1.0,
            "tilt_modulus_out": 1.0,
            "tilt_solve_mode": "coupled",
            "tilt_step_size": 0.5,
            # Intentionally omit `tilt_coupled_steps` to ensure fallback to
            # `tilt_inner_steps` works.
            "tilt_inner_steps": 20,
            "tilt_tol": 1e-12,
            "step_size": 0.0,
            "step_size_mode": "fixed",
        },
        "constraint_modules": [],
        "energy_modules": ["tilt_in", "tilt_out"],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
    }


def test_tilt_solve_mode_coupled_uses_inner_steps_fallback() -> None:
    """`tilt_solve_mode=coupled` should relax even without `tilt_coupled_steps`."""
    mesh = parse_geometry(_single_triangle_data())
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )

    before = float(
        np.linalg.norm(mesh.tilts_in_view()) + np.linalg.norm(mesh.tilts_out_view())
    )
    minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode="coupled")
    after = float(
        np.linalg.norm(mesh.tilts_in_view()) + np.linalg.norm(mesh.tilts_out_view())
    )

    assert after < 0.25 * before
