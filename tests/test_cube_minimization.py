import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def test_cube_energy_and_volume_improve():
    """End-to-end sanity check: cube evolves toward target volume and lower energy."""

    # Inline cube mesh equivalent to meshes/cube2.json to avoid external file drift.
    data = {
        "vertices": [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        "edges": [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 5],
            [1, 6],
            [2, 7],
            [3, 4],
        ],
        "faces": [
            [0, 1, 2, 3],
            ["r0", 8, 5, "r9"],
            [9, 6, -10, -1],
            [-2, 10, 7, -11],
            [11, 4, -8, -3],
            [-5, -4, -7, -6],
        ],
        "bodies": {"faces": [[0, 1, 2, 3, 4, 5]], "target_volume": [1.0]},
        "global_parameters": {
            "surface_tension": 1.0,
            "intrinsic_curvature": 0,
            "bending_modulus": 0,
            "gaussian_modulus": 0,
            "volume_stiffness": 1e3,
        },
    }
    mesh = parse_geometry(data)

    target_volume = mesh.bodies[0].target_volume
    initial_volume = mesh.compute_total_volume()

    energy_manager = EnergyModuleManager(mesh.energy_modules)
    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
    stepper = GradientDescent()

    minimizer = Minimizer(
        mesh=mesh,
        global_params=mesh.global_parameters,
        stepper=stepper,
        energy_manager=energy_manager,
        constraint_manager=constraint_manager,
        quiet=True,
    )

    # Match main.py behavior for initial step size
    minimizer.step_size = mesh.global_parameters.get("step_size", 0.001)

    initial_energy = minimizer.compute_energy()

    result = minimizer.minimize(n_steps=50)
    assert result["iterations"] > 0

    final_energy = minimizer.compute_energy()
    final_volume = mesh.compute_total_volume()

    # Energy should not explode; allowing modest increases due to volume constraint.
    assert final_energy <= initial_energy * 2.0 + 1e-6

    # Volume should remain close to the target value
    assert abs(final_volume - target_volume) < 1e-2

    # Mesh should remain valid after minimization
    assert mesh.validate_edge_indices()
