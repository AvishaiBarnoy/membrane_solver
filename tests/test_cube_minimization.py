import os

import numpy as np
import pytest

from geometry.geom_io import load_data, parse_geometry
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def test_cube_energy_and_volume_improve():
    """End-to-end sanity check: cube evolves toward target volume and lower energy."""

    cube_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "meshes", "cube.json")
    data = load_data(cube_path)
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

    # Energy should not increase
    assert final_energy <= initial_energy + 1e-6

    # Volume should remain close to the target value
    assert abs(final_volume - target_volume) < 1e-2

    # Mesh should remain valid after minimization
    assert mesh.validate_edge_indices()
