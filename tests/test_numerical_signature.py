import numpy as np
import pytest

from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent
from tests.sample_meshes import (
    square_mesh_with_center as _build_square_mesh_with_center,
)


def test_surface_relaxation_fixed_steps_signature() -> None:
    """Numerical signature: fixed seed + fixed N steps yields stable observables."""
    np.random.seed(0)

    mesh = _build_square_mesh_with_center(z_offset=0.2)
    for vid in range(4):
        mesh.vertices[vid].fixed = True

    mesh.energy_modules = ["surface"]
    mesh.constraint_modules = []
    mesh.global_parameters.set("surface_tension", 1.0)
    mesh.global_parameters.set("step_size_mode", "fixed")
    mesh.global_parameters.set("step_size", 2e-2)

    minimizer = Minimizer(
        mesh=mesh,
        global_params=mesh.global_parameters,
        stepper=GradientDescent(max_iter=10),
        energy_manager=EnergyModuleManager(mesh.energy_modules),
        constraint_manager=ConstraintModuleManager(mesh.constraint_modules),
        step_size=2e-2,
        tol=-1.0,
        quiet=True,
    )

    for _ in range(100):
        minimizer.minimize(n_steps=1)

    energy = minimizer.compute_energy()
    center_z = float(mesh.vertices[4].position[2])

    # Tight signature tolerances: intended to catch changes in line search,
    # surface gradients, or caching behavior.
    assert energy == pytest.approx(1.0000000050010467, rel=0.0, abs=1e-10)
    assert center_z == pytest.approx(5.0005233763034544e-05, rel=0.0, abs=1e-10)
