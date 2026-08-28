import numpy as np
import pytest

from core.parameters.global_parameters import GlobalParameters
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent
from tests.sample_meshes import tetra_mesh_with_body as _tetra_mesh_with_body


def test_array_and_dict_pipelines_match_directional_derivative():
    mesh = _tetra_mesh_with_body()
    gp = GlobalParameters(
        {
            "surface_tension": 1.0,
            "volume_constraint_mode": "lagrange",
            "volume_projection_during_minimization": False,
        }
    )

    energy_modules = ["surface"]
    constraint_modules = ["volume"]
    minimizer = Minimizer(
        mesh=mesh,
        global_params=gp,
        stepper=GradientDescent(max_iter=2),
        energy_manager=EnergyModuleManager(energy_modules),
        constraint_manager=ConstraintModuleManager(constraint_modules),
        energy_modules=energy_modules,
        constraint_modules=constraint_modules,
        quiet=True,
    )

    energy_arr, grad_arr = minimizer.compute_energy_and_gradient_array()
    energy_dict, grad_dict = minimizer.compute_energy_and_gradient_dict()
    assert float(energy_arr) == pytest.approx(float(energy_dict), rel=1e-12, abs=1e-12)

    positions = mesh.positions_view()
    rng = np.random.default_rng(0)
    direction = rng.normal(size=positions.shape)
    direction /= float(np.linalg.norm(direction))

    dot_arr = float(np.sum(grad_arr * direction))

    idx_map = mesh.vertex_index_to_row
    dot_dict = 0.0
    for vidx, gvec in grad_dict.items():
        row = idx_map.get(vidx)
        if row is None:
            continue
        dot_dict += float(np.dot(gvec, direction[row]))

    assert dot_arr == pytest.approx(dot_dict, rel=1e-10, abs=1e-10)
