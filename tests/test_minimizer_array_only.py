import os
import sys

import pytest

sys.path.insert(0, os.getcwd())
from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


class _DictOnlyModule:
    @staticmethod
    def compute_energy_and_gradient(mesh, global_params, param_resolver):
        return 0.0, {}


def test_minimizer_rejects_dict_only_energy_modules() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_free_disk_coarse_refinable.yaml")
    )
    energy_manager = EnergyModuleManager([])
    energy_manager.modules["dict_only"] = _DictOnlyModule

    with pytest.raises(TypeError, match="compute_energy_and_gradient_array"):
        Minimizer(
            mesh,
            mesh.global_parameters,
            GradientDescent(),
            energy_manager,
            ConstraintModuleManager([]),
            energy_modules=["dict_only"],
            constraint_modules=[],
            quiet=True,
        )
