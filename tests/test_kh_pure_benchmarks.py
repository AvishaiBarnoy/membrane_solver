import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _compute_energy(path: str) -> float:
    mesh = parse_geometry(load_data(path))
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    return float(minim.compute_energy())


def test_kh_pure_curl_free_has_nonzero_energy():
    energy = _compute_energy("meshes/tilt_benchmarks/kh_pure_curl_free.yaml")
    assert energy > 1e-4


def test_kh_pure_curl_rich_is_near_zero_energy():
    energy = _compute_energy("meshes/tilt_benchmarks/kh_pure_curl_rich.yaml")
    assert energy == pytest.approx(0.0, abs=1e-12)
