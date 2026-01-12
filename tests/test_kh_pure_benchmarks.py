import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.refinement import refine_triangle_mesh
from runtime.steppers.gradient_descent import GradientDescent


def _compute_energy(path: str) -> float:
    mesh = parse_geometry(load_data(path))
    return _compute_energy_mesh(mesh)


def _compute_energy_mesh(mesh) -> float:
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    return float(minim.compute_energy())


def test_kh_pure_divergent_field_has_nonzero_energy():
    energy = _compute_energy("meshes/tilt_benchmarks/kh_pure_curl_free.yaml")
    assert energy > 1e-4


def test_kh_pure_curl_rich_field_is_near_zero_energy():
    """Curl-rich mesh is constructed to be divergence-free, so KH-pure energy ~ 0."""
    energy = _compute_energy("meshes/tilt_benchmarks/kh_pure_curl_rich.yaml")
    assert energy == pytest.approx(0.0, abs=1e-12)


def test_kh_pure_divergent_field_energy_stable_under_refinement():
    """Divergent KH-pure mesh should stay nonzero under refinement."""
    mesh = parse_geometry(load_data("meshes/tilt_benchmarks/kh_pure_curl_free.yaml"))
    e0 = _compute_energy_mesh(mesh)
    refined = refine_triangle_mesh(mesh)
    e1 = _compute_energy_mesh(refined)

    assert e1 > 1e-4
    assert e1 == pytest.approx(e0, rel=0.1)


def test_kh_pure_curl_rich_energy_stable_under_refinement():
    """Divergence-free KH-pure mesh should remain near zero after refinement."""
    mesh = parse_geometry(load_data("meshes/tilt_benchmarks/kh_pure_curl_rich.yaml"))
    e0 = _compute_energy_mesh(mesh)
    refined = refine_triangle_mesh(mesh)
    e1 = _compute_energy_mesh(refined)

    assert e0 == pytest.approx(0.0, abs=1e-12)
    assert e1 == pytest.approx(0.0, abs=1e-10)
