import os
import sys

sys.path.append(os.getcwd())

from geometry.geom_io import load_data, parse_geometry
from modules.energy import bending


def _helfrich_energy(path: str) -> float:
    data = load_data(path)
    mesh = parse_geometry(data)
    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    return bending.compute_total_energy(
        mesh, mesh.global_parameters, positions, idx_map
    )


def test_bench_helfrich_sphere_match_energy_near_zero():
    energy = _helfrich_energy("benchmarks/inputs/bench_helfrich_sphere_match.json")
    assert energy < 1e-10


def test_bench_helfrich_spherical_cap_match_energy_small():
    energy = _helfrich_energy(
        "benchmarks/inputs/bench_helfrich_spherical_cap_match.json"
    )
    # Discrete cap is only approximately constant-curvature; expect small but nonzero energy.
    assert energy < 2e-3


def test_bench_helfrich_local_patch_energy_is_positive():
    energy_uniform = _helfrich_energy(
        "benchmarks/inputs/bench_helfrich_sphere_match.json"
    )
    energy_patch = _helfrich_energy("benchmarks/inputs/bench_helfrich_local_patch.json")
    assert energy_uniform < 1e-10
    assert energy_patch > 1e-3


def test_helfrich_energy_scales_like_c0_squared_on_near_minimal_surface():
    """For H≈0 surfaces, Helfrich energy should scale ~ c0^2."""
    from runtime.constraint_manager import ConstraintModuleManager
    from runtime.energy_manager import EnergyModuleManager
    from runtime.minimizer import Minimizer
    from runtime.steppers.gradient_descent import GradientDescent

    data = load_data("meshes/catenoid.json")
    mesh = parse_geometry(data)

    mesh.global_parameters.set("surface_tension", 1.0)
    mesh.global_parameters.set("step_size", 1e-2)

    em = EnergyModuleManager(mesh.energy_modules)
    cm = ConstraintModuleManager(mesh.constraint_modules)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        em,
        cm,
        quiet=True,
        step_size=1e-2,
        tol=1e-10,
    )
    minim.minimize(n_steps=150)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row

    gp = mesh.global_parameters
    gp.set("bending_modulus", 1.0)
    gp.set("bending_energy_model", "helfrich")

    gp.set("spontaneous_curvature", 1.0)
    e1 = bending.compute_total_energy(mesh, gp, positions, idx_map)

    gp.set("spontaneous_curvature", 2.0)
    e2 = bending.compute_total_energy(mesh, gp, positions, idx_map)

    ratio = float(e2 / max(e1, 1e-15))
    assert ratio == pytest.approx(4.0, rel=0.25)


import pytest
