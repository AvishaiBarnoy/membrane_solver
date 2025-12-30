import os
import sys

sys.path.append(os.getcwd())

from geometry.geom_io import load_data, parse_geometry
from modules.energy import bending


def test_flat_sheet_has_zero_bending_energy():
    data = load_data("meshes/flat_sheet_4x4.yaml")
    mesh = parse_geometry(data)
    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row

    energy = bending.compute_total_energy(
        mesh, mesh.global_parameters, positions, idx_map
    )
    assert energy == pytest.approx(0.0, abs=1e-12)


def test_catenoid_has_near_zero_bending_energy():
    # Load catenoid which is a minimal surface (H=0)
    data = load_data("meshes/catenoid.yaml")
    mesh = parse_geometry(data)
    mesh.global_parameters.set("bending_modulus", 1.0)
    mesh.global_parameters.set("bending_energy_model", "willmore")

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row

    energy = bending.compute_total_energy(
        mesh, mesh.global_parameters, positions, idx_map
    )
    # Catenoid in meshes/ might not be perfectly converged, but energy should be very low
    # compared to e.g. a sphere (4*pi*kappa ~ 12.5)
    assert energy < 0.1


def test_catenoid_surface_minimization_drives_bending_energy_toward_zero():
    from runtime.constraint_manager import ConstraintModuleManager
    from runtime.energy_manager import EnergyModuleManager
    from runtime.minimizer import Minimizer
    from runtime.steppers.gradient_descent import GradientDescent

    data = load_data("meshes/catenoid.yaml")
    mesh = parse_geometry(data)

    # Minimize surface area (soap film): minimal surfaces have H = 0, so Willmore energy ~ 0.
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

    positions0 = mesh.positions_view()
    idx_map0 = mesh.vertex_index_to_row
    gp_bend = mesh.global_parameters
    gp_bend.set("bending_modulus", 1.0)
    gp_bend.set("bending_energy_model", "willmore")
    e0 = bending.compute_total_energy(mesh, gp_bend, positions0, idx_map0)

    minim.minimize(n_steps=100)

    positions1 = mesh.positions_view()
    idx_map1 = mesh.vertex_index_to_row
    e1 = bending.compute_total_energy(mesh, gp_bend, positions1, idx_map1)

    # Discrete curvature + floating-point arithmetic can introduce tiny
    # non-monotonic drift; treat values near machine zero as equivalent.
    assert e1 <= e0 + 1e-10
    assert e1 < 0.05


import pytest

if __name__ == "__main__":
    pytest.main([__file__])
