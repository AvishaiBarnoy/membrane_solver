import os
import sys

sys.path.append(os.getcwd())

from geometry.geom_io import load_data, parse_geometry
from modules.energy import bending


def test_flat_sheet_has_zero_bending_energy():
    # 4x4 flat grid
    data = {
        "vertices": [[x, y, 0] for x in range(4) for y in range(4)],
        "faces": [],  # Will let it be empty or construct simple ones
    }
    # Simple triangulation with edges
    edges = []
    faces = []
    edge_map = {}
    next_eid = 1

    def get_eid(a, b):
        nonlocal next_eid
        key = tuple(sorted((a, b)))
        if key not in edge_map:
            edge_map[key] = next_eid
            edges.append([a, b])
            next_eid += 1
        eid_1based = edge_map[key]
        # Return 0-based index for parse_edge_ref
        if edges[eid_1based - 1][0] == a:
            return eid_1based - 1
        else:
            return f"r{eid_1based - 1}"

    for i in range(3):
        for j in range(3):
            v0 = i * 4 + j
            v1 = (i + 1) * 4 + j
            v2 = (i + 1) * 4 + j + 1
            v3 = i * 4 + j + 1
            # Triangle 1: v0, v1, v2
            faces.append([get_eid(v0, v1), get_eid(v1, v2), get_eid(v2, v0)])
            # Triangle 2: v0, v2, v3
            faces.append([get_eid(v0, v2), get_eid(v2, v3), get_eid(v3, v0)])

    data["edges"] = edges
    data["faces"] = faces
    data["global_parameters"] = {
        "bending_modulus": 1.0,
        "bending_energy_model": "willmore",
    }

    mesh = parse_geometry(data)
    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row

    energy = bending.compute_total_energy(
        mesh, mesh.global_parameters, positions, idx_map
    )
    assert energy == pytest.approx(0.0, abs=1e-12)


def test_catenoid_has_near_zero_bending_energy():
    # Load catenoid which is a minimal surface (H=0)
    data = load_data("meshes/catenoid.json")
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

    data = load_data("meshes/catenoid.json")
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

    assert e1 <= e0
    assert e1 < 0.05


import pytest

if __name__ == "__main__":
    pytest.main([__file__])
