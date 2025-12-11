import math

import numpy as np

from geometry.geom_io import parse_geometry
from runtime.minimizer import Minimizer
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
from runtime.steppers.conjugate_gradient import ConjugateGradient


def _run_minimization(mesh_dict, steps=20):
    mesh = parse_geometry(mesh_dict)
    # Give the optimizer a slightly larger initial step to move faster in tests.
    mesh.global_parameters.set("step_size", 1e-2)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        ConjugateGradient(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    result = minim.minimize(n_steps=steps)
    return minim.mesh, result


def _square_mesh(scale=1.0, target_area=None):
    verts = [
        [0.0, 0.0, 0.0],
        [scale, 0.0, 0.0],
        [scale, scale, 0.0],
        [0.0, scale, 0.0],
    ]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    faces = [[0, 1, 2, 3]]
    bodies = {
        "faces": [[0]],
        "energy": [
            {
                "constraints": ["body_area"],
                "target_area": target_area,
            }
        ],
    }
    return {
        "vertices": verts,
        "edges": edges,
        "faces": faces,
        "bodies": bodies,
        "global_parameters": {"surface_tension": 1.0},
        "instructions": [],
    }


def _tetra_mesh(target_volume):
    verts = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    edges = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]
    faces = [
        [0, 1, 3],
        [1, 2, 4],
        [2, 0, 5],
        [3, 4, 5],
    ]
    bodies = {
        "faces": [[0, 1, 2, 3]],
        "target_volume": [target_volume],
    }
    return {
        "vertices": verts,
        "edges": edges,
        "faces": faces,
        "bodies": bodies,
        "global_parameters": {"surface_tension": 1.0, "volume_constraint_mode": "lagrange"},
        "instructions": [],
    }


def test_square_area_constraint_reduces_to_target():
    mesh_dict = _square_mesh(scale=1.1, target_area=1.0)  # initial area > target
    mesh, _ = _run_minimization(mesh_dict, steps=30)
    area = mesh.compute_total_surface_area()
    assert math.isclose(area, 1.0, rel_tol=1e-3, abs_tol=1e-4)


def test_square_area_constraint_increases_to_target():
    mesh_dict = _square_mesh(scale=0.7, target_area=1.0)  # initial area < target
    mesh, _ = _run_minimization(mesh_dict, steps=40)
    area = mesh.compute_total_surface_area()
    assert math.isclose(area, 1.0, rel_tol=1e-3, abs_tol=1e-4)


def test_tetra_volume_constraint():
    target_vol = 0.2
    mesh_dict = _tetra_mesh(target_volume=target_vol)
    mesh, _ = _run_minimization(mesh_dict, steps=40)
    vol = mesh.compute_total_volume()
    assert math.isclose(vol, target_vol, rel_tol=1e-3, abs_tol=1e-4)
