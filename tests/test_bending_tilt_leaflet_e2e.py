import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _leaflet_patch_input(*, module: str) -> dict:
    """Return a planar patch with a single active leaflet and fixed boundary tilts."""
    vertices = {
        0: [
            0.0,
            0.0,
            0.0,
            {
                "fixed": True,
                "tilt_in": [1.0, 0.0],
                "tilt_out": [0.0, 0.0],
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
            },
        ],
        1: [
            1.0,
            0.0,
            0.0,
            {
                "fixed": True,
                "tilt_in": [0.0, 0.0],
                "tilt_out": [0.0, 0.0],
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
            },
        ],
        2: [
            1.0,
            1.0,
            0.0,
            {
                "fixed": True,
                "tilt_in": [0.0, 0.0],
                "tilt_out": [0.0, 0.0],
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
            },
        ],
        3: [
            0.0,
            1.0,
            0.0,
            {
                "fixed": True,
                "tilt_in": [0.0, 0.0],
                "tilt_out": [0.0, 0.0],
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
            },
        ],
        4: [
            0.5,
            0.5,
            0.0,
            {
                "fixed": True,
                "tilt_in": [0.0, 0.0],
                "tilt_out": [0.0, 0.0],
                "tilt_fixed_in": False,
                "tilt_fixed_out": False,
            },
        ],
    }
    edges = {
        1: [0, 1],
        2: [1, 2],
        3: [2, 3],
        4: [3, 0],
        5: [0, 4],
        6: [1, 4],
        7: [2, 4],
        8: [3, 4],
    }
    faces = {
        0: [1, 6, "r5"],
        1: [2, 7, "r6"],
        2: [3, 8, "r7"],
        3: [4, 5, "r8"],
    }

    if module == "bending_tilt_in":
        vertices[0][3]["tilt_fixed_out"] = True
        vertices[1][3]["tilt_fixed_out"] = True
        vertices[2][3]["tilt_fixed_out"] = True
        vertices[3][3]["tilt_fixed_out"] = True
        vertices[4][3]["tilt_fixed_out"] = True
    else:
        vertices[0][3]["tilt_fixed_in"] = True
        vertices[1][3]["tilt_fixed_in"] = True
        vertices[2][3]["tilt_fixed_in"] = True
        vertices[3][3]["tilt_fixed_in"] = True
        vertices[4][3]["tilt_fixed_in"] = True
        vertices[0][3]["tilt_out"] = [1.0, 0.0]

    return {
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "energy_modules": [module],
        "global_parameters": {
            "surface_tension": 0.0,
            "bending_energy_model": "helfrich",
            "spontaneous_curvature": 0.0,
            "bending_modulus": 1.0,
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.35,
            "tilt_inner_steps": 120,
            "tilt_tol": 1e-12,
        },
        "instructions": [],
    }


def _build_minimizer(mesh) -> Minimizer:
    """Return a minimizer for the provided mesh."""
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def test_bending_tilt_in_relaxes_inner_leaflet_only() -> None:
    """E2E: bending_tilt_in relaxes tilt_in without altering tilt_out."""
    mesh = parse_geometry(_leaflet_patch_input(module="bending_tilt_in"))
    minim = _build_minimizer(mesh)

    t_in_before = mesh.vertices[4].tilt_in.copy()
    t_out_before = mesh.vertices[4].tilt_out.copy()
    e0 = minim.compute_energy()

    minim.minimize(n_steps=1)

    e1 = minim.compute_energy()
    t_in_after = mesh.vertices[4].tilt_in.copy()
    t_out_after = mesh.vertices[4].tilt_out.copy()

    assert e1 <= e0
    assert np.linalg.norm(t_in_after - t_in_before) > 1e-6
    np.testing.assert_allclose(t_out_after, t_out_before, atol=1e-12)


def test_bending_tilt_out_relaxes_outer_leaflet_only() -> None:
    """E2E: bending_tilt_out relaxes tilt_out without altering tilt_in."""
    mesh = parse_geometry(_leaflet_patch_input(module="bending_tilt_out"))
    minim = _build_minimizer(mesh)

    t_in_before = mesh.vertices[4].tilt_in.copy()
    t_out_before = mesh.vertices[4].tilt_out.copy()
    e0 = minim.compute_energy()

    minim.minimize(n_steps=1)

    e1 = minim.compute_energy()
    t_in_after = mesh.vertices[4].tilt_in.copy()
    t_out_after = mesh.vertices[4].tilt_out.copy()

    assert e1 <= e0
    assert np.linalg.norm(t_out_after - t_out_before) > 1e-6
    np.testing.assert_allclose(t_in_after, t_in_before, atol=1e-12)
