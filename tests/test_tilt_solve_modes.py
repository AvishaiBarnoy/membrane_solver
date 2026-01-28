import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _tilt_patch_input(*, solve_mode: str) -> dict:
    vertices = {
        0: [0.0, 0.0, 0.0, {"fixed": True, "tilt": [1.0, 0.0], "tilt_fixed": True}],
        1: [1.0, 0.0, 0.0, {"fixed": True, "tilt": [1.0, 0.0], "tilt_fixed": True}],
        2: [1.0, 1.0, 0.0, {"fixed": True, "tilt": [1.0, 0.0], "tilt_fixed": True}],
        3: [0.0, 1.0, 0.0, {"fixed": True, "tilt": [1.0, 0.0], "tilt_fixed": True}],
        4: [0.5, 0.5, 0.0, {"fixed": True, "tilt": [0.0, 0.0], "tilt_fixed": False}],
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
    return {
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "energy_modules": ["bending_tilt"],
        "global_parameters": {
            "surface_tension": 0.0,
            "bending_modulus": 1.0,
            "spontaneous_curvature": 0.0,
            "tilt_solve_mode": solve_mode,
            "tilt_step_size": 0.2,
            "tilt_inner_steps": 80,
            "tilt_tol": 1e-10,
            "tilt_coupled_steps": 1,
        },
        "instructions": [],
    }


def _build_minimizer(mesh) -> Minimizer:
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def test_nested_mode_relaxes_free_tilt_on_fixed_mesh():
    mesh = parse_geometry(_tilt_patch_input(solve_mode="nested"))
    minim = _build_minimizer(mesh)

    e0 = minim.compute_energy()
    t0 = mesh.vertices[4].tilt.copy()

    minim.minimize(n_steps=1)

    e1 = minim.compute_energy()
    t1 = mesh.vertices[4].tilt.copy()

    assert e1 <= e0
    assert np.linalg.norm(t1 - t0) > 1e-6
    np.testing.assert_allclose(t1, np.array([1.0, 0.0, 0.0]), atol=5e-2)

    for vid in (0, 1, 2, 3):
        np.testing.assert_allclose(mesh.vertices[vid].tilt, np.array([1.0, 0.0, 0.0]))


def test_coupled_mode_makes_progress_on_tilt_field():
    mesh = parse_geometry(_tilt_patch_input(solve_mode="coupled"))
    minim = _build_minimizer(mesh)

    e0 = minim.compute_energy()
    t0 = mesh.vertices[4].tilt.copy()

    minim.minimize(n_steps=1)

    e1 = minim.compute_energy()
    t1 = mesh.vertices[4].tilt.copy()

    assert e1 <= e0
    assert float(t1[0]) > float(t0[0])


def test_cg_solver_respects_zero_iteration_limit():
    mesh = parse_geometry(_tilt_patch_input(solve_mode="nested"))
    mesh.global_parameters.set("tilt_solver", "cg")
    mesh.global_parameters.set("tilt_cg_max_iters", 0)

    minim = _build_minimizer(mesh)

    t0 = mesh.vertices[4].tilt.copy()
    minim.minimize(n_steps=1)
    t1 = mesh.vertices[4].tilt.copy()

    np.testing.assert_allclose(t1, t0, atol=1e-12)


def test_cg_solver_reduces_energy_on_fixed_mesh():
    mesh = parse_geometry(_tilt_patch_input(solve_mode="nested"))
    mesh.global_parameters.set("tilt_solver", "cg")
    mesh.global_parameters.set("tilt_cg_max_iters", 12)
    mesh.global_parameters.set("tilt_step_size", 0.2)

    minim = _build_minimizer(mesh)

    e0 = minim.compute_energy()
    minim.minimize(n_steps=1)
    e1 = minim.compute_energy()

    assert e1 <= e0


def test_cg_solver_respects_fixed_tilts():
    mesh = parse_geometry(_tilt_patch_input(solve_mode="nested"))
    mesh.global_parameters.set("tilt_solver", "cg")
    mesh.global_parameters.set("tilt_cg_max_iters", 12)
    mesh.global_parameters.set("tilt_step_size", 0.2)

    minim = _build_minimizer(mesh)

    fixed_before = {vid: mesh.vertices[vid].tilt.copy() for vid in (0, 1, 2, 3)}
    minim.minimize(n_steps=1)
    for vid in (0, 1, 2, 3):
        np.testing.assert_allclose(mesh.vertices[vid].tilt, fixed_before[vid])
