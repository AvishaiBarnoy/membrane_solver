import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _tilt_leaflet_patch_input(*, solve_mode: str) -> dict:
    vertices = {
        0: [
            0.0,
            0.0,
            0.0,
            {
                "fixed": True,
                "tilt_in": [1.0, 0.0],
                "tilt_out": [0.6, 0.2],
                "tilt_fixed_in": True,
                "tilt_fixed_out": False,
            },
        ],
        1: [
            1.0,
            0.0,
            0.0,
            {
                "fixed": True,
                "tilt_in": [-0.5, 0.4],
                "tilt_out": [-0.8, 0.1],
                "tilt_fixed_in": False,
                "tilt_fixed_out": True,
            },
        ],
        2: [
            0.0,
            1.0,
            0.0,
            {
                "fixed": True,
                "tilt_in": [0.2, -0.1],
                "tilt_out": [0.2, -0.1],
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
            },
        ],
    }
    edges = {1: [0, 1], 2: [1, 2], 3: [2, 0]}
    faces = {0: [1, 2, 3]}
    return {
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "energy_modules": ["tilt_in", "tilt_out"],
        "global_parameters": {
            "tilt_modulus_in": 1.0,
            "tilt_modulus_out": 1.0,
            "tilt_solve_mode": solve_mode,
            "tilt_step_size": 0.25,
            "tilt_inner_steps": 50,
            "tilt_tol": 1e-12,
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


def test_leaflet_tilt_relaxation_respects_fixed_masks() -> None:
    mesh = parse_geometry(_tilt_leaflet_patch_input(solve_mode="nested"))
    minim = _build_minimizer(mesh)

    t_in_fixed = mesh.vertices[0].tilt_in.copy()
    t_out_free = mesh.vertices[0].tilt_out.copy()
    t_in_free = mesh.vertices[1].tilt_in.copy()
    t_out_fixed = mesh.vertices[1].tilt_out.copy()

    minim.minimize(n_steps=1)

    np.testing.assert_allclose(mesh.vertices[0].tilt_in, t_in_fixed, atol=1e-12)
    assert np.linalg.norm(mesh.vertices[0].tilt_out) < np.linalg.norm(t_out_free)
    assert np.linalg.norm(mesh.vertices[1].tilt_in) < np.linalg.norm(t_in_free)
    np.testing.assert_allclose(mesh.vertices[1].tilt_out, t_out_fixed, atol=1e-12)


def test_leaflet_cg_zero_iters_is_noop() -> None:
    mesh = parse_geometry(_tilt_leaflet_patch_input(solve_mode="nested"))
    mesh.global_parameters.set("tilt_solver", "cg")
    mesh.global_parameters.set("tilt_cg_max_iters", 0)
    minim = _build_minimizer(mesh)

    t_in0 = mesh.vertices[0].tilt_in.copy()
    t_out0 = mesh.vertices[0].tilt_out.copy()
    t_in1 = mesh.vertices[1].tilt_in.copy()
    t_out1 = mesh.vertices[1].tilt_out.copy()

    minim.minimize(n_steps=1)

    np.testing.assert_allclose(mesh.vertices[0].tilt_in, t_in0, atol=1e-12)
    np.testing.assert_allclose(mesh.vertices[0].tilt_out, t_out0, atol=1e-12)
    np.testing.assert_allclose(mesh.vertices[1].tilt_in, t_in1, atol=1e-12)
    np.testing.assert_allclose(mesh.vertices[1].tilt_out, t_out1, atol=1e-12)


def test_leaflet_cg_reduces_energy() -> None:
    mesh = parse_geometry(_tilt_leaflet_patch_input(solve_mode="nested"))
    mesh.global_parameters.set("tilt_solver", "cg")
    mesh.global_parameters.set("tilt_cg_max_iters", 12)
    mesh.global_parameters.set("tilt_step_size", 0.2)
    minim = _build_minimizer(mesh)

    e0 = minim.compute_energy()
    minim.minimize(n_steps=1)
    e1 = minim.compute_energy()

    assert e1 <= e0


def test_leaflet_cg_respects_fixed_tilts() -> None:
    mesh = parse_geometry(_tilt_leaflet_patch_input(solve_mode="nested"))
    mesh.global_parameters.set("tilt_solver", "cg")
    mesh.global_parameters.set("tilt_cg_max_iters", 12)
    mesh.global_parameters.set("tilt_step_size", 0.2)
    minim = _build_minimizer(mesh)

    t0_in = mesh.vertices[0].tilt_in.copy()
    t1_out = mesh.vertices[1].tilt_out.copy()
    t2_in = mesh.vertices[2].tilt_in.copy()
    t2_out = mesh.vertices[2].tilt_out.copy()

    minim.minimize(n_steps=1)

    np.testing.assert_allclose(mesh.vertices[0].tilt_in, t0_in, atol=1e-12)
    np.testing.assert_allclose(mesh.vertices[1].tilt_out, t1_out, atol=1e-12)
    np.testing.assert_allclose(mesh.vertices[2].tilt_in, t2_in, atol=1e-12)
    np.testing.assert_allclose(mesh.vertices[2].tilt_out, t2_out, atol=1e-12)
