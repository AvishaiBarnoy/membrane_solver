import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


def _curved_disk_fan_input() -> dict:
    """Return a small fan mesh with a curved interior vertex tagged as 'disk'.

    This is a minimal regression for the theory-mode knob that sets the Helfrich
    base term (2H - c0) to zero on selected presets, making bending_tilt behave
    like a div(t)^2-only penalty on that patch.
    """
    vertices = {
        0: [
            0.0,
            0.0,
            0.2,
            {
                "preset": "disk",
                "tilt_in": [0.0, 0.0],
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
                "preset": "rim",
                "tilt_in": [0.0, 0.0],
                "tilt_out": [0.0, 0.0],
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
            },
        ],
        2: [
            0.0,
            1.0,
            0.0,
            {
                "preset": "rim",
                "tilt_in": [0.0, 0.0],
                "tilt_out": [0.0, 0.0],
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
            },
        ],
        3: [
            -1.0,
            0.0,
            0.0,
            {
                "preset": "rim",
                "tilt_in": [0.0, 0.0],
                "tilt_out": [0.0, 0.0],
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
            },
        ],
        4: [
            0.0,
            -1.0,
            0.0,
            {
                "preset": "rim",
                "tilt_in": [0.0, 0.0],
                "tilt_out": [0.0, 0.0],
                "tilt_fixed_in": True,
                "tilt_fixed_out": True,
            },
        ],
    }

    # Explicit edges are 1-based ids.
    edges = {
        1: [1, 2],
        2: [2, 3],
        3: [3, 4],
        4: [4, 1],
        5: [0, 1],
        6: [0, 2],
        7: [0, 3],
        8: [0, 4],
    }
    faces = {
        0: [5, 1, "r6"],
        1: [6, 2, "r7"],
        2: [7, 3, "r8"],
        3: [8, 4, "r5"],
    }

    return {
        "definitions": {
            "disk": {},
            "rim": {},
        },
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "energy_modules": ["bending_tilt_in"],
        "global_parameters": {
            "surface_tension": 0.0,
            "bending_energy_model": "helfrich",
            "spontaneous_curvature": 0.0,
            "bending_modulus_in": 1.0,
            "tilt_modulus_in": 1.0,
        },
        "instructions": [],
    }


def test_bending_tilt_assume_J0_presets_zeroes_base_term_on_disk_patch() -> None:
    mesh = parse_geometry(_curved_disk_fan_input())
    gp = mesh.global_parameters
    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-12,
    )

    e_with_base = float(minim.compute_energy())
    assert e_with_base > 1e-8

    gp.set("bending_tilt_assume_J0_presets_in", ["disk"])
    e_J0 = float(minim.compute_energy())
    assert e_J0 == pytest.approx(0.0, abs=1e-12)
