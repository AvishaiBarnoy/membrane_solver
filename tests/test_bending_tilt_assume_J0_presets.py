import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry  # noqa: E402
from modules.energy import bending_tilt_out as bending_tilt_out_module  # noqa: E402
from modules.energy.bt_params import (  # noqa: E402
    _base_term_reference_mode,
    _bending_tilt_in_scaffold_shape_stencil_mode,
)
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


def _curved_disk_fan_out_input() -> dict:
    doc = _curved_disk_fan_input()
    doc["energy_modules"] = ["bending_tilt_out"]
    gp = dict(doc["global_parameters"])
    gp["bending_modulus_out"] = gp["bending_modulus_in"]
    gp["tilt_modulus_out"] = gp["tilt_modulus_in"]
    doc["global_parameters"] = gp
    return doc


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


def test_base_term_reference_mode_supports_leaflet_specific_override() -> None:
    params = {
        "bending_tilt_base_term_reference_mode": "flat_reference_zero_J0",
        "bending_tilt_base_term_reference_mode_out": "current_geometry",
    }
    assert _base_term_reference_mode(params, cache_tag="in") == "flat_reference_zero_j0"
    assert _base_term_reference_mode(params, cache_tag="out") == "current_geometry"
    assert _base_term_reference_mode(params) == "flat_reference_zero_j0"


def test_base_term_reference_mode_invalid_leaflet_specific_value_raises() -> None:
    params = {"bending_tilt_base_term_reference_mode_out": "bad_mode"}
    with pytest.raises(ValueError):
        _base_term_reference_mode(params, cache_tag="out")


def test_inner_scaffold_shape_stencil_mode_is_opt_in_and_validated() -> None:
    assert _bending_tilt_in_scaffold_shape_stencil_mode({}) == "off"
    assert (
        _bending_tilt_in_scaffold_shape_stencil_mode(
            {"bending_tilt_in_scaffold_shape_stencil_mode": "trace_boundary_v1"}
        )
        == "trace_boundary_v1"
    )
    with pytest.raises(ValueError):
        _bending_tilt_in_scaffold_shape_stencil_mode(
            {"bending_tilt_in_scaffold_shape_stencil_mode": "bad_mode"}
        )


def test_outer_leaflet_override_keeps_curvature_energy_when_global_mode_is_flat() -> (
    None
):
    mesh = parse_geometry(_curved_disk_fan_out_input())
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

    gp.set("bending_tilt_base_term_reference_mode", "flat_reference_zero_J0")
    e_global_flat = float(minim.compute_energy())
    assert e_global_flat == pytest.approx(0.0, abs=1e-12)

    gp.set("bending_tilt_base_term_reference_mode_out", "current_geometry")
    e_outer_current = float(minim.compute_energy())
    assert e_outer_current > 1.0e-8


def test_bending_tilt_out_gradient_matches_finite_difference() -> None:
    mesh = parse_geometry(_curved_disk_fan_out_input())
    gp = mesh.global_parameters
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    resolver = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-12,
    ).param_resolver

    tilts_out = mesh.tilts_out_view().copy(order="F")
    tilts_out[1, 0] = 0.17
    tilts_out[2, 1] = -0.09
    tilt_grad = np.zeros_like(tilts_out)
    energy = bending_tilt_out_module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=np.zeros_like(positions),
        tilts_out=tilts_out,
        tilt_out_grad_arr=tilt_grad,
    )
    assert np.isfinite(float(energy))

    row = 1
    comp = 0
    eps = 1.0e-7
    plus = tilts_out.copy(order="F")
    minus = tilts_out.copy(order="F")
    plus[row, comp] += eps
    minus[row, comp] -= eps
    e_plus = bending_tilt_out_module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=np.zeros_like(positions),
        tilts_out=plus,
        tilt_out_grad_arr=np.zeros_like(positions),
    )
    e_minus = bending_tilt_out_module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=np.zeros_like(positions),
        tilts_out=minus,
        tilt_out_grad_arr=np.zeros_like(positions),
    )
    fd = (float(e_plus) - float(e_minus)) / (2.0 * eps)
    assert tilt_grad[row, comp] == pytest.approx(fd, rel=1.0e-5, abs=1.0e-7)
