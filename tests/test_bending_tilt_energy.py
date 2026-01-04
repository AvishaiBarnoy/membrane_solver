import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.geom_io import parse_geometry
from modules.energy import bending, bending_tilt
from tests.sample_meshes import SAMPLE_GEOMETRY


def _planar_patch_with_center() -> dict:
    return {
        "vertices": {
            0: [0.0, 0.0, 0.0, {"fixed": True}],
            1: [1.0, 0.0, 0.0, {"fixed": True}],
            2: [1.0, 1.0, 0.0, {"fixed": True}],
            3: [0.0, 1.0, 0.0, {"fixed": True}],
            4: [0.5, 0.5, 0.0, {"fixed": True}],
        },
        "edges": {
            1: [0, 1],
            2: [1, 2],
            3: [2, 3],
            4: [3, 0],
            5: [0, 4],
            6: [1, 4],
            7: [2, 4],
            8: [3, 4],
        },
        "faces": {
            0: [1, 6, "r5"],
            1: [2, 7, "r6"],
            2: [3, 8, "r7"],
            3: [4, 5, "r8"],
        },
        "energy_modules": [],
        "global_parameters": {"surface_tension": 0.0},
        "instructions": [],
    }


def test_bending_tilt_matches_bending_when_tilts_zero():
    mesh = parse_geometry(SAMPLE_GEOMETRY)
    gp = GlobalParameters(
        {
            "bending_modulus": 1.0,
            "spontaneous_curvature": 0.0,
            "bending_energy_model": "helfrich",
        }
    )
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row

    g_b = np.zeros_like(positions)
    e_b = bending.compute_energy_and_gradient_array(
        mesh, gp, resolver, positions=positions, index_map=idx_map, grad_arr=g_b
    )

    g_bt = np.zeros_like(positions)
    e_bt = bending_tilt.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=g_bt,
    )

    assert float(e_bt) == pytest.approx(float(e_b), rel=1e-12, abs=1e-12)
    assert g_bt == pytest.approx(g_b, rel=1e-10, abs=1e-12)


def test_bending_tilt_tilt_gradient_matches_directional_derivative():
    mesh = parse_geometry(_planar_patch_with_center())
    gp = GlobalParameters(
        {
            "bending_modulus": 1.0,
            "spontaneous_curvature": 0.0,
            "bending_energy_model": "helfrich",
        }
    )
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row

    rng = np.random.default_rng(0)
    tilts = rng.normal(size=positions.shape)
    tilts[:, 2] = 0.0

    grad_arr = np.zeros_like(positions)
    tilt_grad = np.zeros_like(positions)
    bending_tilt.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
        tilts=tilts,
        tilt_grad_arr=tilt_grad,
    )

    direction = np.zeros_like(tilts)
    center_row = idx_map[4]
    direction[center_row] = rng.normal(size=3)
    direction[center_row, 2] = 0.0
    direction /= float(np.linalg.norm(direction))

    eps = 1e-6
    e_plus = bending_tilt.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=np.zeros_like(positions),
        tilts=tilts + eps * direction,
    )
    e_minus = bending_tilt.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=np.zeros_like(positions),
        tilts=tilts - eps * direction,
    )

    numeric = (float(e_plus) - float(e_minus)) / (2.0 * eps)
    analytic = float(np.sum(tilt_grad * direction))
    scale = max(1.0, abs(analytic), abs(numeric))
    assert abs(analytic - numeric) / scale < 5e-5
