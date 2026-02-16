import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.geom_io import parse_geometry
from modules.energy import bending
from modules.energy import bending_tilt_leaflet as bt_leaflet
from tests.sample_meshes import SAMPLE_GEOMETRY


def _planar_patch_with_center() -> dict:
    """Return a flat square split into four triangles around a center vertex."""
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


@pytest.mark.parametrize(
    "module_name",
    ["modules.energy.bending_tilt_in", "modules.energy.bending_tilt_out"],
)
def test_bending_tilt_leaflet_matches_bending_when_tilts_zero(
    module_name: str,
) -> None:
    """Leaflet bending-tilt matches pure bending when tilts are zero."""
    module = importlib.import_module(module_name)
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
    e_bt = module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=g_bt,
    )

    assert float(e_bt) == pytest.approx(float(e_b), rel=1e-12, abs=1e-12)
    assert g_bt == pytest.approx(g_b, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize(
    "module_name,tilt_key,grad_key",
    [
        ("modules.energy.bending_tilt_in", "tilts_in", "tilt_in_grad_arr"),
        ("modules.energy.bending_tilt_out", "tilts_out", "tilt_out_grad_arr"),
    ],
)
def test_bending_tilt_leaflet_gradient_matches_directional_derivative(
    module_name: str, tilt_key: str, grad_key: str
) -> None:
    """Leaflet tilt gradients match directional derivatives on a flat patch."""
    module = importlib.import_module(module_name)
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
    module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
        **{tilt_key: tilts, grad_key: tilt_grad},
    )

    direction = np.zeros_like(tilts)
    center_row = idx_map[4]
    direction[center_row] = rng.normal(size=3)
    direction[center_row, 2] = 0.0
    direction /= float(np.linalg.norm(direction))

    eps = 1e-6
    e_plus = module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=np.zeros_like(positions),
        **{tilt_key: tilts + eps * direction},
    )
    e_minus = module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=np.zeros_like(positions),
        **{tilt_key: tilts - eps * direction},
    )

    numeric = (float(e_plus) - float(e_minus)) / (2.0 * eps)
    analytic = float(np.sum(tilt_grad * direction))
    scale = max(1.0, abs(analytic), abs(numeric))
    assert abs(analytic - numeric) / scale < 5e-5


def test_bending_tilt_leaflet_uses_leaflet_bending_modulus() -> None:
    """Leaflet modules honor bending_modulus_in/out when set."""
    mesh = parse_geometry(_planar_patch_with_center())
    gp = GlobalParameters(
        {
            "bending_modulus": 0.0,
            "bending_modulus_in": 1.0,
            "bending_modulus_out": 2.0,
            "spontaneous_curvature": 0.0,
            "bending_energy_model": "helfrich",
        }
    )
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    rng = np.random.default_rng(1)
    tilts = rng.normal(size=positions.shape)
    tilts[:, 2] = 0.0

    mod_in = importlib.import_module("modules.energy.bending_tilt_in")
    mod_out = importlib.import_module("modules.energy.bending_tilt_out")

    e_in = mod_in.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=np.zeros_like(positions),
        tilts_in=tilts,
    )
    e_out = mod_out.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=np.zeros_like(positions),
        tilts_out=tilts,
    )

    assert float(e_out) == pytest.approx(2.0 * float(e_in), rel=1e-6, abs=1e-12)


def test_bending_tilt_leaflet_uses_scalar_scatter_helper(monkeypatch) -> None:
    module = importlib.import_module("modules.energy.bending_tilt_in")
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
    grad = np.zeros_like(positions)

    called = {"count": 0}
    original = bt_leaflet.scatter_triangle_scalar_to_vertices

    def _wrapped(*args, **kwargs):
        called["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(bt_leaflet, "scatter_triangle_scalar_to_vertices", _wrapped)

    _ = module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad,
    )
    assert called["count"] >= 1
