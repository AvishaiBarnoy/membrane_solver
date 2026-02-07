import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Mesh, Vertex
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


class _TiltModule:
    USES_TILT = True

    def compute_energy_and_gradient_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        grad_arr,
        tilts=None,
        tilt_grad_arr=None,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map, grad_arr
        tilts = np.asarray(tilts, dtype=float)
        if tilt_grad_arr is not None:
            tilt_grad_arr += tilts
        return float(np.sum(tilts))


class _NonTiltModule:
    def compute_energy_and_gradient_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        grad_arr,
        tilts=None,
        tilt_grad_arr=None,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map, grad_arr, tilts
        _ = tilt_grad_arr
        return 10.0


def _build_single_vertex_mesh() -> Mesh:
    mesh = Mesh()
    mesh.vertices = {0: Vertex(0, np.array([0.0, 0.0, 0.0]))}
    mesh.edges = {}
    mesh.facets = {}
    mesh.energy_modules = []
    mesh.constraint_modules = []
    mesh.global_parameters = GlobalParameters()
    mesh.build_position_cache()
    return mesh


def test_tilt_only_energy_skips_non_tilt_modules() -> None:
    mesh = _build_single_vertex_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [_NonTiltModule(), _TiltModule()]

    positions = mesh.positions_view()
    tilts = np.array([[1.0, 2.0, 3.0]], dtype=float)

    energy = minim._compute_energy_array_with_tilts(positions=positions, tilts=tilts)
    assert energy == pytest.approx(float(np.sum(tilts)))

    tilt_grad = np.zeros_like(tilts)
    energy_grad = minim._compute_energy_and_tilt_gradient_array(
        positions=positions, tilts=tilts, tilt_grad_arr=tilt_grad
    )
    assert energy_grad == pytest.approx(float(np.sum(tilts)))
    assert np.allclose(tilt_grad, tilts)
