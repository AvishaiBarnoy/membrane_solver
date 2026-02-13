import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.constraint_manager import ConstraintModuleManager


class DummyConstraint:
    def __init__(self, gC):
        self._gC = gC

    def constraint_gradients(self, mesh, global_params):
        if isinstance(self._gC, list):
            return self._gC
        return None

    def constraint_gradient(self, mesh, global_params):
        if isinstance(self._gC, list):
            return None
        return self._gC


class DummyArrayConstraint:
    def __init__(self, constraints):
        self._constraints = constraints

    def constraint_gradients_array(self, mesh, global_params, **kwargs):
        _ = mesh, global_params, kwargs
        return self._constraints


class DummyTiltConstraint:
    def __init__(self, constraints):
        self._constraints = constraints

    def constraint_gradients_tilt_array(self, mesh, global_params, **kwargs):
        _ = mesh, global_params, kwargs
        return self._constraints


class DummyMesh:
    def __init__(self, n_rows: int):
        self._positions = np.zeros((n_rows, 3), dtype=float)
        self.vertex_index_to_row = {i: i for i in range(n_rows)}

    def build_position_cache(self):
        return None

    def positions_view(self):
        return self._positions

    def tilts_in_view(self):
        return np.zeros_like(self._positions)

    def tilts_out_view(self):
        return np.zeros_like(self._positions)


def test_single_constraint_kkt_projection_zeroes_parallel_component():
    cm = ConstraintModuleManager([])
    cm.modules = {
        "dummy": DummyConstraint(
            {
                0: np.array([1.0, 0.0, 0.0]),
                1: np.array([0.0, 1.0, 0.0]),
            }
        )
    }

    grad = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.0, 1.0, 0.0]),
    }

    cm.apply_gradient_modifications(grad, mesh=None, global_params=None)

    assert np.allclose(grad[0], np.zeros(3))
    assert np.allclose(grad[1], np.zeros(3))


def test_multi_constraint_kkt_projection_removes_all_components():
    cm = ConstraintModuleManager([])
    cm.modules = {
        "dummy": DummyConstraint(
            [
                {0: np.array([1.0, 0.0, 0.0])},
                {1: np.array([0.0, 1.0, 0.0])},
            ]
        )
    }

    grad = {
        0: np.array([2.0, 0.0, 0.0]),
        1: np.array([0.0, -3.0, 0.0]),
    }

    cm.apply_gradient_modifications(grad, mesh=None, global_params=None)

    assert np.allclose(grad[0], np.zeros(3))
    assert np.allclose(grad[1], np.zeros(3))


def test_array_constraint_kkt_projection_removes_components():
    cm = ConstraintModuleManager([])
    g0 = np.zeros((2, 3), dtype=float)
    g1 = np.zeros((2, 3), dtype=float)
    g0[0, 0] = 1.0
    g1[1, 1] = 1.0
    cm.modules = {"dummy": DummyArrayConstraint([g0, g1])}

    grad_arr = np.zeros((2, 3), dtype=float)
    grad_arr[0, 0] = 2.0
    grad_arr[1, 1] = -3.0
    mesh = DummyMesh(2)

    cm.apply_gradient_modifications_array(grad_arr, mesh=mesh, global_params=None)
    assert np.allclose(grad_arr, np.zeros_like(grad_arr))


def test_tilt_array_constraint_kkt_projection_removes_components():
    cm = ConstraintModuleManager([])
    g_in = np.zeros((2, 3), dtype=float)
    g_out = np.zeros((2, 3), dtype=float)
    g_in[0, 0] = 1.0
    g_out[1, 1] = 1.0
    cm.modules = {"dummy": DummyTiltConstraint([(g_in, None), (None, g_out)])}

    tilt_in_grad = np.zeros((2, 3), dtype=float)
    tilt_out_grad = np.zeros((2, 3), dtype=float)
    tilt_in_grad[0, 0] = 4.0
    tilt_out_grad[1, 1] = -5.0
    mesh = DummyMesh(2)

    cm.apply_tilt_gradient_modifications_array(
        tilt_in_grad, tilt_out_grad, mesh=mesh, global_params=None
    )
    assert np.allclose(tilt_in_grad, np.zeros_like(tilt_in_grad))
    assert np.allclose(tilt_out_grad, np.zeros_like(tilt_out_grad))
