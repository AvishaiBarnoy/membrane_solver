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
