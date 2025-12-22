import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.constraint_manager import ConstraintModuleManager


class DummyConstraint:
    def __init__(self, gC):
        self._gC = gC

    def constraint_gradient(self, mesh, global_params):
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
