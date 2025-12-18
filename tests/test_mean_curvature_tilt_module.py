import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules import mean_curvature_tilt


class DummyFacet:
    def __init__(self, vertex_indices):
        self.vertex_indices = list(vertex_indices)

    def compute_mean_curvature(self):
        return 2.0

    def compute_divergence_of_tilt(self):
        return 0.5

    def area(self):
        return 3.0

    def dJ_dvertex(self, vidx):
        return (
            np.array([1.0, 0.0, 0.0]) if vidx == self.vertex_indices[0] else np.zeros(3)
        )

    def dDivT_dvertex(self, vidx):
        return (
            np.array([0.0, 1.0, 0.0]) if vidx == self.vertex_indices[1] else np.zeros(3)
        )

    def dDivT_dtilt(self, vidx):
        return np.array([1.0, -1.0]) if vidx == self.vertex_indices[2] else np.zeros(2)


class DummyResolver:
    def get(self, obj, name):
        if name == "spontaneous_curvature":
            return 1.0
        if name == "bending_rigidity":
            return 2.0
        raise KeyError(name)


def test_mean_curvature_tilt_energy_and_grads_shapes():
    mesh = type("Mesh", (), {})()
    mesh.vertices = {0: object(), 1: object(), 2: object()}
    mesh.facets = {0: DummyFacet([0, 1, 2])}

    E, shape_grad, tilt_grad = mean_curvature_tilt.compute_energy_and_gradient(
        mesh, gp=None, resolver=DummyResolver()
    )

    # delta = J - J0 + div_t = 2 - 1 + 0.5 = 1.5
    # E = 0.5 * kappa * delta^2 * A = 0.5 * 2 * 2.25 * 3 = 6.75
    assert np.isclose(E, 6.75)
    assert set(shape_grad.keys()) == {0, 1, 2}
    assert set(tilt_grad.keys()) == {0, 1, 2}
    for g in shape_grad.values():
        assert g.shape == (3,)
    for g in tilt_grad.values():
        assert g.shape == (2,)
