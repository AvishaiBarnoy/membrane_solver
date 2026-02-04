import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Mesh, Vertex
from runtime.steppers.bfgs import BFGS


def test_bfgs_stepper_moves_toward_minimum():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))

    def energy_fn():
        x = mesh.vertices[0].position[0]
        return (x - 1.0) ** 2

    stepper = BFGS()
    grad = {0: np.array([-2.0, 0.0, 0.0])}

    success, _, _accepted_energy = stepper.step(mesh, grad, 0.1, energy_fn)
    assert success
    assert mesh.vertices[0].position[0] > 0.0

    grad2 = {0: np.array([2.0 * (mesh.vertices[0].position[0] - 1.0), 0.0, 0.0])}
    success2, _, _accepted_energy2 = stepper.step(mesh, grad2, 0.1, energy_fn)
    assert success2
