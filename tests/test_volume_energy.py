import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.energy import volume as volume_module
from parameters.global_parameters import GlobalParameters
from parameters.resolver import ParameterResolver


class DummyBody:
    def __init__(self, volume, target_volume=0.0, stiffness=None):
        self._volume = volume
        self.target_volume = target_volume
        self.options = {}
        if stiffness is not None:
            self.options["volume_stiffness"] = stiffness

    def compute_volume(self, mesh, positions=None, index_map=None):
        return self._volume

    def compute_volume_and_gradient(self, mesh, positions=None, index_map=None):
        gradient = {0: np.array([1.0, 0.0, 0.0])}
        return self._volume, gradient


def make_mesh(body):
    mesh = SimpleNamespace()
    mesh.bodies = {0: body}
    mesh.facet_vertex_loops = None
    return mesh


def test_calculate_volume_energy_penalty_mode():
    global_params = GlobalParameters({"volume_constraint_mode": "penalty"})
    global_params.set("volume_stiffness", 2.0)
    global_params.volume_stiffness = 2.0
    body = DummyBody(volume=2.0, target_volume=1.0)
    mesh = make_mesh(body)
    energy = volume_module.calculate_volume_energy(mesh, global_params)
    assert pytest.approx(energy) == 1.0


def test_calculate_volume_energy_lagrange_is_zero():
    global_params = GlobalParameters({"volume_constraint_mode": "lagrange"})
    body = DummyBody(volume=2.0, target_volume=1.0)
    mesh = make_mesh(body)

    energy = volume_module.calculate_volume_energy(mesh, global_params)
    assert energy == 0.0


def test_compute_energy_and_gradient_zero_for_lagrange():
    global_params = GlobalParameters({"volume_constraint_mode": "lagrange"})
    body = DummyBody(volume=2.0, target_volume=0.0)
    mesh = make_mesh(body)
    resolver = ParameterResolver(global_params)

    energy, grad = volume_module.compute_energy_and_gradient(
        mesh, global_params, resolver, compute_gradient=True
    )
    assert energy == 0.0
    assert grad == {}


def test_compute_energy_and_gradient_penalty_mode():
    global_params = GlobalParameters({"volume_constraint_mode": "penalty"})
    global_params.set("volume_stiffness", 4.0)
    global_params.volume_stiffness = 4.0
    body = DummyBody(volume=1.5, target_volume=1.0)
    mesh = make_mesh(body)
    resolver = ParameterResolver(global_params)

    energy, grad = volume_module.compute_energy_and_gradient(
        mesh, global_params, resolver, compute_gradient=True
    )
    assert pytest.approx(energy) == 0.5  # 0.5 * 4 * (0.5)^2
    assert 0 in grad
    np.testing.assert_allclose(grad[0], np.array([2.0, 0.0, 0.0]))
