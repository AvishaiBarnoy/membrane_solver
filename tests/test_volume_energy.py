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
    def __init__(
        self,
        volume,
        target_volume=0.0,
        stiffness=None,
        gradient=None,
    ):
        self._volume = volume
        self.target_volume = target_volume
        self.options = {}
        if stiffness is not None:
            self.options["volume_stiffness"] = stiffness
        self._gradient = gradient or {0: np.array([1.0, 0.0, 0.0])}

    def compute_volume(self, mesh, positions=None, index_map=None):
        return self._volume

    def compute_volume_and_gradient(self, mesh, positions=None, index_map=None):
        return self._volume, self._gradient


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


def test_volume_energy_respects_per_body_override_and_global_default():
    global_params = GlobalParameters({"volume_constraint_mode": "penalty"})
    global_params.set("volume_stiffness", 4.0)

    body0 = DummyBody(
        volume=2.0,
        target_volume=1.0,
        stiffness=2.0,
        gradient={0: np.array([1.0, 0.0, 0.0])},
    )
    body1 = DummyBody(
        volume=3.0,
        target_volume=2.0,
        stiffness=None,
        gradient={1: np.array([0.0, 1.0, 0.0])},
    )
    mesh = SimpleNamespace(bodies={0: body0, 1: body1}, facet_vertex_loops=None)
    resolver = ParameterResolver(global_params)

    energy, grad = volume_module.compute_energy_and_gradient(
        mesh, global_params, resolver, compute_gradient=True
    )

    # body0 uses local k=2, delta=1 -> 0.5*2*(1)^2 = 1
    # body1 uses global k=4, delta=1 -> 0.5*4*(1)^2 = 2
    assert pytest.approx(energy) == 3.0
    assert set(grad) == {0, 1}
    np.testing.assert_allclose(grad[0], np.array([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(grad[1], np.array([0.0, 4.0, 0.0]))
