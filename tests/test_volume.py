"""Consolidated tests for volume energy, gradients, and constraints."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from modules.constraints import volume as volume_constraint
from modules.energy import volume as volume_module
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.base import BaseStepper

# --- Helpers ---


def build_tetra_mesh(target_volume: float = 0.1) -> Mesh:
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2 = Vertex(2, np.array([0.0, 1.0, 0.0]))
    v3 = Vertex(3, np.array([0.0, 0.0, 1.0]))
    vertices = {0: v0, 1: v1, 2: v2, 3: v3}

    edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 0),
        4: Edge(4, 0, 3),
        5: Edge(5, 1, 3),
        6: Edge(6, 2, 3),
    }

    facets = {
        0: Facet(0, [1, 2, 3]),
        1: Facet(1, [1, 5, -4]),
        2: Facet(2, [3, 6, -4]),
        3: Facet(3, [2, 6, -5]),
    }

    body = Body(
        0,
        [0, 1, 2, 3],
        target_volume=target_volume,
        options={"target_volume": target_volume, "volume_stiffness": 2.0},
    )

    return Mesh(vertices=vertices, edges=edges, facets=facets, bodies={0: body})


class DummyBody:
    def __init__(self, volume, target_volume=0.0, stiffness=None, gradient=None):
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


def make_dummy_mesh(body):
    mesh = SimpleNamespace()
    mesh.bodies = {0: body}
    mesh.facet_vertex_loops = None
    return mesh


# --- Energy Module Tests (Mocked) ---


def test_calculate_volume_energy_penalty_mode():
    global_params = GlobalParameters({"volume_constraint_mode": "penalty"})
    global_params.set("volume_stiffness", 2.0)
    body = DummyBody(volume=2.0, target_volume=1.0)
    mesh = make_dummy_mesh(body)
    energy = volume_module.calculate_volume_energy(mesh, global_params)
    assert pytest.approx(energy) == 1.0


def test_calculate_volume_energy_lagrange_is_zero():
    global_params = GlobalParameters({"volume_constraint_mode": "lagrange"})
    body = DummyBody(volume=2.0, target_volume=1.0)
    mesh = make_dummy_mesh(body)
    energy = volume_module.calculate_volume_energy(mesh, global_params)
    assert energy == 0.0


def test_compute_energy_and_gradient_zero_for_lagrange():
    global_params = GlobalParameters({"volume_constraint_mode": "lagrange"})
    body = DummyBody(volume=2.0, target_volume=0.0)
    mesh = make_dummy_mesh(body)
    resolver = ParameterResolver(global_params)
    energy, grad = volume_module.compute_energy_and_gradient(
        mesh, global_params, resolver, compute_gradient=True
    )
    assert energy == 0.0
    assert grad == {}


def test_compute_energy_and_gradient_penalty_mode():
    global_params = GlobalParameters({"volume_constraint_mode": "penalty"})
    global_params.set("volume_stiffness", 4.0)
    body = DummyBody(volume=1.5, target_volume=1.0)
    mesh = make_dummy_mesh(body)
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
    assert pytest.approx(energy) == 3.0
    assert set(grad) == {0, 1}
    np.testing.assert_allclose(grad[0], np.array([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(grad[1], np.array([0.0, 4.0, 0.0]))


# --- Energy Module Integration Tests (Tetrahedron) ---


def test_volume_energy_and_gradient_on_tetrahedron():
    mesh = build_tetra_mesh(target_volume=0.1)
    global_params = GlobalParameters({"volume_constraint_mode": "penalty"})
    global_params.volume_stiffness = 2.0
    param_resolver = ParameterResolver(global_params)

    E, grad = volume_module.compute_energy_and_gradient(
        mesh, global_params, param_resolver
    )

    # Analytical volume of unit tetrahedron: 1/6
    expected_volume = 1.0 / 6.0
    k = 2.0
    V0 = 0.1
    expected_energy = 0.5 * k * (expected_volume - V0) ** 2

    assert np.isclose(E, expected_energy)
    assert set(grad.keys()) == {0, 1, 2, 3}
    for g in grad.values():
        assert g.shape == (3,)
        assert np.all(np.isfinite(g))


def test_calculate_volume_energy_standalone():
    mesh = build_tetra_mesh(target_volume=0.1)
    global_params = GlobalParameters({"volume_constraint_mode": "penalty"})
    global_params.volume_stiffness = 2.0
    energy = volume_module.calculate_volume_energy(mesh, global_params)
    expected_volume = 1.0 / 6.0
    k = 2.0
    V0 = 0.1
    expected_energy = 0.5 * k * (expected_volume - V0) ** 2
    assert np.isclose(energy, expected_energy)


def test_body_compute_volume_gradient_directly():
    mesh = build_tetra_mesh(target_volume=0.1)
    body = mesh.bodies[0]
    grad_vol = body.compute_volume_gradient(mesh)
    assert isinstance(grad_vol, dict)
    assert set(grad_vol.keys()) == {0, 1, 2, 3}
    for g in grad_vol.values():
        assert g.shape == (3,)
        assert np.all(np.isfinite(g))


# --- Constraint Enforcement Tests ---


def test_enforce_volume_constraint_on_tetrahedron():
    target = 0.05
    mesh = build_tetra_mesh(target_volume=target)
    # Perturb one vertex
    mesh.vertices[3].position += np.array([0.2, 0.0, 0.0])

    initial_volume = mesh.compute_total_volume()
    assert not np.isclose(initial_volume, target)

    volume_constraint.enforce_constraint(mesh, tol=1e-10, max_iter=20)

    final_volume = mesh.compute_total_volume()
    assert pytest.approx(target, abs=1e-9) == final_volume


# --- Minimizer Integration Tests ---


class DummyStepper(BaseStepper):
    def step(self, mesh, grad, step_size, energy_fn, constraint_enforcer=None):
        if constraint_enforcer:
            constraint_enforcer(mesh)
        return True, step_size, float(energy_fn())


def test_minimizer_calls_volume_constraint_enforcement(monkeypatch):
    mesh = build_tetra_mesh(target_volume=0.05)
    mesh.energy_modules = []
    mesh.constraint_modules = ["volume"]

    energy_manager = EnergyModuleManager(mesh.energy_modules)
    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)

    minimizer = Minimizer(
        mesh=mesh,
        global_params=mesh.global_parameters,
        stepper=DummyStepper(),
        energy_manager=energy_manager,
        constraint_manager=constraint_manager,
        quiet=True,
    )

    call_count = {"n": 0}

    def fake_enforce(*args, **kwargs):
        call_count["n"] += 1

    minimizer.constraint_manager.enforce_all = fake_enforce

    def fake_compute(self):
        grad = {vidx: np.ones(3) for vidx in self.mesh.vertices}
        return 1.0, grad

    monkeypatch.setattr(Minimizer, "compute_energy_and_gradient", fake_compute)

    minimizer.minimize(n_steps=0)
    assert call_count["n"] == 1
