import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex
from parameters.global_parameters import GlobalParameters
from runtime.minimizer import Minimizer


class DummyEnergyModule:
    def __init__(self, energy=1.0, grad_value=1.0):
        self._energy = energy
        self._grad_value = grad_value

    def compute_energy_and_gradient(
        self, mesh, global_params, resolver, compute_gradient=True
    ):
        if not compute_gradient:
            return self._energy, {}
        grad = {vid: np.array([self._grad_value, 0.0, 0.0]) for vid in mesh.vertices}
        return self._energy, grad


class DummyEnergyManager:
    def __init__(self, mod):
        self.mod = mod
        self.modules = {"dummy": mod}

    def get_module(self, name):
        return self.mod


class DummyConstraintManager:
    def __init__(self):
        self.calls = []

    def get_constraint(self, name):
        return SimpleNamespace(enforce_constraint=lambda m, **kwargs: None)

    def apply_gradient_modifications(self, grad, mesh, global_params):
        return

    def enforce_all(self, mesh, **kwargs):
        self.calls.append(kwargs.get("context"))


class DummyStepper:
    def __init__(self, results):
        self._results = list(results)
        self.reset_calls = 0

    def step(self, mesh, grad, step_size, energy_fn, constraint_enforcer=None):
        return self._results.pop(0)

    def reset(self):
        self.reset_calls += 1


def build_min_mesh(with_body=False):
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[2] = Vertex(2, np.array([0.0, 1.0, 0.0]))
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)
    mesh.facets[0] = Facet(0, [1, 2, 3])
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    if with_body:
        # Minimal body object with target volume + compute_volume
        body = SimpleNamespace(
            index=0,
            target_volume=1.0,
            options={},
            compute_volume=lambda m: 10.0,
            compute_volume_and_gradient=lambda m: (10.0, {0: np.ones(3)}),
        )
        mesh.bodies[0] = body
    mesh.energy_modules = ["dummy"]
    mesh.constraint_modules = ["volume"]
    return mesh


def test_minimize_n_steps_le_zero_enforces_constraints():
    mesh = build_min_mesh()
    gp = GlobalParameters()
    energy = DummyEnergyModule(energy=1.0, grad_value=0.0)
    cm = DummyConstraintManager()
    stepper = DummyStepper(results=[])
    minim = Minimizer(mesh, gp, stepper, DummyEnergyManager(energy), cm, quiet=True)

    out = minim.minimize(n_steps=0)
    assert out["terminated_early"] is True
    assert cm.calls == ["minimize"]


def test_minimize_converges_when_grad_norm_below_tol():
    mesh = build_min_mesh()
    gp = GlobalParameters()
    energy = DummyEnergyModule(energy=2.0, grad_value=0.0)
    cm = DummyConstraintManager()
    stepper = DummyStepper(results=[(True, 1e-3)])
    minim = Minimizer(
        mesh, gp, stepper, DummyEnergyManager(energy), cm, quiet=True, tol=1e-6
    )

    out = minim.minimize(n_steps=5)
    assert out["terminated_early"] is True
    assert out["iterations"] == 1


def test_minimize_terminates_after_max_zero_steps_without_reset():
    mesh = build_min_mesh()
    gp = GlobalParameters({"max_zero_steps": 1, "step_size_floor": 1e-3})
    energy = DummyEnergyModule(energy=2.0, grad_value=1.0)
    cm = DummyConstraintManager()
    # Fail step with step_size <= floor
    stepper = DummyStepper(results=[(False, 1e-4)])
    minim = Minimizer(mesh, gp, stepper, DummyEnergyManager(energy), cm, quiet=True)

    out = minim.minimize(n_steps=3)
    assert out["terminated_early"] is True
    assert out["step_success"] is False
    # When terminating immediately, we return before calling reset().
    assert stepper.reset_calls == 0


def test_minimize_resets_stepper_on_failed_step_that_does_not_terminate():
    mesh = build_min_mesh()
    gp = GlobalParameters({"max_zero_steps": 10, "step_size_floor": 1e-12})
    energy = DummyEnergyModule(energy=2.0, grad_value=1.0)
    cm = DummyConstraintManager()
    stepper = DummyStepper(results=[(False, 1e-3), (True, 1e-3)])
    minim = Minimizer(mesh, gp, stepper, DummyEnergyManager(energy), cm, quiet=True)

    out = minim.minimize(n_steps=2)
    assert out["terminated_early"] is False
    assert stepper.reset_calls == 1


def test_minimize_volume_drift_triggers_mesh_op_enforcement_and_stepper_reset():
    mesh = build_min_mesh(with_body=True)
    gp = GlobalParameters(
        {
            "volume_constraint_mode": "lagrange",
            "volume_projection_during_minimization": False,
            "volume_tolerance": 1e-6,
        }
    )
    energy = DummyEnergyModule(energy=1.0, grad_value=1.0)
    cm = DummyConstraintManager()
    stepper = DummyStepper(results=[(True, 1e-3)])
    minim = Minimizer(mesh, gp, stepper, DummyEnergyManager(energy), cm, quiet=True)

    minim.minimize(n_steps=1)
    # First: enforce during minimize via line search callback is not used here
    # Second: enforcement due to drift uses mesh_operation context
    assert "mesh_operation" in cm.calls
    assert stepper.reset_calls >= 1
