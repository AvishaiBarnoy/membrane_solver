"""Consolidated tests for the Minimizer class."""

import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.constraint_manager import ConstraintModuleManager
from runtime.minimizer import Minimizer

# --- Mocks and Helpers ---


class DummyEnergyModule:
    def __init__(self, energy=1.0, grad_value=1.0):
        self._energy = energy
        self._grad_value = grad_value

    def compute_energy_and_gradient_array(
        self, mesh, global_params, resolver, positions, index_map, grad_arr
    ):
        for vidx, row in index_map.items():
            grad_arr[row] += np.array([self._grad_value, 0.0, 0.0])
        return self._energy

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
    mesh.build_position_cache()
    if with_body:
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


# --- Tests ---


def test_minimizer_dispatches_to_array_path():
    """Verify that Minimizer calls compute_energy_and_gradient_array if available."""
    mesh = build_min_mesh()
    mock_mod = MagicMock()
    mock_mod.compute_energy_and_gradient_array.return_value = 10.0

    gp = GlobalParameters()
    em = MagicMock()
    em.get_module.return_value = mock_mod
    cm = ConstraintModuleManager([])

    minim = Minimizer(mesh, gp, MagicMock(), em, cm, energy_modules=["mock"])
    minim.compute_energy_and_gradient()

    assert mock_mod.compute_energy_and_gradient_array.called
    assert not mock_mod.compute_energy_and_gradient.called


def test_minimizer_falls_back_to_dict_path():
    """Verify that Minimizer rejects legacy modules without array support."""
    mesh = build_min_mesh()
    mock_mod = MagicMock()
    del mock_mod.compute_energy_and_gradient_array
    mock_mod.compute_energy_and_gradient.return_value = (5.0, {0: np.array([1, 1, 1])})

    gp = GlobalParameters()
    em = MagicMock()
    em.get_module.return_value = mock_mod
    cm = ConstraintModuleManager([])

    try:
        Minimizer(mesh, gp, MagicMock(), em, cm, energy_modules=["mock"])
    except TypeError as exc:
        assert "compute_energy_and_gradient_array" in str(exc)
    else:
        raise AssertionError("Expected TypeError for dict-only energy module")


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
    stepper = DummyStepper(results=[(True, 1e-3, 2.0)])
    minim = Minimizer(
        mesh, gp, stepper, DummyEnergyManager(energy), cm, quiet=True, tol=1e-6
    )

    out = minim.minimize(n_steps=5)
    assert out["terminated_early"] is True
    assert out["iterations"] == 1


def test_minimize_terminates_after_max_zero_steps():
    mesh = build_min_mesh()
    gp = GlobalParameters({"max_zero_steps": 1, "step_size_floor": 1e-3})
    energy = DummyEnergyModule(energy=2.0, grad_value=1.0)
    cm = DummyConstraintManager()
    stepper = DummyStepper(results=[(False, 1e-4, 2.0)])
    minim = Minimizer(mesh, gp, stepper, DummyEnergyManager(energy), cm, quiet=True)

    out = minim.minimize(n_steps=3)
    assert out["terminated_early"] is True
    assert out["step_success"] is False
    assert stepper.reset_calls == 0


def test_minimize_resets_stepper_on_failed_step():
    mesh = build_min_mesh()
    gp = GlobalParameters({"max_zero_steps": 10, "step_size_floor": 1e-12})
    energy = DummyEnergyModule(energy=2.0, grad_value=1.0)
    cm = DummyConstraintManager()
    stepper = DummyStepper(results=[(False, 1e-3, 2.0), (True, 1e-3, 2.0)])
    minim = Minimizer(mesh, gp, stepper, DummyEnergyManager(energy), cm, quiet=True)

    minim.minimize(n_steps=2)
    assert stepper.reset_calls == 1


def test_minimize_volume_drift_triggers_enforcement():
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
    stepper = DummyStepper(results=[(True, 1e-3, 1.0)])
    minim = Minimizer(mesh, gp, stepper, DummyEnergyManager(energy), cm, quiet=True)

    minim.minimize(n_steps=1)
    assert "mesh_operation" in cm.calls
    assert stepper.reset_calls >= 1


def test_minimize_energy_consistency_logs_debug(caplog, monkeypatch):
    mesh = build_min_mesh()
    gp = GlobalParameters()
    minim = Minimizer(
        mesh,
        gp,
        DummyStepper([]),
        DummyEnergyManager(DummyEnergyModule()),
        DummyConstraintManager(),
        quiet=True,
    )
    monkeypatch.setattr(minim, "compute_energy", lambda: 2.0)
    monkeypatch.setattr(
        minim, "compute_energy_and_gradient_array", lambda: (2.0, np.zeros((3, 3)))
    )

    with caplog.at_level(logging.DEBUG, logger="membrane_solver"):
        minim.minimize(n_steps=0)

    assert "Energy consistency" in caplog.text
    assert "Energy consistency mismatch" not in caplog.text


def test_minimize_energy_consistency_mismatch_logs_debug(caplog, monkeypatch):
    mesh = build_min_mesh()
    gp = GlobalParameters()
    minim = Minimizer(
        mesh,
        gp,
        DummyStepper([]),
        DummyEnergyManager(DummyEnergyModule()),
        DummyConstraintManager(),
        quiet=True,
    )
    monkeypatch.setattr(minim, "compute_energy", lambda: 1.5)
    monkeypatch.setattr(
        minim, "compute_energy_and_gradient_array", lambda: (2.5, np.zeros((3, 3)))
    )
    monkeypatch.setattr(
        minim, "compute_energy_breakdown", lambda: {"mod_a": 3.0, "mod_b": -1.0}
    )

    with caplog.at_level(logging.DEBUG, logger="membrane_solver"):
        minim.minimize(n_steps=0)

    assert "Energy consistency mismatch" in caplog.text


def test_minimize_energy_consistency_logs_are_debug_only(caplog, monkeypatch):
    mesh = build_min_mesh()
    gp = GlobalParameters()
    minim = Minimizer(
        mesh,
        gp,
        DummyStepper([]),
        DummyEnergyManager(DummyEnergyModule()),
        DummyConstraintManager(),
        quiet=True,
    )
    monkeypatch.setattr(minim, "compute_energy", lambda: 2.0)
    monkeypatch.setattr(
        minim, "compute_energy_and_gradient_array", lambda: (2.0, np.zeros((3, 3)))
    )

    with caplog.at_level(logging.INFO, logger="membrane_solver"):
        minim.minimize(n_steps=0)

    assert "Energy consistency" not in caplog.text
