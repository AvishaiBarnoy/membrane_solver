import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Mesh, Vertex, Edge, Facet, Body
from modules.constraints import volume as volume_constraint
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.base import BaseStepper


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
        options={"target_volume": target_volume},
    )

    return Mesh(vertices=vertices, edges=edges, facets=facets, bodies={0: body})


def test_volume_constraint_enforces_target():
    mesh = build_tetra_mesh(target_volume=0.05)
    mesh.vertices[3].position += np.array([0.2, 0.0, 0.0])

    initial_volume = mesh.compute_total_volume()
    assert not np.isclose(initial_volume, mesh.bodies[0].target_volume)

    volume_constraint.enforce_constraint(mesh, tol=1e-10, max_iter=10)

    final_volume = mesh.compute_total_volume()
    assert pytest.approx(mesh.bodies[0].target_volume, rel=0, abs=1e-9) == final_volume


class DummyStepper(BaseStepper):
    def step(self, mesh, grad, step_size, energy_fn, constraint_enforcer=None):
        if constraint_enforcer:
            constraint_enforcer(mesh)
        return True, step_size


def test_minimizer_calls_constraint_manager(monkeypatch):
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
