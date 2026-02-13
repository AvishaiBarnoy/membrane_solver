import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer


class DummyStepper:
    def __init__(self, *, accepted_energy: float) -> None:
        self.accepted_energy = float(accepted_energy)

    def step(self, mesh, grad, step_size, energy_fn, constraint_enforcer=None):
        _ = (mesh, grad, energy_fn, constraint_enforcer)
        return True, float(step_size), float(self.accepted_energy)


def _build_triangle_mesh() -> Mesh:
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([0.0, 1.0, 0.0])),
    }
    mesh.edges = {1: Edge(1, 0, 1), 2: Edge(2, 1, 2), 3: Edge(3, 2, 0)}
    mesh.facets = {0: Facet(0, edge_indices=[1, 2, 3])}
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    mesh.energy_modules = []
    mesh.constraint_modules = []
    return mesh


def test_accepted_energy_mismatch_logs_only_in_debug(caplog, monkeypatch) -> None:
    mesh = _build_triangle_mesh()
    gp = GlobalParameters({"step_size_mode": "fixed", "step_size": 0.1})

    minim = Minimizer(
        mesh,
        gp,
        DummyStepper(accepted_energy=1.0),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )

    # Force the main loop to execute a step.
    monkeypatch.setattr(
        minim,
        "compute_energy_and_gradient_array",
        lambda: (0.0, np.ones((len(mesh.vertex_ids), 3))),
    )
    # Make the post-step energy differ from accepted_energy.
    monkeypatch.setattr(minim, "compute_energy", lambda: 2.0)
    monkeypatch.setattr(minim, "compute_energy_breakdown", lambda: {"dummy": 2.0})

    with caplog.at_level(logging.INFO, logger="membrane_solver"):
        minim.minimize(n_steps=1)
    assert "Accepted energy mismatch" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="membrane_solver"):
        minim.minimize(n_steps=1)
    assert "Accepted energy mismatch" in caplog.text


def test_step_log_reports_recomputed_full_energy(capsys, monkeypatch) -> None:
    mesh = _build_triangle_mesh()
    gp = GlobalParameters({"step_size_mode": "fixed", "step_size": 0.1})

    minim = Minimizer(
        mesh,
        gp,
        DummyStepper(accepted_energy=1.0),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=False,
    )

    monkeypatch.setattr(
        minim,
        "compute_energy_and_gradient_array",
        lambda: (0.0, np.ones((len(mesh.vertex_ids), 3))),
    )
    monkeypatch.setattr(minim, "compute_energy", lambda: 2.0)
    monkeypatch.setattr(minim, "compute_energy_breakdown", lambda: {"dummy": 2.0})

    minim.minimize(n_steps=1)
    out = capsys.readouterr().out
    assert "Step" in out
    assert "Energy = 2.00000" in out
    assert "Energy = 1.00000" not in out
