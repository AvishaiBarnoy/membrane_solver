import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.io import PropertiesCommand, SaveCommand, VisualizeCommand
from commands.minimization import (
    GoCommand,
    HessianCommand,
    LiveVisCommand,
    SetStepperCommand,
    ShowEdgesCommand,
)
from geometry.entities import Edge, Mesh, Vertex
from runtime.steppers.conjugate_gradient import ConjugateGradient
from runtime.steppers.gradient_descent import GradientDescent


def build_line_mesh():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.build_position_cache()
    return mesh


def test_save_command_writes_to_default_filename(monkeypatch):
    mesh = build_line_mesh()
    ctx = SimpleNamespace(mesh=mesh)
    called = {}

    def fake_save(m, filename):
        called["mesh"] = m
        called["filename"] = filename

    monkeypatch.setattr("commands.io.save_geometry", fake_save)
    SaveCommand().execute(ctx, [])
    assert called["mesh"] is mesh
    assert called["filename"] == "interactive.temp"


def test_visualize_command_calls_plot(monkeypatch):
    mesh = build_line_mesh()
    ctx = SimpleNamespace(mesh=mesh)
    called = {}

    def fake_plot(
        m,
        show_indices=False,
        draw_edges=False,
        transparent=False,
        color_by=None,
        show_tilt_arrows=False,
    ):
        called["mesh"] = m
        called["show_indices"] = show_indices
        called["draw_edges"] = draw_edges
        called["transparent"] = transparent
        called["color_by"] = color_by
        called["show_tilt_arrows"] = show_tilt_arrows

    monkeypatch.setattr("visualization.plotting.plot_geometry", fake_plot)
    VisualizeCommand().execute(ctx, [])
    assert called["mesh"] is mesh
    assert called["show_indices"] is False
    assert called["color_by"] is None
    assert called["show_tilt_arrows"] is False


def test_visualize_command_allows_tilt_coloring(monkeypatch):
    mesh = build_line_mesh()
    minimizer = SimpleNamespace(vis_color_by=None)
    ctx = SimpleNamespace(mesh=mesh, minimizer=minimizer)
    called = {}

    def fake_plot(m, **kwargs):
        called["mesh"] = m
        called.update(kwargs)

    monkeypatch.setattr("visualization.plotting.plot_geometry", fake_plot)
    VisualizeCommand().execute(ctx, ["tilt"])

    assert called["mesh"] is mesh
    assert called.get("color_by") == "tilt_mag"


def test_properties_command_prints_header(capsys):
    mesh = build_line_mesh()
    ctx = SimpleNamespace(mesh=mesh)
    PropertiesCommand().execute(ctx, [])
    out = capsys.readouterr().out
    assert "=== Physical Properties ===" in out
    assert "Vertices:" in out


def test_live_vis_toggles_on_minimizer():
    ctx = SimpleNamespace(minimizer=SimpleNamespace(live_vis=False))
    LiveVisCommand().execute(ctx, [])
    assert ctx.minimizer.live_vis is True
    LiveVisCommand().execute(ctx, [])
    assert ctx.minimizer.live_vis is False


def test_show_edges_command_updates_live_vis(monkeypatch):
    mesh = build_line_mesh()
    minimizer = SimpleNamespace(
        live_vis=True,
        live_vis_state=None,
        live_vis_color_by=None,
        live_vis_show_tilt_arrows=False,
    )
    ctx = SimpleNamespace(mesh=mesh, minimizer=minimizer)
    called = {}

    def fake_update_live_vis(*args, **kwargs):
        called["kwargs"] = kwargs
        return {"fig": None, "ax": None}

    monkeypatch.setattr("visualization.plotting.update_live_vis", fake_update_live_vis)

    ShowEdgesCommand().execute(ctx, ["off"])
    assert ctx.minimizer.live_vis_show_edges is False
    assert called["kwargs"]["show_edges"] is False


def test_set_stepper_command_updates_context_and_minimizer():
    ctx = SimpleNamespace(stepper=GradientDescent(), minimizer=SimpleNamespace())
    ctx.minimizer.stepper = ctx.stepper

    SetStepperCommand("cg").execute(ctx, [])
    assert isinstance(ctx.stepper, ConjugateGradient)
    assert isinstance(ctx.minimizer.stepper, ConjugateGradient)

    SetStepperCommand("gd").execute(ctx, [])
    assert isinstance(ctx.stepper, GradientDescent)
    assert isinstance(ctx.minimizer.stepper, GradientDescent)


def test_hessian_command_runs_without_switching_stepper(monkeypatch):
    mesh = build_line_mesh()

    class DummyMinimizer:
        def __init__(self, m):
            self.mesh = m
            self.step_size = 1e-2
            self._has_enforceable_constraints = False

        def compute_energy_and_gradient(self):
            return 1.0, {0: np.array([1.0, 0.0, 0.0]), 1: np.array([-1.0, 0.0, 0.0])}

        def project_constraints(self, grad):
            return

        def compute_energy(self):
            return 1.0

        def _enforce_constraints(self, _mesh=None):
            return

    ctx = SimpleNamespace(
        mesh=mesh, minimizer=DummyMinimizer(mesh), stepper=GradientDescent()
    )
    ctx.minimizer.stepper = ctx.stepper

    class DummyBFGS:
        def step(self, mesh, grad, step_size, energy_fn, constraint_enforcer=None):
            return True, step_size, float(energy_fn())

    monkeypatch.setattr("commands.minimization.BFGS", DummyBFGS)

    HessianCommand().execute(ctx, [])

    assert isinstance(ctx.stepper, GradientDescent)
    assert isinstance(ctx.minimizer.stepper, GradientDescent)


def test_go_command_runs_minimizer_and_warns_on_collisions(monkeypatch, caplog):
    mesh = build_line_mesh()

    class DummyMinimizer:
        def __init__(self, m):
            self.mesh = m
            self.live_vis = False

        def minimize(self, n_steps=1, callback=None):
            assert n_steps == 3
            assert callback is None
            return {"mesh": self.mesh, "energy": 1.23}

    ctx = SimpleNamespace(
        mesh=mesh, minimizer=DummyMinimizer(mesh), stepper=ConjugateGradient()
    )

    monkeypatch.setattr(
        "commands.minimization.detect_vertex_edge_collisions", lambda m: [(2, 1)]
    )

    with caplog.at_level("WARNING"):
        GoCommand().execute(ctx, ["3"])
    assert "vertex-edge collisions detected" in caplog.text
