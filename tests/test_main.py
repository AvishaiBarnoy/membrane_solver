"""Consolidated tests for the main CLI entry point."""

import json
import os
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Force a re-import of the main module to reflect CLI argument changes
sys.modules.pop("main", None)
import main as main_module


def write_mesh(path, *, instructions=None, **kwargs):
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "edges": [[0, 1]],
        "instructions": instructions or [],
        "global_parameters": kwargs,
    }
    path.write_text(json.dumps(data))


def test_resolve_json_path_accepts_missing_suffix(tmp_path):
    mesh_path = tmp_path / "mesh.json"
    mesh_path.write_text("{}")
    assert main_module.resolve_json_path(str(mesh_path)[:-5]) == str(mesh_path)


def test_main_properties_mode_runs(tmp_path, capsys, monkeypatch):
    mesh_path = tmp_path / "mesh.json"
    write_mesh(mesh_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "-i", str(mesh_path), "--properties", "--non-interactive", "-q"],
    )
    main_module.main()
    assert "=== Physical Properties ===" in capsys.readouterr().out


def test_main_radius_of_gyration_mode_runs(tmp_path, capsys, monkeypatch):
    mesh_path = tmp_path / "mesh.json"
    write_mesh(mesh_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-i",
            str(mesh_path),
            "--radius-of-gyration",
            "--non-interactive",
            "-q",
        ],
    )
    main_module.main()
    assert "Surface radius of gyration" in capsys.readouterr().out


def test_main_executes_instruction_file_and_saves(tmp_path, capsys, monkeypatch):
    mesh_path = tmp_path / "mesh.json"
    out_path = tmp_path / "out.json"
    inst_path = tmp_path / "inst.txt"
    write_mesh(mesh_path)
    inst_path.write_text("properties\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-i",
            str(mesh_path),
            "--instructions",
            str(inst_path),
            "--non-interactive",
            "-o",
            str(out_path),
            "-q",
        ],
    )
    main_module.main()
    assert "=== Physical Properties ===" in capsys.readouterr().out
    assert out_path.exists()


def test_main_interactive_quit(tmp_path, monkeypatch):
    mesh_path = tmp_path / "mesh.json"
    write_mesh(mesh_path)
    monkeypatch.setattr(sys, "argv", ["main.py", "-i", str(mesh_path)])
    monkeypatch.setattr("builtins.input", lambda _="": "q")
    main_module.main()  # Should exit gracefully


def test_main_stepper_override_via_command(monkeypatch):
    from types import SimpleNamespace

    from commands.registry import get_command
    from core.parameters.global_parameters import GlobalParameters
    from runtime.steppers.bfgs import BFGS
    from runtime.steppers.conjugate_gradient import ConjugateGradient
    from runtime.steppers.gradient_descent import GradientDescent

    monkeypatch.setattr(sys, "argv", ["main.py", "-i", "dummy.json"])
    # Can't fully test without a mesh, but we can check the command logic
    ctx = SimpleNamespace(
        stepper=GradientDescent(),
        minimizer=SimpleNamespace(stepper=GradientDescent()),
        mesh=SimpleNamespace(global_parameters=GlobalParameters()),
    )

    # Default is GD
    assert isinstance(ctx.stepper, GradientDescent)

    # Switch to CG
    cmd_cg, _ = get_command("cg")
    cmd_cg.execute(ctx, [])
    assert isinstance(ctx.stepper, ConjugateGradient)
    assert isinstance(ctx.minimizer.stepper, ConjugateGradient)

    # Switch to BFGS
    cmd_bfgs, _ = get_command("bfgs")
    cmd_bfgs.execute(ctx, [])
    assert isinstance(ctx.stepper, BFGS)
    assert isinstance(ctx.minimizer.stepper, BFGS)


def test_main_viz_save_calls_plot_geometry(tmp_path, monkeypatch):
    from visualization import plotting as plotting_module

    mesh_path = tmp_path / "mesh.json"
    write_mesh(mesh_path)
    out_path = tmp_path / "viz.png"

    called = {}

    def fake_plot(mesh, **kwargs):
        called["show"] = kwargs.get("show")
        called["draw_edges"] = kwargs.get("draw_edges")

    monkeypatch.setattr(plotting_module, "plot_geometry", fake_plot)

    class DummyFig:
        def __init__(self):
            self.saved = None

        def savefig(self, path, **_kwargs):
            self.saved = path

    import matplotlib.pyplot as plt

    dummy_fig = DummyFig()
    monkeypatch.setattr(plt, "gcf", lambda: dummy_fig)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-i",
            str(mesh_path),
            "--viz-save",
            str(out_path),
            "--non-interactive",
            "-q",
        ],
    )
    main_module.main()

    assert called["show"] is False
    assert called["draw_edges"] is True
    assert dummy_fig.saved == str(out_path)


def test_main_cli_line_tension_unknown_edges_warns(tmp_path, caplog, monkeypatch):
    mesh_path = tmp_path / "mesh.json"
    write_mesh(mesh_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-i",
            str(mesh_path),
            "--properties",
            "--non-interactive",
            "-q",
            "--line-tension",
            "2.0",
            "--line-tension-edges",
            "999",
        ],
    )
    with caplog.at_level("WARNING"):
        main_module.main()
    assert "Ignoring unknown edge IDs for line tension" in caplog.text


def test_main_cli_line_tension_edges_invalid_value_exits(tmp_path, monkeypatch):
    mesh_path = tmp_path / "mesh.json"
    write_mesh(mesh_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-i",
            str(mesh_path),
            "--properties",
            "--non-interactive",
            "-q",
            "--line-tension",
            "2.0",
            "--line-tension-edges",
            "a,b",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        main_module.main()
    assert excinfo.value.code == 1


def test_main_interactive_command_exception_is_logged(tmp_path, monkeypatch, caplog):
    mesh_path = tmp_path / "mesh.json"
    write_mesh(mesh_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-i",
            str(mesh_path),
        ],
    )

    class Boom:
        def execute(self, context, args):
            raise RuntimeError("boom")

    def fake_get_command(_name):
        return Boom(), []

    calls = {"n": 0}

    def fake_input(_prompt=""):
        calls["n"] += 1
        if calls["n"] == 1:
            return "boom"
        raise EOFError()

    monkeypatch.setattr(main_module, "get_command", fake_get_command)
    monkeypatch.setattr("builtins.input", fake_input)

    with caplog.at_level("ERROR"):
        main_module.main()
    assert "Error executing command" in caplog.text
