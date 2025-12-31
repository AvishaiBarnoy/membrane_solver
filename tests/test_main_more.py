import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.modules.pop("main", None)

import main as main_module


def write_mesh(path, *, instructions=None):
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "edges": [[0, 1]],
        "instructions": instructions or [],
        "global_parameters": {},
    }
    path.write_text(json.dumps(data))


def test_resolve_json_path_accepts_missing_suffix(tmp_path):
    mesh_path = tmp_path / "mesh.json"
    mesh_path.write_text("{}")
    assert main_module.resolve_json_path(str(mesh_path)[:-5]) == str(mesh_path)


def test_main_executes_instruction_file_and_saves_output(tmp_path, capsys, monkeypatch):
    mesh_path = tmp_path / "mesh.json"
    out_path = tmp_path / "out.json"
    log_path = tmp_path / "run.log"
    inst_path = tmp_path / "inst.txt"

    write_mesh(mesh_path)
    inst_path.write_text("properties\nunknown_command\n")

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
            "--log",
            str(log_path),
            "-q",
        ],
    )
    main_module.main()
    out = capsys.readouterr().out
    assert "=== Physical Properties ===" in out
    assert out_path.exists()


def test_main_prompts_for_input_and_handles_interactive_quit(tmp_path, monkeypatch):
    mesh_path = tmp_path / "mesh.json"
    out_path = tmp_path / "out.json"
    log_path = tmp_path / "run.log"
    write_mesh(mesh_path)

    # No -i: main() should prompt for input path, then enter the REPL.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--log",
            str(log_path),
            "-o",
            str(out_path),
            "-q",
        ],
    )

    answers = iter([str(mesh_path), "q"])

    def fake_input(_prompt=""):
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)

    main_module.main()
    assert out_path.exists()


def test_main_interactive_command_exception_is_logged(tmp_path, monkeypatch, caplog):
    mesh_path = tmp_path / "mesh.json"
    log_path = tmp_path / "run.log"
    write_mesh(mesh_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-i",
            str(mesh_path),
            "--log",
            str(log_path),
        ],
    )

    class Boom:
        def execute(self, context, args):
            raise RuntimeError("boom")

    def fake_get_command(_name):
        return Boom(), []

    # First input line triggers exception, second raises EOF to exit.
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
