import json
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main as main_module


def write_line_mesh(path):
    data = {
        "vertices": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        "edges": [
            [0, 1],
        ],
        "global_parameters": {},
        "instructions": [],
    }
    path.write_text(json.dumps(data))


def test_main_properties_mode_runs(tmp_path, capsys, monkeypatch):
    mesh_path = tmp_path / "mesh.json"
    log_path = tmp_path / "run.log"
    write_line_mesh(mesh_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-i",
            str(mesh_path),
            "--properties",
            "--non-interactive",
            "--log",
            str(log_path),
            "-q",
        ],
    )
    main_module.main()
    out = capsys.readouterr().out
    assert "=== Physical Properties ===" in out


def test_main_cli_line_tension_unknown_edges_warns(tmp_path, caplog, monkeypatch):
    mesh_path = tmp_path / "mesh.json"
    log_path = tmp_path / "run.log"
    write_line_mesh(mesh_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-i",
            str(mesh_path),
            "--properties",
            "--non-interactive",
            "--log",
            str(log_path),
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


def test_main_radius_of_gyration_mode_runs(tmp_path, capsys, monkeypatch):
    mesh_path = tmp_path / "mesh.json"
    log_path = tmp_path / "run.log"
    write_line_mesh(mesh_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-i",
            str(mesh_path),
            "--radius-of-gyration",
            "--non-interactive",
            "--log",
            str(log_path),
            "-q",
        ],
    )
    main_module.main()
    out = capsys.readouterr().out
    assert "Surface radius of gyration" in out


def test_main_cli_line_tension_edges_invalid_value_exits(tmp_path, monkeypatch):
    mesh_path = tmp_path / "mesh.json"
    log_path = tmp_path / "run.log"
    write_line_mesh(mesh_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-i",
            str(mesh_path),
            "--properties",
            "--non-interactive",
            "--log",
            str(log_path),
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
