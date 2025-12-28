import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualization import cli as vis_cli


def test_visualization_cli_requires_supported_suffix(tmp_path):
    bad = tmp_path / "mesh.txt"
    bad.write_text("{}")
    with pytest.raises(ValueError, match="must be a JSON or YAML"):
        vis_cli.main([str(bad)])


def test_visualization_cli_missing_file(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        vis_cli.main([str(missing)])


def test_visualization_cli_saves_without_show(tmp_path, monkeypatch):
    mesh_path = tmp_path / "mesh.yaml"
    out_path = tmp_path / "out.png"
    mesh_path.write_text("vertices: [[0,0,0],[1,0,0]]\nedges: [[0,1]]\n")

    called = {}

    def fake_plot(mesh, **kwargs):
        called["show"] = kwargs.get("show")

    monkeypatch.setattr(vis_cli, "plot_geometry", fake_plot)
    monkeypatch.setattr(vis_cli, "parse_geometry", lambda data: object())

    vis_cli.main([str(mesh_path), "--save", str(out_path), "--no-axes"])
    assert called["show"] is False
