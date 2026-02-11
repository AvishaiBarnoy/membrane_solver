import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_importing_commands_io_does_not_import_matplotlib():
    root = Path(__file__).resolve().parents[1]
    code = "import sys, commands.io; print('matplotlib' in sys.modules)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.splitlines()[-1].strip() == "False"


def test_update_live_vis_reuses_collections_for_polygon_facets(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    config_dir = tmp_path / "mplconfig"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MPLCONFIGDIR", str(config_dir))
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    from geometry.entities import Edge, Facet, Mesh, Vertex
    from visualization.plotting import update_live_vis

    mesh = Mesh()
    mesh.vertices = {
        1: Vertex(1, np.array([0.0, 0.0, 0.0], dtype=float)),
        2: Vertex(2, np.array([1.0, 0.0, 0.0], dtype=float)),
        3: Vertex(3, np.array([1.0, 1.0, 0.0], dtype=float)),
        4: Vertex(4, np.array([0.0, 1.0, 0.0], dtype=float)),
    }
    mesh.edges = {
        1: Edge(1, 1, 2),
        2: Edge(2, 2, 3),
        3: Edge(3, 3, 4),
        4: Edge(4, 4, 1),
    }
    mesh.facets = {
        1: Facet(1, [1, 2, 3, 4]),
    }

    was_interactive = plt.isinteractive()
    try:
        state = update_live_vis(mesh, title="t0")
        tri_col = state["tri_collection"]

        mesh.vertices[2].position = mesh.vertices[2].position + np.array(
            [0.1, 0.0, 0.0], dtype=float
        )
        mesh.increment_version()

        state = update_live_vis(mesh, state=state, title="t1")
        assert state["tri_collection"] is tri_col
    finally:
        if "state" in locals() and state and "fig" in state:
            plt.close(state["fig"])
        if not was_interactive:
            plt.ioff()


def test_plot_geometry_shading_toggle(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    config_dir = tmp_path / "mplconfig"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MPLCONFIGDIR", str(config_dir))
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

    from geometry.entities import Edge, Facet, Mesh, Vertex
    from visualization.plotting import plot_geometry

    mesh = Mesh()
    mesh.vertices = {
        1: Vertex(1, np.array([0.0, 0.0, 0.0], dtype=float)),
        2: Vertex(2, np.array([1.0, 0.0, 0.0], dtype=float)),
        3: Vertex(3, np.array([0.0, 1.0, 0.0], dtype=float)),
    }
    mesh.edges = {
        1: Edge(1, 1, 2),
        2: Edge(2, 2, 3),
        3: Edge(3, 3, 1),
    }
    mesh.facets = {1: Facet(1, [1, 2, 3])}

    was_interactive = plt.isinteractive()
    try:
        plot_geometry(mesh, show=False, draw_edges=False)
        plot_geometry(mesh, show=False, draw_edges=True)
    finally:
        if not was_interactive:
            plt.ioff()
