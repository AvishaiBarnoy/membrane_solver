import json
import os
import sys

import matplotlib

matplotlib.use("Agg")


def test_main_viz_save_calls_plot_geometry(tmp_path, monkeypatch):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    import main as main_module
    from visualization import plotting as plotting_module

    mesh_path = tmp_path / "mesh.json"
    mesh_path.write_text(
        json.dumps(
            {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "edges": [[0, 1]],
                "global_parameters": {},
                "instructions": [],
            }
        )
    )
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
