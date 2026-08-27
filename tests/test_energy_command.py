from commands.executor import execute_command_line
from tests.minimizer_test_utils import build_command_context as _build_context


def test_energy_command_prints_breakdown(capsys):
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "energy_modules": ["surface"],
        "global_parameters": {"surface_tension": 1.0},
        "instructions": [],
    }
    ctx = _build_context(data)

    execute_command_line(ctx, "energy")
    out = capsys.readouterr().out
    assert "Current Total Energy" in out
    assert "surface" in out


def test_energy_command_prints_curvature_stats(capsys):
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "energy_modules": ["surface"],
        "global_parameters": {"surface_tension": 1.0},
        "instructions": [],
    }
    ctx = _build_context(data)

    execute_command_line(ctx, "energy stats")
    out = capsys.readouterr().out
    assert "Curvature diagnostics" in out
