import pytest

from commands.executor import execute_command_line
from tests.minimizer_test_utils import build_command_context as _build_context


def test_t_prefix_sets_step_size():
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "energy_modules": ["surface"],
        "global_parameters": {"surface_tension": 1.0, "step_size": 1e-2},
        "instructions": [],
    }
    ctx = _build_context(data)
    assert ctx.mesh.global_parameters.get("step_size") == pytest.approx(1e-2)

    execute_command_line(ctx, "t1e-3")
    assert ctx.mesh.global_parameters.get("step_size") == pytest.approx(1e-3)
    assert ctx.mesh.global_parameters.get("step_size_mode") == "fixed"
    assert ctx.minimizer.step_size == pytest.approx(1e-3)


def test_tf_reenables_adaptive_step_sizing():
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "energy_modules": ["surface"],
        "global_parameters": {"surface_tension": 1.0, "step_size": 1e-2},
        "instructions": [],
    }
    ctx = _build_context(data)
    execute_command_line(ctx, "t1e-3")
    assert ctx.mesh.global_parameters.get("step_size_mode") == "fixed"

    execute_command_line(ctx, "tf")
    assert ctx.mesh.global_parameters.get("step_size_mode") == "adaptive"
