import pytest

from commands.executor import execute_command_line
from geometry.geom_io import parse_geometry
from runtime.minimizer import Minimizer
from tests.minimizer_test_utils import build_command_context as _build_context


def test_macro_invoked_by_name_executes_sequence():
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "macros": {"gogo": "g 1; g 2; g3"},
        "instructions": [],
    }
    ctx = _build_context(data)

    calls: list[int] = []

    def fake_minimize(self, n_steps=1, callback=None):
        calls.append(int(n_steps))
        return {"mesh": self.mesh, "energy": 0.0, "gradient": {}, "step_success": True}

    ctx.minimizer.minimize = fake_minimize.__get__(ctx.minimizer, Minimizer)

    execute_command_line(ctx, "gogo")
    assert calls == [1, 2, 3]


def test_macro_recursion_is_rejected():
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "macros": {"loop": "loop"},
        "instructions": [],
    }
    ctx = _build_context(data)
    with pytest.raises(RuntimeError, match="Recursive macro call"):
        execute_command_line(ctx, "loop")


def test_unknown_macro_logs_warning(caplog):
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "macros": {},
        "instructions": [],
    }
    ctx = _build_context(data)
    with caplog.at_level("WARNING"):
        execute_command_line(ctx, "nope")
    assert "Unknown instruction" in caplog.text


def test_history_skips_unknown_instruction():
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "macros": {},
        "instructions": [],
    }
    ctx = _build_context(data)
    execute_command_line(ctx, "nope")
    assert ctx.history == []


def test_history_records_macro_lines():
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "macros": {"gogo": "g 1; g 2; g3"},
        "instructions": [],
    }
    ctx = _build_context(data)
    execute_command_line(ctx, "gogo")
    assert ctx.history == ["g 1", "g 2", "g3"]


def test_macros_survive_polygon_triangulation():
    data = {
        "vertices": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
        "faces": [[0, 1, 2, 3]],
        "macros": {"gogo": "g 1; r; g 2"},
        "instructions": [],
    }
    mesh = parse_geometry(data)
    assert "gogo" in getattr(mesh, "macros", {})


def test_macros_survive_equiangulation_copy():
    from runtime.equiangulation import equiangulate_mesh

    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "macros": {"gogo": "g 1; u; g 2"},
        "instructions": [],
    }
    mesh = parse_geometry(data)
    out = equiangulate_mesh(mesh)
    assert "gogo" in getattr(out, "macros", {})
