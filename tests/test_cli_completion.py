import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.completion import command_line_completions, command_name_completions


def test_command_name_completion_uses_last_semicolon_segment():
    candidates = command_name_completions(
        text="t",
        line_buffer="g10; t",
        command_names=["g", "t", "tf", "tilt_stats"],
        macro_names=[],
    )
    assert "t" in candidates
    assert "tf" in candidates
    assert "tilt_stats" in candidates


def test_command_name_completion_does_not_complete_args():
    candidates = command_name_completions(
        text="x",
        line_buffer="set vertex 0 x",
        command_names=["set", "save"],
        macro_names=[],
    )
    assert candidates == []


def test_energy_subcommand_completion():
    candidates = command_line_completions(
        text="",
        line_buffer="energy ",
        command_names=["energy", "set"],
        macro_names=[],
    )
    assert "breakdown" in candidates
    assert "curvature" in candidates
    assert "total" in candidates
    assert "ref" in candidates


def test_energy_subcommand_completion_prefix():
    candidates = command_line_completions(
        text="c",
        line_buffer="energy c",
        command_names=["energy", "set"],
        macro_names=[],
    )
    assert candidates == ["curvature"]
