import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.completion import command_name_completions


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
