import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.theory_parity_guarded_gate import evaluate_guarded_gate


def _trend(all_within_tolerance: bool) -> dict:
    return {
        "summary": {
            "all_within_tolerance": bool(all_within_tolerance),
        }
    }


def test_guarded_gate_first_failure_does_not_fail_ci() -> None:
    new_state, decision = evaluate_guarded_gate(
        previous_state={"consecutive_failures": 0},
        trend=_trend(False),
        required_consecutive_failures=2,
    )
    assert new_state["consecutive_failures"] == 1
    assert decision["failed_now"] is True
    assert decision["should_fail"] is False


def test_guarded_gate_second_consecutive_failure_fails_ci() -> None:
    new_state, decision = evaluate_guarded_gate(
        previous_state={"consecutive_failures": 1},
        trend=_trend(False),
        required_consecutive_failures=2,
    )
    assert new_state["consecutive_failures"] == 2
    assert decision["failed_now"] is True
    assert decision["should_fail"] is True


def test_guarded_gate_success_resets_streak() -> None:
    new_state, decision = evaluate_guarded_gate(
        previous_state={"consecutive_failures": 4},
        trend=_trend(True),
        required_consecutive_failures=2,
    )
    assert new_state["consecutive_failures"] == 0
    assert decision["failed_now"] is False
    assert decision["should_fail"] is False
