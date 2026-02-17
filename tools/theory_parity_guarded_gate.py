#!/usr/bin/env python3
"""Apply guarded CI gating from fixed-lane theory parity trend artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TREND_PATH = (
    Path("benchmarks") / "outputs" / "diagnostics" / "theory_parity_trend.yaml"
)
DEFAULT_STATE_PATH = (
    Path("benchmarks") / "outputs" / "diagnostics" / "theory_parity_ci_state.yaml"
)


def _load_yaml(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else dict(default)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def evaluate_guarded_gate(
    *,
    previous_state: dict[str, Any],
    trend: dict[str, Any],
    required_consecutive_failures: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return updated state and gate decision from one trend artifact."""
    prev_streak = int(previous_state.get("consecutive_failures", 0))
    failed_now = not bool(trend["summary"]["all_within_tolerance"])

    next_streak = prev_streak + 1 if failed_now else 0
    should_fail = failed_now and next_streak >= int(required_consecutive_failures)

    new_state = {
        "consecutive_failures": int(next_streak),
        "last_run_failed": bool(failed_now),
    }
    decision = {
        "failed_now": bool(failed_now),
        "previous_streak": int(prev_streak),
        "next_streak": int(next_streak),
        "required_consecutive_failures": int(required_consecutive_failures),
        "should_fail": bool(should_fail),
    }
    return new_state, decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trend", type=Path, default=DEFAULT_TREND_PATH)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--required-consecutive-failures", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    trend = _load_yaml(Path(args.trend))
    state_path = Path(args.state)
    previous = _load_yaml(state_path, default={})
    next_state, decision = evaluate_guarded_gate(
        previous_state=previous,
        trend=trend,
        required_consecutive_failures=int(args.required_consecutive_failures),
    )
    _save_yaml(state_path, next_state)
    print(
        "guarded_gate:"
        f" failed_now={decision['failed_now']}"
        f" previous_streak={decision['previous_streak']}"
        f" next_streak={decision['next_streak']}"
        f" threshold={decision['required_consecutive_failures']}"
        f" should_fail={decision['should_fail']}"
    )
    return 1 if bool(decision["should_fail"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
