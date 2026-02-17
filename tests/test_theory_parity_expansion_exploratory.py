import copy
import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.reproduce_theory_parity import (  # noqa: E402
    _default_state,
    _stage_suffix,
    update_expansion_state,
)

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "tests" / "fixtures" / "theory_parity_expansion_policy.yaml"


def _report(energy: float, ratios: dict[str, float]) -> dict:
    return {
        "metrics": {
            "final_energy": float(energy),
            "reduced_terms": {
                "elastic_measured": 1.0,
                "contact_measured": -1.0,
                "total_measured": float(energy),
            },
            "theory": {
                "ratios": {k: float(v) for k, v in ratios.items()},
            },
        }
    }


@pytest.mark.acceptance
def test_expansion_policy_includes_stage4_refine_then_g10() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    assert _stage_suffix(policy, 4) == ["r", "g10"]


@pytest.mark.acceptance
def test_stage4_promotion_after_three_stage3_converged_runs() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    state = _default_state()
    state["current_stage"] = 3

    ratios = {
        "theta_ratio": 2.0,
        "elastic_ratio": 2.0,
        "contact_ratio": 2.0,
        "total_ratio": 2.0,
    }

    for _ in range(3):
        state, decisions = update_expansion_state(
            state=state,
            report=_report(-0.5, ratios),
            policy=policy,
        )

    assert int(state["current_stage"]) == 4
    assert bool(decisions["promoted_to_stage4"]) is True
    assert state["stage3_anchor_ratios"] == ratios


@pytest.mark.acceptance
def test_stage4_rolls_back_and_locks_after_two_failures() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    state = _default_state()
    state.update(
        {
            "current_stage": 4,
            "stage3_anchor_ratios": {
                "theta_ratio": 2.0,
                "elastic_ratio": 2.0,
                "contact_ratio": 2.0,
                "total_ratio": 2.0,
            },
            "stage3_anchor_energy": -0.5,
            "stage4_fail_streak": 0,
            "stage4_locked": False,
        }
    )

    bad_ratios = {
        "theta_ratio": 3.0,
        "elastic_ratio": 3.0,
        "contact_ratio": 3.0,
        "total_ratio": 3.0,
    }

    state, decisions1 = update_expansion_state(
        state=copy.deepcopy(state),
        report=_report(-0.5, bad_ratios),
        policy=policy,
    )
    assert bool(decisions1["rolled_back_to_stage3"]) is False
    assert int(state["current_stage"]) == 4

    state, decisions2 = update_expansion_state(
        state=state,
        report=_report(-0.5, bad_ratios),
        policy=policy,
    )
    assert bool(decisions2["rolled_back_to_stage3"]) is True
    assert int(state["current_stage"]) == 3
    assert bool(state["stage4_locked"]) is True
