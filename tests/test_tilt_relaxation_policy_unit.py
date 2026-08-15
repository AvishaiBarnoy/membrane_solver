from __future__ import annotations

import pytest

from runtime.tilt_relaxation_policy import resolve_tilt_relaxation_policy


def test_policy_uses_existing_defaults_and_normalizes_values() -> None:
    policy = resolve_tilt_relaxation_policy(
        {
            "tilt_cg_rejection_fallback": " GD ",
            "tilt_projection_cadence": "PER_PASS",
            "tilt_projection_interval": 3,
        }
    )
    assert (
        policy.cg_rejection_fallback,
        policy.projection_cadence,
        policy.projection_interval,
    ) == ("gd", "per_pass", 3)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"tilt_cg_rejection_fallback": "bad"}, "tilt_cg_rejection_fallback"),
        ({"tilt_projection_cadence": "bad"}, "tilt_projection_cadence"),
    ],
)
def test_policy_preserves_validation_errors(params, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_tilt_relaxation_policy(params)


def test_policy_preserves_falsey_interval_defaulting() -> None:
    assert (
        resolve_tilt_relaxation_policy(
            {"tilt_projection_interval": 0}
        ).projection_interval
        == 1
    )
