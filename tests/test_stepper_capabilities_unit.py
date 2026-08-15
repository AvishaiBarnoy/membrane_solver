from __future__ import annotations

from runtime.stepper_capabilities import (
    supports_trial_energy_fn,
    trial_energy_step_kwargs,
)


class _TrialStepper:
    def step(self, *, trial_energy_fn=None):
        return trial_energy_fn


class _PlainStepper:
    def step(self):
        return None


def test_trial_energy_capability_uses_declared_step_signature():
    assert supports_trial_energy_fn(_TrialStepper())
    assert not supports_trial_energy_fn(_PlainStepper())
    assert not supports_trial_energy_fn(object())


def test_trial_energy_step_kwargs_preserve_none_for_supported_steppers():
    assert (
        trial_energy_step_kwargs(
            supports_trial_energy=False, trial_energy_fn=lambda positions: 0.0
        )
        == {}
    )
    assert trial_energy_step_kwargs(
        supports_trial_energy=True, trial_energy_fn=None
    ) == {"trial_energy_fn": None}
