"""Pure capability detection for optimization steppers."""

from __future__ import annotations

import inspect
from collections.abc import Callable


def trial_energy_step_kwargs(
    *,
    supports_trial_energy: bool,
    trial_energy_fn: Callable | None,
) -> dict[str, Callable | None]:
    """Build the legacy optional stepper keyword without changing its value."""
    if not supports_trial_energy:
        return {}
    return {"trial_energy_fn": trial_energy_fn}


def supports_trial_energy_fn(stepper: object) -> bool:
    """Return whether the step method declares the trial-energy callback."""
    step_fn = getattr(stepper, "step", None)
    if step_fn is None:
        return False
    try:
        return "trial_energy_fn" in inspect.signature(step_fn).parameters
    except (TypeError, ValueError):
        return False
