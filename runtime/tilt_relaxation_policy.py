"""Validated immutable policy inputs for leaflet tilt relaxation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TiltRelaxationPolicy:
    cg_rejection_fallback: str
    projection_cadence: str
    projection_interval: int


def resolve_tilt_relaxation_policy(global_params) -> TiltRelaxationPolicy:
    """Read the established fallback and projection configuration contract."""
    fallback = str(global_params.get("tilt_cg_rejection_fallback", "off") or "off")
    fallback = fallback.strip().lower()
    if fallback not in {"off", "gd"}:
        raise ValueError("tilt_cg_rejection_fallback must be 'off' or 'gd'.")
    cadence = str(
        global_params.get("tilt_projection_cadence", "per_step") or "per_step"
    )
    cadence = cadence.strip().lower()
    if cadence not in {"per_step", "per_pass"}:
        raise ValueError("tilt_projection_cadence must be 'per_step' or 'per_pass'.")
    interval = int(global_params.get("tilt_projection_interval", 1) or 1)
    if interval < 1:
        raise ValueError("tilt_projection_interval must be >= 1.")
    return TiltRelaxationPolicy(fallback, cadence, interval)
