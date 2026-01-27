"""Helpers for mapping physical contact parameters to solver strength values.

The Kozlov/Barnoy-style "contact" contribution used throughout the caveolin
examples is linear in the boundary tilt angle:

    F_cont = -2π R h (Δε / a) θ_B

In a discrete mesh, the corresponding soft driving term is implemented as a
line integral along the tagged rim,

    E_source = -∮ γ (t · r_hat) dl,

with a strength parameter γ that has units of energy per length.

This module provides a small resolver that lets YAML specify the contact
parameters (Δε, a, h) and converts them into the dimensionless γ used by the
solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContactStrength:
    """Resolved contact strength and its components."""

    gamma: float
    """Line strength used by the solver (dimensionless)."""

    gamma_raw: float | None
    """Unscaled γ computed from `h*(Δε/a)` before unit conversion, if available."""


def resolve_contact_line_strength(
    param_resolver,
    obj,
    *,
    strength_key: str,
    contact_suffix: str = "",
) -> ContactStrength:
    """Resolve the line strength `gamma` for a rim source module.

    Resolution order:
    1) `strength_key` (e.g. `tilt_rim_source_strength_in`) if provided.
    2) Contact parameters via `tilt_rim_source_contact_*` keys.

    Contact parameter variants:
    - Direct line strength: `tilt_rim_source_contact_gamma{suffix}`
    - Or via `gamma_raw = h * (Δε/a)`:
      - `tilt_rim_source_contact_h{suffix}`
      - `tilt_rim_source_contact_delta_epsilon_over_a{suffix}` OR
        (`tilt_rim_source_contact_delta_epsilon{suffix}` and
         `tilt_rim_source_contact_a{suffix}`)

    Units / scaling:
    - By default, contact parameters are assumed to already be in solver units.
    - If `tilt_rim_source_contact_units` is set to `si`/`physical`, the computed
      physical line strength (J/m) is converted to solver units via:

        gamma = gamma_phys * L0 / kappa_ref

      where `L0 = tilt_rim_source_contact_length_unit_m` (meters per mesh unit)
      and `kappa_ref = tilt_rim_source_contact_kappa_ref_J` (Joules per solver
      energy unit; typically the monolayer bending modulus when κ=1).
    """
    val = param_resolver.get(obj, strength_key)
    if val is None:
        val = param_resolver.get(None, strength_key)
    if val is not None:
        return ContactStrength(gamma=float(val), gamma_raw=None)

    def get_key(base: str):
        key = f"{base}{contact_suffix}"
        got = param_resolver.get(obj, key)
        if got is None:
            got = param_resolver.get(None, key)
        if got is not None or not contact_suffix:
            return got
        got = param_resolver.get(obj, base)
        if got is None:
            got = param_resolver.get(None, base)
        return got

    gamma_direct = get_key("tilt_rim_source_contact_gamma")
    if gamma_direct is not None:
        gamma_raw = float(gamma_direct)
        return ContactStrength(
            gamma=_convert_contact_strength_units(param_resolver, gamma_raw),
            gamma_raw=gamma_raw,
        )

    h = get_key("tilt_rim_source_contact_h")
    if h is None:
        return ContactStrength(gamma=0.0, gamma_raw=None)

    delta_eps_over_a = get_key("tilt_rim_source_contact_delta_epsilon_over_a")
    if delta_eps_over_a is None:
        delta_eps = get_key("tilt_rim_source_contact_delta_epsilon")
        a = get_key("tilt_rim_source_contact_a")
        if delta_eps is None or a is None:
            return ContactStrength(gamma=0.0, gamma_raw=None)
        delta_eps_over_a = float(delta_eps) / float(a)

    gamma_raw = float(h) * float(delta_eps_over_a)
    return ContactStrength(
        gamma=_convert_contact_strength_units(param_resolver, gamma_raw),
        gamma_raw=gamma_raw,
    )


def _convert_contact_strength_units(param_resolver, gamma_raw: float) -> float:
    units = param_resolver.get(None, "tilt_rim_source_contact_units")
    units = str(units or "solver").strip().lower()
    if units in {"solver", "sim", "simulation", "dimensionless"}:
        return float(gamma_raw)
    if units not in {"si", "physical", "physical_si"}:
        return float(gamma_raw)

    length_unit_m = param_resolver.get(None, "tilt_rim_source_contact_length_unit_m")
    kappa_ref_j = param_resolver.get(None, "tilt_rim_source_contact_kappa_ref_J")
    if length_unit_m is None or kappa_ref_j is None:
        return float(gamma_raw)

    length_unit_m = float(length_unit_m)
    kappa_ref_j = float(kappa_ref_j)
    if abs(length_unit_m) < 1e-30 or abs(kappa_ref_j) < 1e-30:
        return float(gamma_raw)

    return float(gamma_raw) * length_unit_m / kappa_ref_j


__all__ = ["ContactStrength", "resolve_contact_line_strength"]
