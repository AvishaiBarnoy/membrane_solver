"""Canonicalize supported geometry-input aliases in place."""

from __future__ import annotations

import logging

logger = logging.getLogger("membrane_solver")


CONSTRAINT_NAME_ALIASES = {
    "pin_surface_group_to_shape": "pin_to_plane",
}

PIN_TO_PLANE_KEY_ALIASES = {
    "pin_surface_group_to_shape_mode": "pin_to_plane_mode",
    "pin_surface_group_to_shape_group": "pin_to_plane_group",
    "pin_surface_group_to_shape_normal": "pin_to_plane_normal",
    "pin_surface_group_to_shape_point": "pin_to_plane_point",
}


def canonical_constraint_name(name: str) -> str:
    """Return the supported canonical constraint name for ``name``."""
    canonical = CONSTRAINT_NAME_ALIASES.get(name, name)
    if canonical != name:
        logger.info(
            "Constraint alias '%s' mapped to '%s'.",
            name,
            canonical,
        )
    return canonical


def normalize_constraint_names(raw_constraints: object) -> list[str]:
    """Normalize a constraint-name string or list to canonical names."""
    if raw_constraints is None:
        return []
    if isinstance(raw_constraints, str):
        values = [raw_constraints]
    elif isinstance(raw_constraints, list):
        values = list(raw_constraints)
    else:
        err_msg = "constraint modules should be in a list or a single string"
        logger.error(err_msg)
        raise TypeError(err_msg)
    return [canonical_constraint_name(str(value)) for value in values]


def apply_pin_to_plane_aliases_in_place(options: object) -> object:
    """Replace supported pin-to-plane aliases in ``options`` in place.

    Canonical keys retain precedence when both forms are supplied. Non-dict
    values are returned unchanged for compatibility with optional entity input.
    """
    if not isinstance(options, dict):
        return options
    for alias_key, canonical_key in PIN_TO_PLANE_KEY_ALIASES.items():
        if alias_key not in options:
            continue
        if canonical_key not in options:
            options[canonical_key] = options[alias_key]
        options.pop(alias_key, None)
    return options
