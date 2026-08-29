"""Pure option-inheritance policies for triangle refinement."""

from __future__ import annotations

import numpy as np


def merge_constraints(options: dict, additions: list[str]) -> None:
    """Append missing constraints in their established order."""
    if not additions:
        return
    existing = options.get("constraints")
    if existing is None:
        options["constraints"] = list(additions)
        return
    merged = [existing] if isinstance(existing, str) else list(existing)
    for item in additions:
        if item not in merged:
            merged.append(item)
    options["constraints"] = merged


def has_fixed_constraint(options: dict | None) -> bool:
    if not options:
        return False
    if bool(options.get("fixed", False)):
        return True
    constraints = options.get("constraints")
    return constraints == "fixed" or (
        isinstance(constraints, list) and "fixed" in constraints
    )


def apply_preset_definitions(options: dict, definitions: dict) -> tuple[dict, bool]:
    """Apply preset defaults while retaining explicit option precedence."""
    preset = options.get("preset")
    defaults = definitions.get(preset) if preset else None
    if not isinstance(defaults, dict):
        return options, False

    merged = defaults.copy()
    merged.update(options)

    def as_list(value):
        if value is None:
            return []
        return [value] if isinstance(value, str) else list(value)

    constraints = as_list(defaults.get("constraints"))
    for item in as_list(options.get("constraints")):
        if item not in constraints:
            constraints.append(item)
    if constraints:
        merged["constraints"] = constraints
    else:
        merged.pop("constraints", None)
    merged.setdefault("preset", preset)
    return merged, bool(defaults.get("fixed", False)) or has_fixed_constraint(merged)


def is_ring_like_preset(preset: object, definitions: dict) -> bool:
    options = definitions.get(preset) if preset is not None else None
    return isinstance(options, dict) and any(
        key in options
        for key in (
            "pin_to_circle_group",
            "rim_slope_match_group",
            "tilt_thetaB_group_in",
        )
    )


def choose_midpoint_preset(
    first_options: dict, second_options: dict, definitions: dict
) -> tuple[str | None, bool]:
    """Choose the established deterministic midpoint preset."""
    first = first_options.get("preset")
    second = second_options.get("preset")
    if first is None and second is None:
        return None, False

    def is_disk(preset: object) -> bool:
        return preset is not None and str(preset).startswith("disk")

    first_ring = is_ring_like_preset(first, definitions)
    second_ring = is_ring_like_preset(second, definitions)
    if first is None:
        return (None, False) if second_ring else (second, True)
    if second is None:
        return (None, False) if first_ring else (first, True)
    if first == second:
        return first, True
    if first_ring and not second_ring:
        return second, True
    if second_ring and not first_ring:
        return first, True
    if first_ring and second_ring:
        if first == "disk_edge":
            return second, False
        if second == "disk_edge":
            return first, False
        return first, False
    if first == "disk_edge" or (is_disk(first) and not is_disk(second)):
        return second, True
    if second == "disk_edge" or (is_disk(second) and not is_disk(first)):
        return first, True
    return first, True


def inherit_rigid_disk_group(first_options: dict, second_options: dict) -> dict | None:
    first = first_options.get("rigid_disk_group")
    second = second_options.get("rigid_disk_group")
    if first is None or second is None or str(first) != str(second):
        return None
    return {"rigid_disk_group": str(first)}


def inherit_disk_target_options(
    first_options: dict, second_options: dict
) -> dict | None:
    keys = ("tilt_disk_target_group_in", "tilt_disk_target_group_out")
    merged = {
        key: first
        for key in keys
        if (first := first_options.get(key)) is not None
        and first == second_options.get(key)
    }
    return merged or None


def inherit_disk_interface_tags(
    first_options: dict, second_options: dict
) -> dict | None:
    def belongs_to_disk(options: dict) -> bool:
        return any(
            str(options.get(key) or "").strip() == "disk"
            for key in (
                "tilt_thetaB_group_in",
                "tilt_thetaB_group",
                "rim_slope_match_group",
            )
        )

    if not (belongs_to_disk(first_options) and belongs_to_disk(second_options)):
        return None
    merged = {"rim_slope_match_group": "disk", "tilt_thetaB_group_in": "disk"}
    if (
        str(first_options.get("tilt_thetaB_group") or "") == "disk"
        or str(second_options.get("tilt_thetaB_group") or "") == "disk"
    ):
        merged["tilt_thetaB_group"] = "disk"
    return merged


def _has_constraint(options: dict, name: str) -> bool:
    constraints = options.get("constraints")
    return constraints == name or (
        isinstance(constraints, list) and name in constraints
    )


def _inherit_shared_constraint_options(
    first_options: dict,
    second_options: dict,
    *,
    constraint: str,
    keys: tuple[str, ...],
) -> dict | None:
    if not (
        _has_constraint(first_options, constraint)
        and _has_constraint(second_options, constraint)
    ):
        return None

    merged: dict = {}
    for key in keys:
        first = first_options.get(key)
        second = second_options.get(key)
        if first is None:
            value = second
        elif second is None:
            value = first
        elif isinstance(first, (list, tuple, np.ndarray)) or isinstance(
            second, (list, tuple, np.ndarray)
        ):
            try:
                equal = bool(
                    np.allclose(
                        np.asarray(first, dtype=float),
                        np.asarray(second, dtype=float),
                    )
                )
            except Exception:
                equal = False
            if not equal:
                return None
            value = first
        elif first == second:
            value = first
        else:
            return None
        if value is not None:
            merged[key] = value
    return merged


def inherit_pin_to_circle_options(
    first_options: dict, second_options: dict
) -> dict | None:
    merged = _inherit_shared_constraint_options(
        first_options,
        second_options,
        constraint="pin_to_circle",
        keys=(
            "pin_to_circle_group",
            "pin_to_circle_mode",
            "pin_to_circle_radius",
            "pin_to_circle_normal",
            "pin_to_circle_point",
        ),
    )
    if merged is None:
        return None
    preset = first_options.get("preset")
    if preset is not None and preset == second_options.get("preset"):
        merged["preset"] = preset
    return merged


def inherit_pin_to_plane_options(
    first_options: dict, second_options: dict
) -> dict | None:
    return _inherit_shared_constraint_options(
        first_options,
        second_options,
        constraint="pin_to_plane",
        keys=(
            "pin_to_plane_group",
            "pin_to_plane_mode",
            "pin_to_plane_normal",
            "pin_to_plane_point",
        ),
    )
