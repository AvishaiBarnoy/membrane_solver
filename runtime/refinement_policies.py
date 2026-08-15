"""Pure refinement option-policy helpers."""

from __future__ import annotations

import numpy as np


def choose_midpoint_preset(
    v1: dict, v2: dict, definitions: dict
) -> tuple[str | None, bool]:
    p1, p2 = v1.get("preset"), v2.get("preset")
    if p1 is None and p2 is None:
        return None, False

    def disk(p):
        return p is not None and str(p).startswith("disk")

    def ring(p):
        value = definitions.get(p)
        return isinstance(value, dict) and any(
            k in value
            for k in (
                "pin_to_circle_group",
                "rim_slope_match_group",
                "tilt_thetaB_group_in",
            )
        )

    if p1 is None:
        return (None, False) if ring(p2) else (p2, True)
    if p2 is None:
        return (None, False) if ring(p1) else (p1, True)
    if p1 == p2:
        return p1, True
    if ring(p1) and not ring(p2):
        return p2, True
    if ring(p2) and not ring(p1):
        return p1, True
    if ring(p1) and ring(p2):
        return (
            (p2, False)
            if p1 == "disk_edge"
            else ((p1, False) if p2 == "disk_edge" else (p1, False))
        )
    if p1 == "disk_edge" or (disk(p1) and not disk(p2)):
        return p2, True
    if p2 == "disk_edge" or (disk(p2) and not disk(p1)):
        return p1, True
    return p1, True


def inherit_rigid_disk_group(v1_options: dict, v2_options: dict) -> dict | None:
    """Return a group only when both parent vertices agree."""
    first = v1_options.get("rigid_disk_group")
    second = v2_options.get("rigid_disk_group")
    if first is None or second is None or str(first) != str(second):
        return None
    return {"rigid_disk_group": str(first)}


def merge_constraints_in_place(options: dict, additions: list[str]) -> None:
    """Append only missing constraints while preserving existing order."""
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
    """Apply a vertex preset while retaining caller-supplied option precedence."""
    preset = options.get("preset")
    if not preset:
        return options, False
    defaults = definitions.get(preset)
    if not isinstance(defaults, dict):
        return options, False

    merged = defaults.copy()
    merged.update(options)

    def as_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    merged_constraints = as_list(defaults.get("constraints"))
    for item in as_list(options.get("constraints")):
        if item not in merged_constraints:
            merged_constraints.append(item)
    if merged_constraints:
        merged["constraints"] = merged_constraints
    else:
        merged.pop("constraints", None)
    if "preset" not in merged:
        merged["preset"] = preset
    preset_fixed = bool(defaults.get("fixed", False)) or has_fixed_constraint(merged)
    return merged, preset_fixed


def is_ring_like_preset(preset: object, definitions: dict) -> bool:
    """Return whether a preset carries boundary-ring constraint metadata."""
    if preset is None:
        return False
    options = definitions.get(preset)
    if not isinstance(options, dict):
        return False
    return any(
        key in options
        for key in (
            "pin_to_circle_group",
            "rim_slope_match_group",
            "tilt_thetaB_group_in",
        )
    )


def inherit_disk_interface_tags(v1_options: dict, v2_options: dict) -> dict | None:
    def disk_group(options: dict) -> bool:
        return any(
            str(options.get(key) or "").strip() == "disk"
            for key in (
                "tilt_thetaB_group_in",
                "tilt_thetaB_group",
                "rim_slope_match_group",
            )
        )

    if not (disk_group(v1_options) and disk_group(v2_options)):
        return None
    result = {"rim_slope_match_group": "disk", "tilt_thetaB_group_in": "disk"}
    if (
        str(v1_options.get("tilt_thetaB_group") or "") == "disk"
        or str(v2_options.get("tilt_thetaB_group") or "") == "disk"
    ):
        result["tilt_thetaB_group"] = "disk"
    return result


def inherit_disk_target_options(v1_options: dict, v2_options: dict) -> dict | None:
    """Retain only matching inner/outer disk target tags."""
    keys = ("tilt_disk_target_group_in", "tilt_disk_target_group_out")
    merged = {
        key: first
        for key in keys
        if (first := v1_options.get(key)) is not None and first == v2_options.get(key)
    }
    return merged or None


def inherit_pin_to_plane_options(v1_options: dict, v2_options: dict) -> dict | None:
    """Return shared plane metadata when both parents carry the constraint."""

    def constrained(options: dict) -> bool:
        value = options.get("constraints")
        return value == "pin_to_plane" or (
            isinstance(value, list) and "pin_to_plane" in value
        )

    if not (constrained(v1_options) and constrained(v2_options)):
        return None
    merged: dict = {}
    for key in (
        "pin_to_plane_group",
        "pin_to_plane_mode",
        "pin_to_plane_normal",
        "pin_to_plane_point",
    ):
        first, second = v1_options.get(key), v2_options.get(key)
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
                        np.asarray(first, dtype=float), np.asarray(second, dtype=float)
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


def inherit_pin_to_circle_options(v1_options: dict, v2_options: dict) -> dict | None:
    """Return shared circle metadata and matching preset for constrained parents."""

    def constrained(options: dict) -> bool:
        value = options.get("constraints")
        return value == "pin_to_circle" or (
            isinstance(value, list) and "pin_to_circle" in value
        )

    if not (constrained(v1_options) and constrained(v2_options)):
        return None
    merged: dict = {}
    for key in (
        "pin_to_circle_group",
        "pin_to_circle_mode",
        "pin_to_circle_radius",
        "pin_to_circle_normal",
        "pin_to_circle_point",
    ):
        first, second = v1_options.get(key), v2_options.get(key)
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
                        np.asarray(first, dtype=float), np.asarray(second, dtype=float)
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
    preset = v1_options.get("preset")
    if preset is not None and preset == v2_options.get("preset"):
        merged["preset"] = preset
    return merged
