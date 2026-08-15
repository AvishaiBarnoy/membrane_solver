from __future__ import annotations

import pytest

from runtime.refinement_policies import (
    apply_preset_definitions,
    choose_midpoint_preset,
    has_fixed_constraint,
    inherit_disk_interface_tags,
    inherit_disk_target_options,
    inherit_pin_to_circle_options,
    inherit_pin_to_plane_options,
    inherit_rigid_disk_group,
    is_ring_like_preset,
    merge_constraints_in_place,
)


@pytest.mark.parametrize(
    "left,right,definitions,expected",
    [
        ({}, {}, {}, (None, False)),
        ({"preset": "disk"}, {"preset": "disk"}, {}, ("disk", True)),
        ({"preset": "disk_edge"}, {"preset": "disk"}, {}, ("disk", True)),
        ({"preset": "disk"}, {"preset": "membrane"}, {}, ("membrane", True)),
        (
            {"preset": "rim"},
            {"preset": "membrane"},
            {"rim": {"pin_to_circle_group": "rim"}},
            ("membrane", True),
        ),
    ],
)
def test_choose_midpoint_preset_preserves_precedence(
    left, right, definitions, expected
):
    assert choose_midpoint_preset(left, right, definitions) == expected


def test_rigid_disk_group_requires_matching_parent_values():
    assert inherit_rigid_disk_group(
        {"rigid_disk_group": "disk"}, {"rigid_disk_group": "disk"}
    ) == {"rigid_disk_group": "disk"}
    assert inherit_rigid_disk_group({"rigid_disk_group": "disk"}, {}) is None


def test_constraint_merge_preserves_order_and_deduplicates():
    options = {"constraints": "pin_to_circle"}
    merge_constraints_in_place(options, ["pin_to_circle", "pin_to_plane"])
    assert options["constraints"] == ["pin_to_circle", "pin_to_plane"]


def test_fixed_constraint_detection_preserves_supported_forms():
    assert has_fixed_constraint({"fixed": True})
    assert has_fixed_constraint({"constraints": "fixed"})
    assert has_fixed_constraint({"constraints": ["fixed"]})
    assert not has_fixed_constraint({"constraints": ("fixed",)})


def test_preset_definitions_merge_constraints_and_preserve_explicit_options():
    options = {
        "preset": "rim",
        "constraints": "pin_to_circle",
        "pin_to_circle_radius": 2.0,
    }
    merged, fixed = apply_preset_definitions(
        options,
        {
            "rim": {
                "constraints": ["fixed", "pin_to_circle"],
                "pin_to_circle_radius": 1.0,
            }
        },
    )

    assert merged == {
        "preset": "rim",
        "constraints": ["fixed", "pin_to_circle"],
        "pin_to_circle_radius": 2.0,
    }
    assert fixed
    assert options["constraints"] == "pin_to_circle"


def test_preset_definitions_leave_options_unchanged_without_valid_definition():
    options = {"preset": "missing", "constraints": ["fixed"]}
    merged, fixed = apply_preset_definitions(options, {"missing": []})

    assert merged is options
    assert not fixed


@pytest.mark.parametrize(
    "preset,definitions,expected",
    [
        (None, {"rim": {"pin_to_circle_group": "rim"}}, False),
        ("missing", {}, False),
        ("rim", {"rim": {"pin_to_circle_group": "rim"}}, True),
        ("disk", {"disk": {"tilt_thetaB_group_in": "disk"}}, True),
        ("bulk", {"bulk": {"fixed": True}}, False),
    ],
)
def test_ring_like_preset_requires_boundary_metadata(preset, definitions, expected):
    assert is_ring_like_preset(preset, definitions) is expected


def test_disk_interface_tags_require_both_parents_on_disk():
    assert inherit_disk_interface_tags(
        {"rim_slope_match_group": "disk"},
        {"tilt_thetaB_group_in": "disk", "tilt_thetaB_group": "disk"},
    ) == {
        "rim_slope_match_group": "disk",
        "tilt_thetaB_group_in": "disk",
        "tilt_thetaB_group": "disk",
    }
    assert inherit_disk_interface_tags({"rim_slope_match_group": "disk"}, {}) is None


def test_disk_target_options_keep_only_matching_tags():
    assert inherit_disk_target_options(
        {"tilt_disk_target_group_in": "disk", "tilt_disk_target_group_out": "outer"},
        {"tilt_disk_target_group_in": "disk", "tilt_disk_target_group_out": "other"},
    ) == {"tilt_disk_target_group_in": "disk"}


def test_plane_options_accept_equal_vectors_and_reject_conflicts():
    shared = {"constraints": "pin_to_plane", "pin_to_plane_normal": [0, 0, 1]}
    assert inherit_pin_to_plane_options(shared, dict(shared))[
        "pin_to_plane_normal"
    ] == [0, 0, 1]
    assert (
        inherit_pin_to_plane_options(
            shared, {"constraints": "pin_to_plane", "pin_to_plane_normal": [1, 0, 0]}
        )
        is None
    )


def test_circle_options_preserve_matching_preset_and_reject_conflicts():
    shared = {
        "constraints": "pin_to_circle",
        "pin_to_circle_radius": 1.0,
        "preset": "disk_edge",
    }
    assert inherit_pin_to_circle_options(shared, dict(shared))["preset"] == "disk_edge"
    assert (
        inherit_pin_to_circle_options(
            shared, {"constraints": "pin_to_circle", "pin_to_circle_radius": 2.0}
        )
        is None
    )
