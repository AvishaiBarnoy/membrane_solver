"""Parameter and configuration helpers for hard rim-matching constraints."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("membrane_solver")
_WARNED_DISK_EQUALS_RIM = False


def _resolve_group(global_params, key: str) -> str | None:
    raw = None if global_params is None else global_params.get(key)
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _sanitize_disk_group(
    *, rim_group: str | None, disk_group: str | None
) -> str | None:
    """Disable degenerate disk-group coupling when it equals the rim group."""
    if rim_group is None or disk_group is None:
        return disk_group
    if disk_group != rim_group:
        return disk_group
    global _WARNED_DISK_EQUALS_RIM
    if not _WARNED_DISK_EQUALS_RIM:
        logger.warning(
            "rim_slope_match_disk_group matches rim_slope_match_group (%s); "
            "skipping disk-side coupling to avoid degenerate constraints.",
            rim_group,
        )
        _WARNED_DISK_EQUALS_RIM = True
    return None


def _resolve_center(global_params) -> np.ndarray:
    center = (
        None if global_params is None else global_params.get("rim_slope_match_center")
    )
    if center is None:
        center = [0.0, 0.0, 0.0]
    return np.asarray(center, dtype=float).reshape(3)


def _resolve_matching_mode(global_params) -> str:
    raw = None if global_params is None else global_params.get("rim_slope_match_mode")
    mode = "pointwise_radial_v1" if raw is None else str(raw).strip().lower()
    if mode not in {
        "pointwise_radial_v1",
        "ring_average_radial_v1",
        "shared_rim_staggered_v1",
        "physical_edge_staggered_v1",
    }:
        raise ValueError(
            "rim_slope_match_mode must be 'pointwise_radial_v1' or "
            "'ring_average_radial_v1' or 'shared_rim_staggered_v1' or "
            "'physical_edge_staggered_v1'."
        )
    return mode


def _uses_scaffold_trace_lane(global_params, matching_mode: str) -> bool:
    """Return whether the current physical-edge lane uses an explicit scaffold trace shell."""
    if matching_mode != "physical_edge_staggered_v1":
        return False
    trace_radius = (
        None
        if global_params is None
        else global_params.get("parity_trace_layer_radius")
    )
    outer_shells = (
        0
        if global_params is None
        else int(global_params.get("parity_outer_shells") or 0)
    )
    return trace_radius is not None and outer_shells > 0


def _uses_outer_shell_tilt_matching(matching_mode: str) -> bool:
    return matching_mode in {
        "shared_rim_staggered_v1",
        "physical_edge_staggered_v1",
    }


def _use_curved_free_disk_shell2_tilt_continuation(global_params) -> bool:
    """Return whether the shared-rim curved free-disk lane should target shell 2."""
    if global_params is None:
        return False
    return (
        str(global_params.get("rim_slope_match_mode") or "").strip().lower()
        == "shared_rim_staggered_v1"
        and str(global_params.get("rim_slope_match_group") or "").strip() == "rim"
        and str(global_params.get("rim_slope_match_outer_group") or "").strip()
        == "outer"
        and str(global_params.get("rim_slope_match_disk_group") or "").strip() == "disk"
        and str(global_params.get("tilt_thetaB_group_in") or "").strip() == "rim"
        and bool(global_params.get("tilt_out_exclude_shared_rim_outer_rows"))
    )


def _use_disk_theta_targeting(global_params, matching_mode: str) -> bool:
    if matching_mode == "physical_edge_staggered_v1":
        if _uses_scaffold_trace_lane(global_params, matching_mode):
            return False
        return True
    if global_params is None:
        return False
    return bool(str(global_params.get("theory_parity_lane") or "").strip())


def _scaffold_mesh_operation_projection_mode(global_params) -> str:
    """Return scaffold hard-projection behavior for mesh-op/finalize contexts."""
    if global_params is None:
        return "project"
    raw = global_params.get("rim_slope_match_scaffold_mesh_operation_mode")
    mode = str(raw or "project").strip().lower()
    if mode not in {"project", "preserve_trace_v1"}:
        raise ValueError(
            "rim_slope_match_scaffold_mesh_operation_mode must be "
            "'project' or 'preserve_trace_v1'."
        )
    return mode
