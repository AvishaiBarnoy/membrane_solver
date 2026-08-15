"""Opt-in shape fallback policy used after a rejected primary step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from geometry.entities import Mesh
from runtime.steppers.line_search import backtracking_line_search_array


@dataclass(frozen=True)
class TraceZFallbackResult:
    """Outcome and observational data for one trace-z fallback attempt."""

    success: bool
    step_size: float
    energy: float
    stats: dict[str, object]


def shape_scaffold_rejected_step_fallback_mode(global_params) -> str:
    """Return the configured fallback mode using the established validation."""
    raw = global_params.get("shape_scaffold_rejected_step_fallback", "off")
    mode = str(raw or "off").strip().lower()
    allowed = {"off", "trace_z"}
    if mode not in allowed:
        raise ValueError(
            "Unknown shape_scaffold_rejected_step_fallback "
            f"{raw!r}; expected one of {sorted(allowed)}"
        )
    return mode


def shape_scaffold_rejected_step_fallback_enabled(global_params) -> bool:
    """Return whether the trace-z fallback is eligible for this configuration."""
    if shape_scaffold_rejected_step_fallback_mode(global_params) != "trace_z":
        return False
    mesh_op_mode = str(
        global_params.get("rim_slope_match_scaffold_mesh_operation_mode", "") or ""
    ).strip()
    preserve_groups = global_params.get(
        "pin_to_circle_mesh_operation_preserve_normal_groups", []
    )
    if isinstance(preserve_groups, str):
        preserve_groups = [preserve_groups]
    return mesh_op_mode == "preserve_trace_v1" and "trace_layer" in set(
        preserve_groups or []
    )


def scaffold_trace_rows(mesh: Mesh) -> np.ndarray:
    """Return dense rows belonging to the shape-scaffold trace layer."""
    rows: list[int] = []
    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", {}) or {}
        if str(opts.get("pin_to_circle_group") or "") == "trace_layer":
            rows.append(int(row))
    return np.asarray(rows, dtype=int)


def try_shape_scaffold_trace_z_fallback(
    *,
    mesh: Mesh,
    global_params,
    stepper,
    grad_arr: np.ndarray,
    step_size_in: float,
    energy_fn: Callable[[], float],
    reduced_flag: bool,
    accept_rule: str | None,
    constraint_enforcer: Callable[[], None] | None,
    attempted_count: int,
    accepted_count: int,
    publish_stats: Callable[[dict[str, object]], None],
    record_attempt: Callable[[], int],
    record_accept: Callable[[], int],
) -> TraceZFallbackResult:
    """Attempt the established trace-z fallback without lifecycle side effects."""
    mode = shape_scaffold_rejected_step_fallback_mode(global_params)
    stats: dict[str, object] = {
        "mode": mode,
        "attempted": False,
        "accepted": False,
        "attempted_count": int(attempted_count),
        "accepted_count": int(accepted_count),
        "reason": "",
        "trace_count": 0,
        "trace_descent_z_mean": 0.0,
        "trace_dz_mean": 0.0,
        "trace_dz_max": 0.0,
        "energy_before": float("nan"),
        "energy_after": float("nan"),
        "step_size_in": float(step_size_in),
        "step_size_out": float(step_size_in),
    }
    publish_stats(stats)
    if not shape_scaffold_rejected_step_fallback_enabled(global_params):
        stats["reason"] = "disabled"
        return TraceZFallbackResult(False, step_size_in, float("nan"), stats)

    trace_rows = scaffold_trace_rows(mesh)
    stats["trace_count"] = int(trace_rows.size)
    if trace_rows.size == 0:
        stats["reason"] = "no_trace_rows"
        return TraceZFallbackResult(False, step_size_in, float("nan"), stats)

    direction = np.zeros_like(grad_arr)
    direction[trace_rows, 2] = -np.asarray(grad_arr[trace_rows, 2], dtype=float)
    trace_descent_z = direction[trace_rows, 2]
    trace_descent_z_mean = float(np.mean(trace_descent_z))
    stats["trace_descent_z_mean"] = trace_descent_z_mean
    if not np.isfinite(trace_descent_z_mean) or trace_descent_z_mean <= 0.0:
        stats["reason"] = "non_positive_trace_z_descent"
        return TraceZFallbackResult(False, step_size_in, float("nan"), stats)

    before_positions = mesh.positions_view().copy(order="F")
    stats["energy_before"] = float(energy_fn())
    stats["attempted"] = True
    stats["attempted_count"] = record_attempt()
    setattr(mesh, "_line_search_reduced_energy", bool(reduced_flag))
    if reduced_flag and accept_rule is not None:
        setattr(mesh, "_line_search_reduced_accept_rule", str(accept_rule))
    try:
        success, new_step, accepted_energy = backtracking_line_search_array(
            mesh,
            direction,
            grad_arr,
            step_size_in,
            energy_fn,
            mesh.vertex_ids,
            max_iter=getattr(stepper, "max_iter", 10),
            beta=getattr(stepper, "beta", 0.7),
            c=getattr(stepper, "c", 1e-4),
            gamma=getattr(stepper, "gamma", 1.5),
            alpha_max_factor=getattr(stepper, "alpha_max_factor", 10.0),
            constraint_enforcer=constraint_enforcer,
        )
    finally:
        if hasattr(mesh, "_line_search_reduced_energy"):
            delattr(mesh, "_line_search_reduced_energy")
        if hasattr(mesh, "_line_search_reduced_accept_rule"):
            delattr(mesh, "_line_search_reduced_accept_rule")

    after_positions = mesh.positions_view()
    trace_dz = after_positions[trace_rows, 2] - before_positions[trace_rows, 2]
    stats.update(
        {
            "accepted": bool(success),
            "reason": "accepted" if success else "line_search_rejected",
            "step_size_out": float(new_step),
            "energy_after": float(accepted_energy),
            "trace_dz_mean": float(np.mean(trace_dz)) if trace_dz.size else 0.0,
            "trace_dz_max": float(np.max(trace_dz)) if trace_dz.size else 0.0,
            "accepted_count": record_accept() if success else int(accepted_count),
        }
    )
    return TraceZFallbackResult(
        bool(success),
        float(new_step),
        float(accepted_energy),
        stats,
    )
