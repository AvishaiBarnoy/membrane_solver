#!/usr/bin/env python3
"""Diagnostics-only audit for scaffold parity energy/magnitude imbalance."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import special

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from commands.executor import execute_command_line
from geometry.tilt_operators import (
    _resolve_transport_model,
    compute_divergence_from_basis,
    p1_triangle_divergence,
)
from modules.constraints.local_interface_shells import (
    build_local_interface_shell_data,
    radial_unit_vectors,
)
from modules.constraints.rim_slope_match_out import (
    _build_matching_data,
    matching_residual_diagnostics,
)
from modules.energy import tilt_thetaB_contact_in as thetaB_contact_in
from modules.energy.bending_utils import _compute_effective_areas
from modules.energy.bt_divergence import _inner_recovered_divergence
from modules.energy.bt_params import (
    _assume_J0_center_xy,
    _assume_J0_presets,
    _assume_J0_radius_max,
    _base_term_boundary_group,
    _base_term_reference_mode,
    _base_term_region_mode,
    _base_term_region_radius,
    _use_inner_recovered_divergence,
)
from modules.energy.bt_payload import (
    _leaflet_static_tilt_payload,
    _leaflet_triangle_payload,
)
from modules.energy.bt_selection import (
    _apply_inner_divergence_update_mode,
    _base_term_region_zero_rows,
    _collect_group_rows,
    _collect_preset_rows,
)
from runtime.tilt_optimization import _optimize_thetaB_scalar
from tools.reproduce_theory_parity import (
    DEFAULT_PROTOCOL,
    DEFAULT_TEX_BENDING_MODULUS,
    DEFAULT_TEX_TILT_MODULUS,
    DEFAULT_THEORY_RADIUS,
    _build_context,
    _collect_report_from_context,
    _run_protocol_with_parity_activation,
)
from tools.theory_parity_interface_profiles import (
    build_gap_filled_outer_shell_scaffold_fixture,
    build_outer_shell_scaffold_fixture,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_scaffold_gapfill_eps005_n3_release.yaml"
)
PHYSICAL_EDGE_DEFAULT_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml"
)
PHYSICAL_EDGE_GHOST_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_ghost_eps005.yaml"
)
BASE_THEORY_PARITY_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
FIXED_D_SCAFFOLD_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_scaffold_eps005_n3_d005.yaml"
)
QUICK_PROTOCOL = ("g1",)
LONG_SCAFFOLD_PROTOCOL = (
    "g40",
    "r",
    "V5",
    "g100",
    "V1",
    "energy",
    "V1",
    "energy",
    "V1",
    "energy",
    "V1",
    "energy",
    "V1",
    "energy",
    "V1",
    "energy",
    "V1",
    "energy",
    "V1",
    "energy",
    "V1",
    "energy",
    "V1",
    "energy",
    "V1",
    "energy",
    "V1",
    "energy",
    "V5",
    "energy",
    "V5",
    "energy",
    "V5",
    "energy",
    "V5",
    "energy",
    "V5",
    "energy",
    "V10",
    "energy",
    "V10",
    "energy",
    "V10",
    "energy",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_ratio(num: float, den: float) -> float:
    if abs(float(den)) <= 1.0e-16:
        return 0.0
    return float(num) / float(den)


def _snapshot_state(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    return {
        "positions": mesh.positions_view().copy(),
        "tilts": (
            mesh.tilts_view().copy(order="F") if hasattr(mesh, "tilts_view") else None
        ),
        "tilts_in": mesh.tilts_in_view().copy(order="F"),
        "tilts_out": mesh.tilts_out_view().copy(order="F"),
        "global_params": dict(mesh.global_parameters.to_dict()),
        "step_size": float(getattr(ctx.minimizer, "step_size", 0.0)),
    }


def _restore_state(ctx, state: dict[str, Any]) -> None:
    mesh = ctx.mesh
    mesh.build_position_cache()
    positions = np.asarray(state["positions"], dtype=float)
    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = positions[row]
    tilts = state.get("tilts")
    if tilts is not None:
        mesh.set_tilts_from_array(np.asarray(tilts, dtype=float))
    mesh.set_tilts_in_from_array(np.asarray(state["tilts_in"], dtype=float))
    mesh.set_tilts_out_from_array(np.asarray(state["tilts_out"], dtype=float))
    params = mesh.global_parameters
    params._params.clear()
    params._params.update(copy.deepcopy(state["global_params"]))
    ctx.minimizer.step_size = float(state["step_size"])
    for attr in ("_line_search_reduced_energy", "_line_search_reduced_accept_rule"):
        if hasattr(mesh, attr):
            delattr(mesh, attr)
    mesh.increment_version()


def _state_delta(ctx, state: dict[str, Any]) -> dict[str, float]:
    mesh = ctx.mesh
    return {
        "positions_max_abs": float(
            np.max(np.abs(mesh.positions_view() - np.asarray(state["positions"])))
        ),
        "tilts_in_max_abs": float(
            np.max(np.abs(mesh.tilts_in_view() - np.asarray(state["tilts_in"])))
        ),
        "tilts_out_max_abs": float(
            np.max(np.abs(mesh.tilts_out_view() - np.asarray(state["tilts_out"])))
        ),
        "scalar_param_changed": float(
            dict(mesh.global_parameters.to_dict()) != dict(state["global_params"])
        ),
    }


def _edge_collision_count(mesh) -> int:
    seen: set[tuple[int, int]] = set()
    collisions = 0
    for edge in mesh.edges.values():
        key = tuple(sorted((int(edge.tail_index), int(edge.head_index))))
        if key in seen:
            collisions += 1
        else:
            seen.add(key)
    return collisions


def _row_roles(mesh) -> list[str]:
    roles: list[str] = []
    for vid in mesh.vertex_ids:
        opts = dict(getattr(mesh.vertices[int(vid)], "options", None) or {})
        if opts.get("outer_shell_release_ring"):
            roles.append("release_ring")
        elif opts.get("outer_shell_scaffold_index") is not None:
            roles.append(f"support_shell_{int(opts['outer_shell_scaffold_index'])}")
        elif str(opts.get("pin_to_circle_group") or "") == "trace_layer":
            roles.append("trace_shell")
        elif str(opts.get("rim_slope_match_group") or "") == "rim":
            roles.append("trace_shell")
        elif str(opts.get("preset") or "") == "disk":
            roles.append("disk")
        elif str(opts.get("pin_to_circle_group") or "") == "outer":
            roles.append("outer_boundary")
        else:
            roles.append("outer_bulk")
    return roles


def _mesh_topology_audit(mesh) -> dict[str, Any]:
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    tri_rows, tri_facets = mesh.triangle_row_cache()
    positions = mesh.positions_view()
    roles = _row_roles(mesh)
    role_counts = {role: int(roles.count(role)) for role in sorted(set(roles))}
    ring_radii: dict[str, float] = {}
    radii = np.linalg.norm(positions[:, :2], axis=1)
    for role in sorted(set(roles)):
        rows = np.asarray([i for i, r in enumerate(roles) if r == role], dtype=int)
        if rows.size:
            ring_radii[role] = float(np.mean(radii[rows]))

    tri_count = 0 if tri_rows is None else int(len(tri_rows))
    area_min = 0.0
    area_sum = 0.0
    negative_or_zero_area_count = 0
    if tri_rows is not None and len(tri_rows):
        p0 = positions[tri_rows[:, 0]]
        p1 = positions[tri_rows[:, 1]]
        p2 = positions[tri_rows[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        area_min = float(np.min(area))
        area_sum = float(np.sum(area))
        negative_or_zero_area_count = int(np.count_nonzero(area <= 1.0e-14))

    return {
        "n_vertices": int(len(mesh.vertices)),
        "n_edges": int(len(mesh.edges)),
        "n_facets": int(len(mesh.facets)),
        "n_triangle_rows": tri_count,
        "n_triangle_facets": int(len(tri_facets)),
        "edge_collision_count": int(_edge_collision_count(mesh)),
        "role_counts": role_counts,
        "mean_role_radii": ring_radii,
        "area_sum": area_sum,
        "area_min": area_min,
        "negative_or_zero_area_count": negative_or_zero_area_count,
        "mesh_version": int(getattr(mesh, "_version", 0)),
        "topology_version": int(getattr(mesh, "_topology_version", 0)),
        "vertex_ids_version": int(getattr(mesh, "_vertex_ids_version", 0)),
    }


def _boundary_payload(mesh, positions: np.ndarray):
    try:
        shell_data = build_local_interface_shell_data(
            mesh, positions=positions, group="disk"
        )
    except AssertionError:
        return None
    disk_rows = np.asarray(shell_data.disk_rows, dtype=int)
    if disk_rows.size == 0:
        return None
    _, r_hat = radial_unit_vectors(positions[disk_rows])
    return disk_rows, r_hat


def _representative_rows(mesh) -> dict[str, int]:
    roles = _row_roles(mesh)
    out: dict[str, int] = {}
    for role in ("disk", "trace_shell", "support_shell_1", "outer_bulk"):
        for row, row_role in enumerate(roles):
            if row_role == role:
                out[role] = int(row)
                break
    return out


def _module_energy(ctx, name: str, module, positions, tilts_in, tilts_out) -> float:
    ev = ctx.minimizer._evaluation_manager
    index_map = ctx.mesh.vertex_index_to_row
    scale = float(ctx.minimizer._experimental_energy_scale_for_module(str(name)))
    grad_dummy = np.zeros_like(positions)
    if hasattr(module, "compute_energy_array"):
        try:
            energy = ev._call_module_energy_array(
                module,
                positions=positions,
                index_map=index_map,
                tilts_in=tilts_in,
                tilts_out=tilts_out,
            )
        except TypeError:
            energy = ev._call_module_energy_array(
                module, positions=positions, index_map=index_map
            )
        return float(scale) * float(energy)
    if hasattr(module, "compute_energy_and_gradient_array"):
        try:
            energy = ev._call_module_array(
                module,
                positions=positions,
                index_map=index_map,
                grad_arr=grad_dummy,
                tilts_in=tilts_in,
                tilts_out=tilts_out,
                tilt_in_grad_arr=None,
                tilt_out_grad_arr=None,
            )
        except TypeError:
            energy = ev._call_module_array(
                module,
                positions=positions,
                index_map=index_map,
                grad_arr=grad_dummy,
            )
        return float(scale) * float(energy)
    try:
        energy, _grad = module.compute_energy_and_gradient(
            ctx.mesh,
            ctx.mesh.global_parameters,
            ctx.minimizer.param_resolver,
            compute_gradient=False,
        )
    except TypeError:
        energy, _grad = module.compute_energy_and_gradient(
            ctx.mesh, ctx.mesh.global_parameters, ctx.minimizer.param_resolver
        )
    return float(scale) * float(energy)


def _module_energy_audit(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    positions = mesh.positions_view()
    base_in = mesh.tilts_in_view().copy(order="F")
    base_out = mesh.tilts_out_view().copy(order="F")
    payload = _boundary_payload(mesh, positions)
    direction_in = np.zeros_like(base_in)
    if payload is not None:
        rows, r_hat = payload
        direction_in[rows] = r_hat
    else:
        rows = np.zeros(0, dtype=int)
    eps = 1.0e-6

    breakdown = ctx.minimizer.compute_energy_breakdown()
    modules: dict[str, Any] = {}
    for name, module in zip(
        ctx.minimizer.energy_module_names, ctx.minimizer.energy_modules
    ):
        e0 = _module_energy(ctx, name, module, positions, base_in, base_out)
        e_plus = _module_energy(
            ctx, name, module, positions, base_in + eps * direction_in, base_out
        )
        e_minus = _module_energy(
            ctx, name, module, positions, base_in - eps * direction_in, base_out
        )
        modules[str(name)] = {
            "energy": float(e0),
            "breakdown_energy": float(breakdown.get(str(name), 0.0)),
            "energy_minus_breakdown": float(e0 - float(breakdown.get(str(name), 0.0))),
            "boundary_tilt_slope_fd": float((e_plus - e_minus) / (2.0 * eps)),
        }

    total_slope = float(sum(row["boundary_tilt_slope_fd"] for row in modules.values()))
    return {
        "boundary_rows": int(len(rows)),
        "module_count": int(len(modules)),
        "modules": modules,
        "total_boundary_tilt_slope_fd_by_modules": total_slope,
    }


def _boundary_tilt_direction(ctx) -> tuple[np.ndarray, np.ndarray]:
    mesh = ctx.mesh
    base_in = mesh.tilts_in_view().copy(order="F")
    direction_in = np.zeros_like(base_in)
    payload = _boundary_payload(mesh, mesh.positions_view())
    if payload is None:
        return direction_in, np.zeros(0, dtype=int)
    rows, r_hat = payload
    direction_in[rows] = r_hat
    return direction_in, np.asarray(rows, dtype=int)


def _module_slopes_after_transform(ctx, transform) -> dict[str, Any]:
    mesh = ctx.mesh
    direction_in, rows = _boundary_tilt_direction(ctx)
    if rows.size == 0:
        return {"boundary_rows": 0, "modules": {}, "total_slope": 0.0}
    eps = 1.0e-6
    base_state = _snapshot_state(ctx)
    modules: dict[str, dict[str, float]] = {}
    try:
        for name, module in zip(
            ctx.minimizer.energy_module_names, ctx.minimizer.energy_modules
        ):
            vals: list[float] = []
            for sign in (1.0, -1.0):
                _restore_state(ctx, base_state)
                mesh.set_tilts_in_from_array(
                    np.asarray(base_state["tilts_in"]) + sign * eps * direction_in
                )
                mesh.increment_version()
                transform()
                vals.append(
                    _module_energy(
                        ctx,
                        str(name),
                        module,
                        mesh.positions_view(),
                        mesh.tilts_in_view(),
                        mesh.tilts_out_view(),
                    )
                )
            modules[str(name)] = {
                "boundary_tilt_slope_fd": (vals[0] - vals[1]) / (2.0 * eps)
            }
    finally:
        _restore_state(ctx, base_state)
    total = float(sum(float(row["boundary_tilt_slope_fd"]) for row in modules.values()))
    return {"boundary_rows": int(rows.size), "modules": modules, "total_slope": total}


def _contact_geometry(ctx) -> dict[str, float]:
    mesh = ctx.mesh
    positions = mesh.positions_view()
    payload = thetaB_contact_in._boundary_geometry_payload(
        mesh,
        group=thetaB_contact_in._resolve_group(ctx.minimizer.param_resolver),
        positions=positions,
        param_resolver=ctx.minimizer.param_resolver,
    )
    if payload is None:
        return {
            "boundary_rows": 0,
            "theta_contact_mean": 0.0,
            "arc_length_total": 0.0,
            "R_eff": 0.0,
            "R_theory": float(
                mesh.global_parameters.get("theory_radius") or DEFAULT_THEORY_RADIUS
            ),
        }
    rows, weights, r_hat, r_len, wsum = payload
    theta_vals = np.einsum("ij,ij->i", mesh.tilts_in_view()[rows], r_hat)
    theta_mean = float(np.sum(weights * theta_vals) / wsum) if wsum > 0.0 else 0.0
    r_eff = float(np.sum(weights * r_len) / wsum) if wsum > 0.0 else 0.0
    return {
        "boundary_rows": int(len(rows)),
        "theta_contact_mean": theta_mean,
        "arc_length_total": float(wsum),
        "R_eff": r_eff,
        "R_theory": float(
            mesh.global_parameters.get("theory_radius") or DEFAULT_THEORY_RADIUS
        ),
    }


def _checkpoint(ctx, label: str) -> dict[str, Any]:
    mesh = ctx.mesh
    residual = matching_residual_diagnostics(
        mesh, mesh.global_parameters, mesh.positions_view()
    )
    before = _snapshot_state(ctx)
    energy = float(ctx.minimizer.compute_energy())
    after_compute_delta = _state_delta(ctx, before)
    contact = _contact_geometry(ctx)
    breakdown = ctx.minimizer.compute_energy_breakdown()
    return {
        "label": label,
        "energy": energy,
        "breakdown": {str(k): float(v) for k, v in breakdown.items()},
        "outer_residual_mean": float(
            (residual.get("outer_residual") or {}).get("mean") or 0.0
        ),
        "inner_residual_mean": float(
            (residual.get("inner_residual") or {}).get("mean") or 0.0
        ),
        "thetaB_value": float(mesh.global_parameters.get("tilt_thetaB_value") or 0.0),
        "theta_contact_mean": float(contact["theta_contact_mean"]),
        "energy_eval_mutation": after_compute_delta,
    }


def _run_one_step_cadence_trace(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    minim = ctx.minimizer
    state = _snapshot_state(ctx)
    trace: list[dict[str, Any]] = []
    line_search: dict[str, Any] = {}
    try:
        trace.append(_checkpoint(ctx, "start"))
        if minim._has_enforceable_constraints:
            minim.enforce_constraints_after_mesh_ops(mesh)
            mesh.project_tilts_to_tangent()
            mesh.increment_version()
        trace.append(_checkpoint(ctx, "after_mesh_op_constraints"))
        minim._update_scalar_params()
        trace.append(_checkpoint(ctx, "after_first_scalar_update"))
        tilt_mode = str(minim.global_params.get("tilt_solve_mode", "fixed") or "fixed")
        if minim._uses_leaflet_tilts():
            minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode=tilt_mode)
        else:
            minim._relax_tilts(positions=mesh.positions_view(), mode=tilt_mode)
        trace.append(_checkpoint(ctx, "after_tilt_relaxation"))
        minim._update_scalar_params()
        trace.append(_checkpoint(ctx, "after_second_scalar_update"))
        if minim._uses_leaflet_tilts():
            _optimize_thetaB_scalar(minim, tilt_mode=tilt_mode, iteration=0)
        trace.append(_checkpoint(ctx, "after_thetaB_scalar_optimize"))
        energy, grad_arr = minim.compute_energy_and_gradient_array()
        minim.project_constraints_array(grad_arr)
        line_search["pre_step_energy"] = float(energy)
        line_search["projected_gradient_norm"] = float(np.linalg.norm(grad_arr))
        trace.append(_checkpoint(ctx, "after_gradient_projection"))
        reduced_flag = (
            bool(minim.global_params.get("line_search_reduced_energy", False))
            and int(
                minim.global_params.get("line_search_reduced_tilt_inner_steps", 0) or 0
            )
            > 0
        )
        setattr(mesh, "_line_search_reduced_energy", reduced_flag)
        if reduced_flag:
            setattr(
                mesh,
                "_line_search_reduced_accept_rule",
                str(
                    minim.global_params.get("line_search_reduced_accept_rule", "armijo")
                    or "armijo"
                ),
            )
        try:
            step_mode = str(
                minim.global_params.get("step_size_mode", "adaptive") or "adaptive"
            ).lower()
            fixed_step = float(
                minim.global_params.get("step_size", minim.step_size) or minim.step_size
            )
            step_size_in = fixed_step if step_mode == "fixed" else minim.step_size
            kwargs = {}
            if minim._stepper_supports_trial_energy_fn():
                kwargs["trial_energy_fn"] = minim._line_search_trial_energy_fn()
            success, new_step, accepted_energy = minim.stepper.step(
                mesh,
                grad_arr,
                step_size_in,
                minim._line_search_energy_fn(),
                constraint_enforcer=(
                    minim._enforce_constraints
                    if minim._has_enforceable_constraints
                    else None
                ),
                **kwargs,
            )
            line_search.update(
                {
                    "attempted": True,
                    "success": bool(success),
                    "new_step_size": float(new_step),
                    "accepted_energy": float(accepted_energy),
                    "reduced_energy": bool(reduced_flag),
                }
            )
        finally:
            for attr in (
                "_line_search_reduced_energy",
                "_line_search_reduced_accept_rule",
            ):
                if hasattr(mesh, attr):
                    delattr(mesh, attr)
        trace.append(_checkpoint(ctx, "after_line_search"))
        mesh.project_tilts_to_tangent()
        mesh.increment_version()
        trace.append(_checkpoint(ctx, "after_post_step_tangent_projection"))
    except Exception as exc:  # pragma: no cover - diagnostics path
        line_search["error"] = str(exc)
    finally:
        restore_delta_before_restore = _state_delta(ctx, state)
        _restore_state(ctx, state)
    return {
        "checkpoints": trace,
        "line_search": line_search,
        "state_delta_before_restore": restore_delta_before_restore,
        "state_delta_after_restore": _state_delta(ctx, state),
    }


def _coupled_stationarity_audit(ctx) -> dict[str, Any]:
    state = _snapshot_state(ctx)

    def no_op() -> None:
        return None

    def enforce_constraints() -> None:
        ctx.minimizer._enforce_constraints()
        ctx.mesh.increment_version()

    def enforce_and_relax() -> None:
        enforce_constraints()
        mode = str(
            ctx.minimizer.global_params.get("tilt_solve_mode", "fixed") or "fixed"
        )
        ctx.minimizer._relax_leaflet_tilts(
            positions=ctx.mesh.positions_view(), mode=mode
        )

    def line_search_trial_like() -> None:
        before = _snapshot_state(ctx)
        _ = float(ctx.minimizer._line_search_energy_fn()())
        # Keep any trial-time reduced tilt/scalar mutation visible to the slope probe.
        _ = before

    try:
        before_enforce = float(ctx.minimizer.compute_energy())
        ctx.minimizer._enforce_constraints()
        after_enforce = float(ctx.minimizer.compute_energy())
        _restore_state(ctx, state)
        states = {
            "fixed_state": _module_slopes_after_transform(ctx, no_op),
            "constrained_state": _module_slopes_after_transform(
                ctx, enforce_constraints
            ),
            "constrained_tilt_relaxed": _module_slopes_after_transform(
                ctx, enforce_and_relax
            ),
            "line_search_trial_like": _module_slopes_after_transform(
                ctx, line_search_trial_like
            ),
        }
        trace = _run_one_step_cadence_trace(ctx)
    finally:
        _restore_state(ctx, state)
    return {
        "energy_delta_after_enforce": float(after_enforce - before_enforce),
        "states": states,
        "cadence_trace": trace,
        "state_delta_after_audit": _state_delta(ctx, state),
    }


def _triangle_role_bins(mesh) -> dict[str, Any]:
    mesh.build_facet_vertex_loops()
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return {}
    positions = mesh.positions_view()
    roles = np.asarray(_row_roles(mesh), dtype=object)
    p0 = positions[tri_rows[:, 0]]
    p1 = positions[tri_rows[:, 1]]
    p2 = positions[tri_rows[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    out: dict[str, dict[str, float]] = {}
    for tri, area in zip(tri_rows, areas):
        tri_roles = [str(roles[int(row)]) for row in tri]
        role = max(set(tri_roles), key=tri_roles.count)
        entry = out.setdefault(role, {"triangle_count": 0.0, "area": 0.0})
        entry["triangle_count"] += 1.0
        entry["area"] += float(area)
    return out


def _bulk_boundary_split(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    tin = mesh.tilts_in_view()
    tout = mesh.tilts_out_view()
    roles = _row_roles(mesh)
    out: dict[str, Any] = {}
    for role in sorted(set(roles)):
        rows = np.asarray([i for i, r in enumerate(roles) if r == role], dtype=int)
        out[role] = {
            "vertex_count": int(rows.size),
            "tilt_in_norm_mean": (
                float(np.mean(np.linalg.norm(tin[rows], axis=1))) if rows.size else 0.0
            ),
            "tilt_out_norm_mean": (
                float(np.mean(np.linalg.norm(tout[rows], axis=1))) if rows.size else 0.0
            ),
        }
    return {"vertex_bins": out, "triangle_area_bins": _triangle_role_bins(mesh)}


def _role_rows(mesh) -> dict[str, np.ndarray]:
    roles = np.asarray(_row_roles(mesh), dtype=object)
    return {
        str(role): np.flatnonzero(roles == role).astype(int)
        for role in sorted(set(str(x) for x in roles))
    }


def _module_gradient_norms_by_role(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    roles = _role_rows(mesh)
    out: dict[str, Any] = {}
    for name, module in zip(
        ctx.minimizer.energy_module_names, ctx.minimizer.energy_modules
    ):
        grad = np.zeros_like(positions)
        tin_grad = np.zeros_like(mesh.tilts_in_view())
        tout_grad = np.zeros_like(mesh.tilts_out_view())
        try:
            if hasattr(module, "compute_energy_and_gradient_array"):
                ctx.minimizer._evaluation_manager._call_module_array(
                    module,
                    positions=positions,
                    index_map=index_map,
                    grad_arr=grad,
                    tilts_in=mesh.tilts_in_view(),
                    tilts_out=mesh.tilts_out_view(),
                    tilt_in_grad_arr=tin_grad,
                    tilt_out_grad_arr=tout_grad,
                )
            else:
                continue
        except TypeError:
            try:
                ctx.minimizer._evaluation_manager._call_module_array(
                    module,
                    positions=positions,
                    index_map=index_map,
                    grad_arr=grad,
                )
            except Exception:
                continue
        except Exception:
            continue
        scale = float(ctx.minimizer._experimental_energy_scale_for_module(str(name)))
        grad *= scale
        tin_grad *= scale
        tout_grad *= scale
        out[str(name)] = {
            role: {
                "rows": int(rows.size),
                "shape_grad_norm": float(np.linalg.norm(grad[rows])),
                "tilt_in_grad_norm": float(np.linalg.norm(tin_grad[rows])),
                "tilt_out_grad_norm": float(np.linalg.norm(tout_grad[rows])),
            }
            for role, rows in roles.items()
        }
    return out


def _theory_profile_tilt_field(ctx) -> np.ndarray:
    mesh = ctx.mesh
    positions = mesh.positions_view()
    gp = mesh.global_parameters
    kappa = float(gp.get("bending_modulus_in") or gp.get("bending_modulus") or 1.0)
    kappa_t = float(gp.get("tilt_modulus_in") or 1.0)
    radius = float(gp.get("theory_radius") or DEFAULT_THEORY_RADIUS)
    theta = float(
        gp.get("tilt_thetaB_value") or _contact_geometry(ctx)["theta_contact_mean"]
    )
    lam = float(np.sqrt(kappa_t / kappa)) if kappa > 0.0 and kappa_t > 0.0 else 0.0
    radii = np.linalg.norm(positions[:, :2], axis=1)
    _, r_hat = radial_unit_vectors(positions)
    field = np.zeros_like(mesh.tilts_in_view())
    if lam <= 0.0 or radius <= 0.0:
        return field
    inside = radii <= radius + 1.0e-12
    x_R = lam * radius
    i1_R = float(special.iv(1, x_R))
    k1_R = float(special.kv(1, x_R))
    if abs(i1_R) > 1.0e-16:
        field[inside] = (
            theta * (special.iv(1, lam * radii[inside]) / i1_R)[:, None] * r_hat[inside]
        )
    outside = ~inside
    if abs(k1_R) > 1.0e-16:
        field[outside] = (
            theta
            * (special.kv(1, lam * np.maximum(radii[outside], radius)) / k1_R)[:, None]
            * r_hat[outside]
        )
    return field


def _elastic_field_probe(
    ctx, label: str, tilts_in: np.ndarray, tilts_out: np.ndarray
) -> dict[str, float]:
    mesh = ctx.mesh
    state = _snapshot_state(ctx)
    try:
        mesh.set_tilts_in_from_array(tilts_in)
        mesh.set_tilts_out_from_array(tilts_out)
        mesh.increment_version()
        breakdown = ctx.minimizer.compute_energy_breakdown()
        return {
            "label": label,
            "bending_tilt_in": float(breakdown.get("bending_tilt_in") or 0.0),
            "bending_tilt_out": float(breakdown.get("bending_tilt_out") or 0.0),
            "tilt_in": float(breakdown.get("tilt_in") or 0.0),
            "tilt_out": float(breakdown.get("tilt_out") or 0.0),
            "elastic": float(
                (breakdown.get("bending_tilt_in") or 0.0)
                + (breakdown.get("bending_tilt_out") or 0.0)
                + (breakdown.get("tilt_in") or 0.0)
                + (breakdown.get("tilt_out") or 0.0)
            ),
        }
    finally:
        _restore_state(ctx, state)


def _elastic_magnitude_audit(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    breakdown = ctx.minimizer.compute_energy_breakdown()
    roles = _role_rows(mesh)
    tin = mesh.tilts_in_view().copy(order="F")
    tout = mesh.tilts_out_view().copy(order="F")
    radial = np.zeros_like(tin)
    _, r_hat = radial_unit_vectors(mesh.positions_view())
    radial[:] = r_hat
    theta = float(_contact_geometry(ctx)["theta_contact_mean"])
    probes = [
        _elastic_field_probe(ctx, "current", tin, tout),
        _elastic_field_probe(ctx, "zero_tilt", np.zeros_like(tin), np.zeros_like(tout)),
        _elastic_field_probe(
            ctx, "uniform_radial_theta", theta * radial, theta * radial
        ),
        _elastic_field_probe(
            ctx,
            "theory_bessel_in_only",
            _theory_profile_tilt_field(ctx),
            np.zeros_like(tout),
        ),
    ]
    role_stats: dict[str, Any] = {}
    for role, rows in roles.items():
        role_stats[role] = {
            "rows": int(rows.size),
            "tilt_in_norm_mean": (
                float(np.mean(np.linalg.norm(tin[rows], axis=1))) if rows.size else 0.0
            ),
            "tilt_out_norm_mean": (
                float(np.mean(np.linalg.norm(tout[rows], axis=1))) if rows.size else 0.0
            ),
        }
    return {
        "elastic_breakdown": {
            "bending_tilt_in": float(breakdown.get("bending_tilt_in") or 0.0),
            "bending_tilt_out": float(breakdown.get("bending_tilt_out") or 0.0),
            "tilt_in": float(breakdown.get("tilt_in") or 0.0),
            "tilt_out": float(breakdown.get("tilt_out") or 0.0),
        },
        "role_stats": role_stats,
        "module_gradient_norms_by_role": _module_gradient_norms_by_role(ctx),
        "field_probes": probes,
        "state_delta_after_audit": _state_delta(ctx, _snapshot_state(ctx)),
    }


def _row_summary(values: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    if rows.size == 0:
        return {
            "count": 0.0,
            "sum": 0.0,
            "mean": 0.0,
            "abs_sum": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    vals = np.asarray(values, dtype=float)[rows]
    return {
        "count": float(rows.size),
        "sum": float(np.sum(vals)),
        "mean": float(np.mean(vals)),
        "abs_sum": float(np.sum(np.abs(vals))),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _leaflet_base_term_decomposition(ctx, *, leaflet: str) -> dict[str, Any]:
    mesh = ctx.mesh
    gp = mesh.global_parameters
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    cache_tag = str(leaflet)
    kappa_key = f"bending_modulus_{cache_tag}"
    tilts = mesh.tilts_in_view() if cache_tag == "in" else mesh.tilts_out_view()
    div_sign = 1.0 if cache_tag == "in" else -1.0
    payload = _leaflet_triangle_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        cache_tag=cache_tag,
        ctx=ctx.minimizer.energy_context(),
    )
    tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
    tri_rows_full = np.asarray(payload["tri_rows_full"], dtype=np.int32)
    if tri_rows.size == 0 or tri_rows_full.size == 0:
        return {"leaflet": cache_tag, "available": False}

    k_vecs = np.asarray(payload["k_vecs"], dtype=float)
    vertex_areas_vor = np.asarray(payload["vertex_areas_vor"], dtype=float)
    weights = np.asarray(payload["weights"], dtype=float)
    transport_model = _resolve_transport_model(
        gp.get("tilt_transport_model", "ambient_v1") if gp is not None else "ambient_v1"
    )
    g0_use = payload["g0"]
    g1_use = payload["g1"]
    g2_use = payload["g2"]
    if g0_use is not None and g1_use is not None and g2_use is not None:
        div_tri = compute_divergence_from_basis(
            mesh=mesh,
            tilts=tilts,
            tri_rows=tri_rows,
            g0=np.asarray(g0_use, dtype=float),
            g1=np.asarray(g1_use, dtype=float),
            g2=np.asarray(g2_use, dtype=float),
            positions=positions,
            transport_model=transport_model,
        )
    else:
        div_tri, _, _g0, _g1, _g2 = p1_triangle_divergence(
            mesh=mesh,
            positions=positions,
            tilts=tilts,
            tri_rows=tri_rows,
            transport_model=transport_model,
        )
    div_term = float(div_sign) * div_tri
    div_term = _apply_inner_divergence_update_mode(
        mesh,
        gp,
        positions=positions,
        tri_rows=tri_rows,
        cache_tag=cache_tag,
        div_term=div_term,
    )
    use_recovered_div = _use_inner_recovered_divergence(gp, cache_tag=cache_tag)
    if use_recovered_div:
        tri_area = payload.get("tri_area")
        if tri_area is None:
            tri_area = mesh.p1_triangle_shape_gradient_cache(positions)[0]
        div_eval_tri, _, _div_eval_vertex_area = _inner_recovered_divergence(
            global_params=gp,
            cache_tag=cache_tag,
            tri_rows=tri_rows,
            tri_area=np.asarray(tri_area, dtype=float),
            div_tri=div_term,
            n_vertices=len(mesh.vertex_ids),
            ctx=ctx.minimizer.energy_context(),
            scratch_tag=f"btl_diag_{cache_tag}",
        )
    else:
        div_eval_tri = div_term

    vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token=f"bending_tilt_leaflet_diag_{cache_tag}",
        compute_vertex_areas=True,
    )
    static = _leaflet_static_tilt_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        k_vecs=k_vecs,
        vertex_areas_vor=vertex_areas_vor,
        tri_rows=tri_rows,
        kappa_key=kappa_key,
        cache_tag=cache_tag,
    )
    base_term = np.asarray(static["base_term"], dtype=float)
    c0_arr = np.asarray(static["c0_arr"], dtype=float)
    h_vor = np.asarray(static["h_vor"], dtype=float)
    kappa_arr = np.asarray(static["kappa_arr"], dtype=float)
    is_interior = np.asarray(static["is_interior"], dtype=bool)
    base_tri = base_term[tri_rows]
    div_tri_eval = np.asarray(div_eval_tri, dtype=float)
    kappa_tri = kappa_arr[tri_rows]

    base_vertex = np.zeros(len(mesh.vertex_ids), dtype=float)
    div_vertex = np.zeros(len(mesh.vertex_ids), dtype=float)
    cross_vertex = np.zeros(len(mesh.vertex_ids), dtype=float)
    total_vertex = np.zeros(len(mesh.vertex_ids), dtype=float)
    for col, va in enumerate((va0_eff, va1_eff, va2_eff)):
        rows = tri_rows[:, col]
        base_e = 0.5 * kappa_tri[:, col] * base_tri[:, col] ** 2 * va
        div_e = 0.5 * kappa_tri[:, col] * div_tri_eval**2 * va
        cross_e = kappa_tri[:, col] * base_tri[:, col] * div_tri_eval * va
        np.add.at(base_vertex, rows, base_e)
        np.add.at(div_vertex, rows, div_e)
        np.add.at(cross_vertex, rows, cross_e)
        np.add.at(total_vertex, rows, base_e + div_e + cross_e)

    raw_base = 2.0 * h_vor - c0_arr
    presets = _assume_J0_presets(gp, cache_tag=cache_tag)
    radius_max = _assume_J0_radius_max(gp, cache_tag=cache_tag)
    center_xy = _assume_J0_center_xy(gp)
    preset_rows = (
        _collect_preset_rows(
            mesh,
            presets=presets,
            cache_tag=cache_tag,
            index_map=index_map,
            radius_max=radius_max,
            center_xy=center_xy,
        )
        if presets
        else np.zeros(0, dtype=int)
    )
    region_rows = _base_term_region_zero_rows(
        mesh,
        gp,
        cache_tag=cache_tag,
        index_map=index_map,
    )
    boundary_group = _base_term_boundary_group(gp, cache_tag=cache_tag)
    group_rows = (
        _collect_group_rows(mesh, group=boundary_group, index_map=index_map)
        if boundary_group
        else np.zeros(0, dtype=int)
    )
    boundary_rows = np.asarray(
        [index_map[vid] for vid in mesh.boundary_vertex_ids if vid in index_map],
        dtype=int,
    )
    zeroed_by_any = np.flatnonzero(
        (np.abs(raw_base) > 1.0e-14) & (np.abs(base_term) <= 1.0e-14)
    )

    roles = _role_rows(mesh)
    role_summaries: dict[str, Any] = {}
    for role, rows in roles.items():
        role_summaries[role] = {
            "rows": int(rows.size),
            "base_term": _row_summary(base_term, rows),
            "raw_base_term": _row_summary(raw_base, rows),
            "h_vor": _row_summary(h_vor, rows),
            "c0": _row_summary(c0_arr, rows),
            "vertex_area_vor": _row_summary(vertex_areas_vor, rows),
            "vertex_area_eff": _row_summary(vertex_areas_eff, rows),
            "base_energy": _row_summary(base_vertex, rows),
            "div_energy": _row_summary(div_vertex, rows),
            "cross_energy": _row_summary(cross_vertex, rows),
            "total_energy": _row_summary(total_vertex, rows),
        }

    module_name = f"bending_tilt_{cache_tag}"
    module_energy = float(
        ctx.minimizer.compute_energy_breakdown().get(module_name) or 0.0
    )
    return {
        "leaflet": cache_tag,
        "available": True,
        "module_energy": module_energy,
        "decomposed_total_energy": float(np.sum(total_vertex)),
        "energy_minus_decomposition": float(module_energy - np.sum(total_vertex)),
        "base_energy": float(np.sum(base_vertex)),
        "div_energy": float(np.sum(div_vertex)),
        "cross_energy": float(np.sum(cross_vertex)),
        "base_term_abs_sum": float(np.sum(np.abs(base_term))),
        "raw_base_term_abs_sum": float(np.sum(np.abs(raw_base))),
        "interior_rows": int(np.count_nonzero(is_interior)),
        "noninterior_rows": int(np.count_nonzero(~is_interior)),
        "zeroed_rows": {
            "boundary": int(boundary_rows.size),
            "boundary_group": int(group_rows.size),
            "presets": int(preset_rows.size),
            "region": int(region_rows.size),
            "raw_nonzero_to_zero": int(zeroed_by_any.size),
        },
        "config": {
            "assume_J0_presets": list(presets),
            "assume_J0_radius_max": None if radius_max is None else float(radius_max),
            "base_term_boundary_group": boundary_group or "",
            "base_term_reference_mode": _base_term_reference_mode(gp),
            "base_term_region_mode": _base_term_region_mode(gp),
            "base_term_region_radius": (
                None
                if _base_term_region_radius(gp) is None
                else float(_base_term_region_radius(gp))
            ),
            "spontaneous_curvature": float(gp.get("spontaneous_curvature") or 0.0),
            f"spontaneous_curvature_{cache_tag}": float(
                gp.get(f"spontaneous_curvature_{cache_tag}") or 0.0
            ),
        },
        "role_summaries": role_summaries,
    }


def _bending_tilt_base_term_audit(ctx) -> dict[str, Any]:
    state = _snapshot_state(ctx)
    try:
        leaflets = {
            "in": _leaflet_base_term_decomposition(ctx, leaflet="in"),
            "out": _leaflet_base_term_decomposition(ctx, leaflet="out"),
        }
    finally:
        _restore_state(ctx, state)
    return {
        "leaflets": leaflets,
        "state_delta_after_audit": _state_delta(ctx, state),
    }


def _base_term_summary_for_fixture(path: Path, label: str) -> dict[str, Any]:
    ctx = _build_context(path)
    audit = _bending_tilt_base_term_audit(ctx)
    out = {"label": label, "fixture": str(path.name), "leaflets": {}}
    for leaflet, row in audit["leaflets"].items():
        if not bool(row.get("available", False)):
            out["leaflets"][leaflet] = {"available": False}
            continue
        role_totals = {
            role: float(summary["total_energy"]["sum"])
            for role, summary in row["role_summaries"].items()
        }
        largest_role = (
            max(role_totals, key=lambda key: abs(role_totals[key]))
            if role_totals
            else ""
        )
        out["leaflets"][leaflet] = {
            "available": True,
            "module_energy": float(row["module_energy"]),
            "base_energy": float(row["base_energy"]),
            "div_energy": float(row["div_energy"]),
            "cross_energy": float(row["cross_energy"]),
            "largest_role": largest_role,
            "largest_role_total_energy": float(role_totals.get(largest_role, 0.0)),
            "zeroed_rows": row["zeroed_rows"],
            "config": row["config"],
        }
    return out


def _base_term_fixture_comparison() -> dict[str, Any]:
    fixtures = [
        ("scaffold_gapfill_initial", DEFAULT_FIXTURE),
        ("physical_edge_default_initial", PHYSICAL_EDGE_DEFAULT_FIXTURE),
        ("physical_edge_ghost_initial", PHYSICAL_EDGE_GHOST_FIXTURE),
        ("base_theory_parity_initial", BASE_THEORY_PARITY_FIXTURE),
    ]
    rows: list[dict[str, Any]] = []
    for label, path in fixtures:
        try:
            rows.append(_base_term_summary_for_fixture(path, label))
        except Exception as exc:  # pragma: no cover - diagnostics path
            rows.append({"label": label, "fixture": str(path.name), "error": str(exc)})
    return {"mode": "initial_state", "fixtures": rows}


def _constraint_audit(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    positions = mesh.positions_view()
    before = matching_residual_diagnostics(mesh, mesh.global_parameters, positions)
    energy_before = float(ctx.minimizer.compute_energy())
    ctx.minimizer._enforce_constraints()
    after = matching_residual_diagnostics(
        mesh, mesh.global_parameters, mesh.positions_view()
    )
    energy_after = float(ctx.minimizer.compute_energy())

    def _outer_abs(diag: dict[str, Any]) -> float:
        return abs(_as_float((diag.get("outer_residual") or {}).get("mean")))

    def _inner_abs(diag: dict[str, Any]) -> float:
        return abs(_as_float((diag.get("inner_residual") or {}).get("mean")))

    return {
        "available_before": bool(before.get("available", False)),
        "available_after": bool(after.get("available", False)),
        "outer_residual_abs_before": _outer_abs(before),
        "outer_residual_abs_after": _outer_abs(after),
        "inner_residual_abs_before": _inner_abs(before),
        "inner_residual_abs_after": _inner_abs(after),
        "energy_delta_after_enforce": float(energy_after - energy_before),
        "constraint_modules": [str(x) for x in mesh.constraint_modules],
    }


def _row_position_summary(positions: np.ndarray, rows) -> dict[str, Any]:
    row_arr = np.asarray(rows if rows is not None else [], dtype=int)
    if row_arr.size == 0:
        return {
            "count": 0,
            "radius_mean": 0.0,
            "radius_median": 0.0,
            "height_mean": 0.0,
            "height_median": 0.0,
        }
    pos = positions[row_arr]
    radii = np.linalg.norm(pos[:, :2], axis=1)
    return {
        "count": int(row_arr.size),
        "radius_mean": float(np.mean(radii)),
        "radius_median": float(np.median(radii)),
        "height_mean": float(np.mean(pos[:, 2])),
        "height_median": float(np.median(pos[:, 2])),
    }


def _interface_target_audit(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    state = _snapshot_state(ctx)
    positions = mesh.positions_view()
    data = _build_matching_data(mesh, mesh.global_parameters, positions)
    before = matching_residual_diagnostics(mesh, mesh.global_parameters, positions)
    if data is None:
        return {
            "available": False,
            "reason": "missing_matching_data",
            "diagnostics_before": before,
            "state_delta_after_audit": _state_delta(ctx, state),
        }
    row_summaries = {
        "disk_rows": _row_position_summary(positions, data.get("disk_rows")),
        "rim_rows": _row_position_summary(positions, data.get("rim_rows")),
        "outer_rows": _row_position_summary(positions, data.get("outer_rows")),
        "tilt_rows": _row_position_summary(positions, data.get("tilt_rows")),
    }

    ctx.minimizer._enforce_constraints()
    after = matching_residual_diagnostics(
        mesh, mesh.global_parameters, mesh.positions_view()
    )
    projection_delta = _state_delta(ctx, state)
    _restore_state(ctx, state)

    return {
        "available": True,
        "matching_mode": str(before.get("matching_mode", "")),
        "target_source": str(data.get("target_source", "unknown")),
        "row_summaries": row_summaries,
        "projection_delta": projection_delta,
        "diagnostics_before": before,
        "diagnostics_after": after,
        "state_delta_after_audit": _state_delta(ctx, state),
    }


def _position_direction_for_role(ctx, role: str, mode: str) -> np.ndarray:
    mesh = ctx.mesh
    positions = mesh.positions_view()
    direction = np.zeros_like(positions)
    rows = _role_rows(mesh).get(role, np.zeros(0, dtype=int))
    if rows.size == 0:
        return direction
    if mode == "z":
        direction[rows, 2] = 1.0
    elif mode == "radial":
        _, r_hat = radial_unit_vectors(positions[rows])
        direction[rows] = r_hat
    return direction


def _set_positions_from_array(ctx, positions: np.ndarray) -> None:
    mesh = ctx.mesh
    mesh.build_position_cache()
    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = positions[row]
    mesh.increment_version()


def _finite_difference_total_slope(ctx, perturb, transform=None) -> float:
    state = _snapshot_state(ctx)
    eps = 1.0e-6
    vals: list[float] = []
    try:
        for sign in (1.0, -1.0):
            _restore_state(ctx, state)
            perturb(sign * eps)
            if transform is not None:
                transform()
            vals.append(float(ctx.minimizer.compute_energy()))
    finally:
        _restore_state(ctx, state)
    return float((vals[0] - vals[1]) / (2.0 * eps))


def _constrained_gradient_audit(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    positions = mesh.positions_view()
    energy, grad_arr = ctx.minimizer.compute_energy_and_gradient_array()
    raw_grad = np.asarray(grad_arr, dtype=float).copy()
    projected_grad = raw_grad.copy()
    ctx.minimizer.project_constraints_array(projected_grad)

    def enforce() -> None:
        ctx.minimizer._enforce_constraints()
        mesh.increment_version()

    def enforce_relax() -> None:
        enforce()
        mode = str(
            ctx.minimizer.global_params.get("tilt_solve_mode", "fixed") or "fixed"
        )
        ctx.minimizer._relax_leaflet_tilts(positions=mesh.positions_view(), mode=mode)

    rows: list[dict[str, Any]] = []

    def add_position_probe(label: str, direction: np.ndarray) -> None:
        if not np.any(direction):
            return

        def perturb(scale: float) -> None:
            _set_positions_from_array(ctx, positions + scale * direction)

        rows.append(
            {
                "label": label,
                "kind": "position",
                "raw_fd_slope": _finite_difference_total_slope(ctx, perturb),
                "enforced_fd_slope": _finite_difference_total_slope(
                    ctx, perturb, enforce
                ),
                "enforced_relaxed_fd_slope": _finite_difference_total_slope(
                    ctx, perturb, enforce_relax
                ),
                "raw_gradient_dot_direction": float(np.sum(raw_grad * direction)),
                "projected_gradient_dot_direction": float(
                    np.sum(projected_grad * direction)
                ),
            }
        )

    def add_tilt_probe(label: str, direction: np.ndarray) -> None:
        tin0 = mesh.tilts_in_view().copy(order="F")
        if not np.any(direction):
            return

        def perturb(scale: float) -> None:
            mesh.set_tilts_in_from_array(tin0 + scale * direction)
            mesh.increment_version()

        rows.append(
            {
                "label": label,
                "kind": "tilt_in",
                "raw_fd_slope": _finite_difference_total_slope(ctx, perturb),
                "enforced_fd_slope": _finite_difference_total_slope(
                    ctx, perturb, enforce
                ),
                "enforced_relaxed_fd_slope": _finite_difference_total_slope(
                    ctx, perturb, enforce_relax
                ),
                "raw_gradient_dot_direction": 0.0,
                "projected_gradient_dot_direction": 0.0,
            }
        )

    direction_in, _ = _boundary_tilt_direction(ctx)
    add_tilt_probe("boundary_radial_tilt_in", direction_in)
    add_position_probe(
        "trace_shell_height", _position_direction_for_role(ctx, "trace_shell", "z")
    )
    add_position_probe(
        "trace_shell_radius", _position_direction_for_role(ctx, "trace_shell", "radial")
    )
    add_position_probe(
        "disk_rim_height", _position_direction_for_role(ctx, "disk", "z")
    )
    add_position_probe(
        "disk_rim_radius", _position_direction_for_role(ctx, "disk", "radial")
    )
    add_position_probe(
        "first_support_shell_height",
        _position_direction_for_role(ctx, "support_shell_1", "z"),
    )
    add_position_probe(
        "first_support_shell_radius",
        _position_direction_for_role(ctx, "support_shell_1", "radial"),
    )
    add_position_probe(
        "release_ring_height", _position_direction_for_role(ctx, "release_ring", "z")
    )
    add_position_probe(
        "release_ring_radius",
        _position_direction_for_role(ctx, "release_ring", "radial"),
    )

    residual = matching_residual_diagnostics(
        mesh, mesh.global_parameters, mesh.positions_view()
    )
    return {
        "base_energy": float(energy),
        "base_outer_residual_mean": float(
            (residual.get("outer_residual") or {}).get("mean") or 0.0
        ),
        "base_inner_residual_mean": float(
            (residual.get("inner_residual") or {}).get("mean") or 0.0
        ),
        "probes": rows,
    }


def _refinement_trace(
    mesh_path: Path, protocol: tuple[str, ...]
) -> list[dict[str, Any]]:
    ctx = _build_context(mesh_path)
    rows = [{"label": "initial", **_mesh_topology_audit(ctx.mesh)}]
    for idx, cmd in enumerate(protocol):
        execute_command_line(ctx, cmd)
        if cmd == "r" or cmd.startswith("V") or cmd.startswith("g"):
            rows.append({"label": f"{idx}:{cmd}", **_mesh_topology_audit(ctx.mesh)})
    return rows


def _protocol_snapshot_audit(
    mesh_path: Path, protocol: tuple[str, ...]
) -> list[dict[str, Any]]:
    ctx = _build_context(mesh_path)
    rows: list[dict[str, Any]] = []

    def snapshot(label: str, command: str | None = None) -> dict[str, Any]:
        row = _checkpoint(ctx, label)
        row["command"] = command or ""
        try:
            report = _collect_report_from_context(
                ctx=ctx, mesh_path=mesh_path, protocol=tuple()
            )
            diagnostics = report["metrics"].get("diagnostics", {})
            tex = report["metrics"].get("tex_benchmark", {})
            row.update(
                {
                    "thetaB_report_value": float(
                        report["metrics"].get("thetaB_value") or 0.0
                    ),
                    "tex_total_ratio": float(
                        (tex.get("ratios") or {}).get("total_ratio") or 0.0
                    ),
                    "outer_split": diagnostics.get("outer_split", {}),
                    "interface_traces_at_R": diagnostics.get(
                        "interface_traces_at_R", {}
                    ),
                    "interface_directors": diagnostics.get("interface_directors", {}),
                    "outer_profile_parity": diagnostics.get("outer_profile_parity", {}),
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostics path
            row["report_error"] = str(exc)
        return row

    rows.append(snapshot("initial"))
    for idx, cmd in enumerate(protocol):
        execute_command_line(ctx, cmd)
        if cmd == "energy" or cmd == "r" or cmd.startswith("V") or cmd.startswith("g"):
            rows.append(snapshot(f"{idx}:{cmd}", cmd))
    return rows


def _write_temp_fixture(doc: dict[str, Any], directory: Path, label: str) -> Path:
    path = directory / f"{label}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def _apply_scaffold_runtime_options(doc: dict[str, Any], label: str) -> dict[str, Any]:
    doc = copy.deepcopy(doc)
    gp = dict(doc.get("global_parameters") or {})
    gp["theory_parity_lane"] = label
    gp["rim_slope_match_mode"] = "physical_edge_staggered_v1"
    gp["rim_slope_match_scaffold_projector_mode"] = "continuity_v2"
    gp["tilt_thetaB_contact_work_mode"] = "field_linear"
    gp["tilt_solver"] = "cg"
    gp["tilt_cg_max_iters"] = 120
    gp["tilt_mass_mode_in"] = "consistent"
    doc["global_parameters"] = gp
    constraints = [str(x) for x in (doc.get("constraint_modules") or [])]
    doc["constraint_modules"] = [
        x for x in constraints if x != "tilt_thetaB_boundary_in"
    ]
    return doc


def _variant_report_summary(path: Path, protocol: tuple[str, ...]) -> dict[str, Any]:
    ctx = _build_context(path)
    _run_protocol_with_parity_activation(ctx, protocol=protocol)
    report = _collect_report_from_context(ctx=ctx, mesh_path=path, protocol=protocol)
    driver = report["metrics"]["diagnostics"].get("scaffold_boundary_driver", {})
    ratios = report["metrics"]["theory"]["ratios"]
    return {
        "thetaB_value": float(report["metrics"]["thetaB_value"]),
        "final_energy": float(report["metrics"]["final_energy"]),
        "stationarity_residual": float(driver.get("stationarity_residual", 0.0)),
        "theta_ratio": float(ratios.get("theta_ratio", 0.0)),
        "elastic_ratio": float(ratios.get("elastic_ratio", 0.0)),
        "total_ratio": float(ratios.get("total_ratio", 0.0)),
    }


def _resolution_matrix(mode: str, protocol: tuple[str, ...]) -> dict[str, Any]:
    if mode == "none":
        return {"mode": "none", "variants": []}

    base_doc = yaml.safe_load(PHYSICAL_EDGE_DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    eps_values = [0.005] if mode == "quick" else [0.0025, 0.005, 0.01]
    shells_values = [1, 3] if mode == "quick" else [1, 2, 3, 4]
    variant_protocol = protocol if mode == "full" else QUICK_PROTOCOL
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="scaffold_audit_") as tmp:
        tmpdir = Path(tmp)
        for eps in eps_values:
            for shells in shells_values:
                label = f"gapfill_eps{eps:g}_n{shells}"
                doc = build_gap_filled_outer_shell_scaffold_fixture(
                    base_doc=base_doc,
                    label=label,
                    trace_radius=(7.0 / 15.0) + float(eps),
                    outer_shells=int(shells),
                    planar_geometry=False,
                )
                doc = _apply_scaffold_runtime_options(doc, label)
                path = _write_temp_fixture(doc, tmpdir, label)
                try:
                    summary = _variant_report_summary(path, variant_protocol)
                    summary.update(
                        {
                            "label": label,
                            "epsilon": float(eps),
                            "outer_shells": int(shells),
                            "mode": "gap_filled_release",
                        }
                    )
                except Exception as exc:  # pragma: no cover - diagnostics path
                    summary = {
                        "label": label,
                        "epsilon": float(eps),
                        "outer_shells": int(shells),
                        "mode": "gap_filled_release",
                        "error": str(exc),
                    }
                rows.append(summary)

        if mode != "quick":
            label = "fixed_d_eps0.005_n3"
            doc = build_outer_shell_scaffold_fixture(
                base_doc=base_doc,
                label=label,
                trace_radius=(7.0 / 15.0) + 0.005,
                outer_shells=3,
                outer_shells_d=0.05,
                planar_geometry=False,
            )
            doc = _apply_scaffold_runtime_options(doc, label)
            path = _write_temp_fixture(doc, tmpdir, label)
            summary = _variant_report_summary(path, variant_protocol)
            summary.update(
                {
                    "label": label,
                    "epsilon": 0.005,
                    "outer_shells": 3,
                    "mode": "fixed_d",
                }
            )
            rows.append(summary)

    finite = [
        row
        for row in rows
        if "error" not in row
        and np.isfinite(_as_float(row.get("stationarity_residual")))
    ]
    monotonic_total_ratio_by_shell = False
    if len(finite) >= 2 and len({row.get("epsilon") for row in finite}) == 1:
        sorted_rows = sorted(finite, key=lambda row: int(row["outer_shells"]))
        vals = [abs(float(row["total_ratio"]) - 1.0) for row in sorted_rows]
        monotonic_total_ratio_by_shell = all(
            vals[i + 1] <= vals[i] + 1.0e-12 for i in range(len(vals) - 1)
        )
    return {
        "mode": mode,
        "protocol": list(variant_protocol),
        "variants": rows,
        "monotonic_total_ratio_improvement_by_shell": bool(
            monotonic_total_ratio_by_shell
        ),
    }


def _advanced_flags(ctx, report: dict[str, Any]) -> dict[str, Any]:
    gp = ctx.mesh.global_parameters
    driver = report["metrics"]["diagnostics"].get("scaffold_boundary_driver", {})
    theory = report["metrics"]["theory"]
    tex = report["metrics"]["tex_benchmark"]
    return {
        "line_search_reduced_energy": bool(gp.get("line_search_reduced_energy")),
        "tilt_thetaB_optimize": bool(gp.get("tilt_thetaB_optimize")),
        "contact_work_mode": str(gp.get("tilt_thetaB_contact_work_mode") or "scalar"),
        "projector_mode": str(gp.get("rim_slope_match_scaffold_projector_mode") or ""),
        "legacy_theta_star": float(theory.get("thetaB_star", 0.0)),
        "tex_theta_star": float(tex.get("thetaB_star", 0.0)),
        "theta_star_legacy_over_tex": _safe_ratio(
            float(theory.get("thetaB_star", 0.0)), float(tex.get("thetaB_star", 0.0))
        ),
        "stationarity_residual": float(driver.get("stationarity_residual", 0.0)),
    }


def _benchmark_terms(
    *,
    theta_meas: float,
    elastic_meas: float,
    contact_meas: float,
    total_meas: float,
    drive: float,
    kappa_value: float,
    kappa_t_value: float,
    radius_value: float,
) -> dict[str, Any]:
    theta_star = 0.0
    elastic_star = 0.0
    contact_star = 0.0
    total_star = 0.0
    if (
        kappa_value > 0.0
        and kappa_t_value > 0.0
        and drive != 0.0
        and radius_value > 0.0
    ):
        lam = float(np.sqrt(kappa_t_value / kappa_value))
        x = float(lam * radius_value)
        ratio_i = float(special.iv(0, x) / special.iv(1, x))
        ratio_k = float(special.kv(0, x) / special.kv(1, x))
        den = float(ratio_i + 0.5 * ratio_k)
        theta_star = float(drive / (np.sqrt(kappa_value * kappa_t_value) * den))
        fin_star = float(
            np.pi * kappa_value * radius_value * lam * ratio_i * theta_star**2
        )
        fout_star = float(
            np.pi * kappa_value * radius_value * lam * 0.5 * ratio_k * theta_star**2
        )
        elastic_star = float(fin_star + fout_star)
        contact_star = float(-2.0 * np.pi * radius_value * drive * theta_star)
        total_star = float(elastic_star + contact_star)
    return {
        "radius": float(radius_value),
        "kappa": float(kappa_value),
        "kappa_t": float(kappa_t_value),
        "drive": float(drive),
        "thetaB_star": theta_star,
        "elastic_star": elastic_star,
        "contact_star": contact_star,
        "total_star": total_star,
        "ratios": {
            "theta_ratio": _safe_ratio(theta_meas, theta_star),
            "elastic_ratio": _safe_ratio(elastic_meas, elastic_star),
            "contact_ratio": _safe_ratio(contact_meas, contact_star),
            "total_ratio": _safe_ratio(total_meas, total_star),
        },
    }


def _energy_normalization_audit(ctx, report: dict[str, Any]) -> dict[str, Any]:
    gp = ctx.mesh.global_parameters
    breakdown = ctx.minimizer.compute_energy_breakdown()
    contact = _contact_geometry(ctx)
    theta_meas = float(report["metrics"].get("thetaB_value") or 0.0)
    total_meas = float(ctx.minimizer.compute_energy())
    contact_meas = float(breakdown.get("tilt_thetaB_contact_in") or 0.0)
    elastic_in = float(
        (breakdown.get("tilt_in") or 0.0) + (breakdown.get("bending_tilt_in") or 0.0)
    )
    elastic_out = float(
        (breakdown.get("tilt_out") or 0.0) + (breakdown.get("bending_tilt_out") or 0.0)
    )
    elastic_meas = float(elastic_in + elastic_out)
    drive = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)
    kappa_in = float(gp.get("bending_modulus_in") or gp.get("bending_modulus") or 0.0)
    kappa_out = float(gp.get("bending_modulus_out") or gp.get("bending_modulus") or 0.0)
    kappa_t_in = float(gp.get("tilt_modulus_in") or 0.0)
    kappa_t_out = float(gp.get("tilt_modulus_out") or 0.0)
    r_theory = float(contact["R_theory"])
    r_eff = float(contact["R_eff"])
    contact_formula_eff = float(
        -2.0 * np.pi * r_eff * drive * float(contact["theta_contact_mean"])
    )
    contact_formula_theory = float(
        -2.0 * np.pi * r_theory * drive * float(contact["theta_contact_mean"])
    )
    matrix = {
        "legacy_anchor": _benchmark_terms(
            theta_meas=theta_meas,
            elastic_meas=elastic_meas,
            contact_meas=contact_meas,
            total_meas=total_meas,
            drive=drive,
            kappa_value=kappa_in + kappa_out,
            kappa_t_value=kappa_t_in + kappa_t_out,
            radius_value=r_theory,
        ),
        "tex_benchmark": _benchmark_terms(
            theta_meas=theta_meas,
            elastic_meas=elastic_meas,
            contact_meas=contact_meas,
            total_meas=total_meas,
            drive=drive,
            kappa_value=float(DEFAULT_TEX_BENDING_MODULUS),
            kappa_t_value=float(DEFAULT_TEX_TILT_MODULUS),
            radius_value=r_theory,
        ),
        "in_only_elastic": _benchmark_terms(
            theta_meas=theta_meas,
            elastic_meas=elastic_in,
            contact_meas=contact_meas,
            total_meas=elastic_in + contact_meas,
            drive=drive,
            kappa_value=kappa_in,
            kappa_t_value=kappa_t_in,
            radius_value=r_theory,
        ),
        "out_only_elastic": _benchmark_terms(
            theta_meas=theta_meas,
            elastic_meas=elastic_out,
            contact_meas=0.0,
            total_meas=elastic_out,
            drive=drive,
            kappa_value=kappa_out,
            kappa_t_value=kappa_t_out,
            radius_value=r_theory,
        ),
        "contact_R_eff": _benchmark_terms(
            theta_meas=theta_meas,
            elastic_meas=elastic_meas,
            contact_meas=contact_meas,
            total_meas=total_meas,
            drive=drive,
            kappa_value=kappa_in + kappa_out,
            kappa_t_value=kappa_t_in + kappa_t_out,
            radius_value=r_eff,
        ),
    }
    return {
        "measured": {
            "total": total_meas,
            "elastic": elastic_meas,
            "elastic_in": elastic_in,
            "elastic_out": elastic_out,
            "contact": contact_meas,
            "breakdown_sum": float(sum(float(v) for v in breakdown.values())),
            "thetaB_value": theta_meas,
        },
        "parameters": {
            "kappa_in": kappa_in,
            "kappa_out": kappa_out,
            "kappa_sum": float(kappa_in + kappa_out),
            "kappa_t_in": kappa_t_in,
            "kappa_t_out": kappa_t_out,
            "kappa_t_sum": float(kappa_t_in + kappa_t_out),
            "drive": drive,
            "R_theory": r_theory,
            "R_eff": r_eff,
        },
        "contact_geometry": contact,
        "identities": {
            "elastic_minus_active_module_sum": float(
                elastic_meas - (elastic_in + elastic_out)
            ),
            "total_minus_breakdown_sum": float(
                total_meas - sum(float(v) for v in breakdown.values())
            ),
            "contact_minus_formula_R_eff": float(contact_meas - contact_formula_eff),
            "contact_minus_formula_R_theory": float(
                contact_meas - contact_formula_theory
            ),
            "R_eff_over_R_theory": _safe_ratio(r_eff, r_theory),
        },
        "normalization_matrix": matrix,
    }


def _cadence_variants(mesh_path: Path) -> dict[str, Any]:
    base_doc = yaml.safe_load(Path(mesh_path).read_text(encoding="utf-8"))
    variants = [
        ("baseline", {}),
        ("line_search_reduced_energy_off", {"line_search_reduced_energy": False}),
        (
            "line_search_decrease_only",
            {
                "line_search_reduced_energy": True,
                "line_search_reduced_accept_rule": "decrease_only",
            },
        ),
        ("thetaB_optimize_off", {"tilt_thetaB_optimize": False}),
        ("scaffold_projector_off", {"rim_slope_match_scaffold_projector_mode": None}),
    ]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="scaffold_cadence_variants_") as tmp:
        tmpdir = Path(tmp)
        for label, overrides in variants:
            doc = copy.deepcopy(base_doc)
            gp = dict(doc.get("global_parameters") or {})
            gp.update(overrides)
            gp["theory_parity_lane"] = f"cadence_variant_{label}"
            doc["global_parameters"] = gp
            path = _write_temp_fixture(doc, tmpdir, label)
            try:
                summary = _variant_report_summary(path, QUICK_PROTOCOL)
                summary["label"] = label
                summary["overrides"] = overrides
            except Exception as exc:  # pragma: no cover - diagnostics path
                summary = {"label": label, "overrides": overrides, "error": str(exc)}
            rows.append(summary)
    return {"protocol": list(QUICK_PROTOCOL), "variants": rows}


def run_audit(
    *,
    mesh_path: Path = DEFAULT_FIXTURE,
    protocol: tuple[str, ...] = QUICK_PROTOCOL,
    resolution_mode: str = "quick",
) -> dict[str, Any]:
    ctx = _build_context(Path(mesh_path))
    before_topology = _mesh_topology_audit(ctx.mesh)
    _run_protocol_with_parity_activation(ctx, protocol=protocol)
    report = _collect_report_from_context(
        ctx=ctx, mesh_path=Path(mesh_path), protocol=protocol
    )
    after_topology = _mesh_topology_audit(ctx.mesh)

    return {
        "meta": {
            "fixture": (
                str(Path(mesh_path).resolve().relative_to(ROOT))
                if Path(mesh_path).resolve().is_relative_to(ROOT)
                else str(mesh_path)
            ),
            "protocol": list(protocol),
            "resolution_mode": str(resolution_mode),
        },
        "mesh_topology": {
            "before_protocol": before_topology,
            "after_protocol": after_topology,
        },
        "refinement_trace": _refinement_trace(Path(mesh_path), protocol),
        "module_energy_audit": _module_energy_audit(ctx),
        "interface_target_audit": _interface_target_audit(ctx),
        "constraint_audit": _constraint_audit(ctx),
        "coupled_stationarity_audit": _coupled_stationarity_audit(ctx),
        "elastic_magnitude_audit": _elastic_magnitude_audit(ctx),
        "bending_tilt_base_term_audit": _bending_tilt_base_term_audit(ctx),
        "base_term_fixture_comparison": _base_term_fixture_comparison(),
        "constrained_gradient_audit": _constrained_gradient_audit(ctx),
        "protocol_snapshot_audit": _protocol_snapshot_audit(Path(mesh_path), protocol),
        "energy_normalization_audit": _energy_normalization_audit(ctx, report),
        "bulk_boundary_split": _bulk_boundary_split(ctx),
        "resolution_matrix": _resolution_matrix(str(resolution_mode), protocol),
        "cadence_variants": _cadence_variants(Path(mesh_path)),
        "advanced_flags": _advanced_flags(ctx, report),
        "parity_summary": {
            "thetaB_value": float(report["metrics"]["thetaB_value"]),
            "final_energy": float(report["metrics"]["final_energy"]),
            "theory_ratios": report["metrics"]["theory"]["ratios"],
            "scaffold_boundary_driver": report["metrics"]["diagnostics"].get(
                "scaffold_boundary_driver", {}
            ),
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--protocol",
        nargs="*",
        default=list(QUICK_PROTOCOL),
        help="Protocol commands to run before auditing.",
    )
    parser.add_argument(
        "--protocol-mode",
        choices=("quick", "default", "long"),
        default=None,
        help="Convenience protocol selector; overridden by --protocol when set.",
    )
    parser.add_argument(
        "--resolution-mode",
        choices=("none", "quick", "full"),
        default="quick",
        help="How much temporary scaffold resolution probing to run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.protocol_mode == "default":
        protocol = tuple(DEFAULT_PROTOCOL)
    elif args.protocol_mode == "long":
        protocol = tuple(LONG_SCAFFOLD_PROTOCOL)
    elif args.protocol_mode == "quick":
        protocol = tuple(QUICK_PROTOCOL)
    else:
        protocol = tuple(str(x) for x in args.protocol)
    audit = run_audit(
        mesh_path=args.mesh,
        protocol=protocol,
        resolution_mode=str(args.resolution_mode),
    )
    text = yaml.safe_dump(audit, sort_keys=False)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
