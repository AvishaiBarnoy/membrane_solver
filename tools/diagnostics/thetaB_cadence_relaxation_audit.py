#!/usr/bin/env python3
"""Diagnostics for thetaB scan cadence and coupled tilt relaxation behavior."""

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from geometry.tilt_operators import (  # noqa: E402
    _resolve_transport_model,
    compute_divergence_from_basis,
    p1_triangle_divergence,
)
from modules.constraints.local_interface_shells import (  # noqa: E402
    build_local_interface_shell_data,
    radial_unit_vectors,
)
from modules.energy import bending_tilt_in as bending_tilt_in_module  # noqa: E402
from modules.energy import bending_tilt_out as bending_tilt_out_module  # noqa: E402
from modules.energy import tilt_out as tilt_out_module  # noqa: E402
from modules.energy.bending_utils import _compute_effective_areas  # noqa: E402
from modules.energy.leaflet_presence import (  # noqa: E402
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)
from runtime.projections.curved_disk import (  # noqa: E402
    project_curved_free_disk_shape_dofs,
)
from runtime.projections.tilt import build_leaflet_trial_tilts  # noqa: E402
from runtime.tilt_optimization import _optimize_thetaB_scalar  # noqa: E402
from tools.diagnostics.free_disk_profile_protocol import (  # noqa: E402
    _bending_tilt_leaflet_region_split,
    _tilt_in_region_split,
    _tilt_out_region_split,
)
from tools.diagnostics.parity_acceptance_triage import (  # noqa: E402
    DEFAULT_FIXTURE,
    DEFAULT_PROTOCOL,
    GHOST_FIXTURE,
    _as_float,
    _interface_summary,
    _reduced_terms_summary,
    _runtime_breakdown_summary,
)
from tools.diagnostics.scaffold_energy_imbalance_audit import (  # noqa: E402
    _base_term_summary_for_fixture,
)
from tools.diagnostics.utils import (  # noqa: E402
    radial_projection,
    row_region_mask_dict,
)
from tools.reproduce_theory_parity import (  # noqa: E402
    _build_context,
    _collect_report_from_context,
    _run_protocol_with_parity_activation,
)
from tools.theory_parity_interface_profiles import (  # noqa: E402
    build_full_physics_trace_fixture,
    build_gap_filled_outer_shell_scaffold_fixture,
    build_outer_shell_scaffold_fixture,
)

ROOT = Path(__file__).resolve().parents[2]
FULL_COUPLING_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_physical_edge_full_coupling_v1.yaml"
)
FULL_COUPLING_TRACE_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_physical_edge_full_coupling_trace_eps005_v1.yaml"
)
TRACE_CONVERGENCE_EPSILONS = (0.0025, 0.005, 0.01)
TRACE_CONVERGENCE_GEOMETRIES = (
    "no_trace_current",
    "trace_only",
    "fixed_support",
    "gapfill_release",
)
SCAFFOLD_COLLAPSE_GEOMETRIES = ("fixed_support", "gapfill_release")
SCAFFOLD_COLLAPSE_VARIANTS = (
    "plain",
    "projector_only",
    "runtime_options",
    "runtime_options_gd_fallback",
)
SCAFFOLD_SUPPORT_OWNERSHIP_VARIANTS = (
    "runtime_options",
    "runtime_options_gd_fallback",
    "support_tilt_frozen",
    "support_geometry_frozen",
    "support_passive_trace_only",
)
TRACE_CONTINUATION_LANDSCAPE_VARIANTS = (
    "trace_only",
    "fixed_support_runtime_options_gd_fallback",
    "gapfill_release_runtime_options_gd_fallback",
)
TRACE_CONTINUATION_LANDSCAPE_MODES = (
    "trace_tilt",
    "support_tilt",
    "trace_support_tilt",
    "trace_tilt_height",
)
TRACE_CONTINUATION_LANDSCAPE_ALPHAS = (0.25, 0.5, 1.0)
BENDING_TILT_OUT_INTERFACE_REFERENCE_MODES = (
    "current_geometry",
    "flat_reference_zero_J0",
)
SCAFFOLD_GEOMETRY_SPACING_VARIANTS = (
    {"label": "trace_only_eps005", "geometry": "trace_only", "epsilon": 0.005},
    {
        "label": "fixed_support_eps005_d0025",
        "geometry": "fixed_support",
        "epsilon": 0.005,
        "outer_shells": 3,
        "outer_shells_d": 0.025,
    },
    {
        "label": "fixed_support_eps005_d005",
        "geometry": "fixed_support",
        "epsilon": 0.005,
        "outer_shells": 3,
        "outer_shells_d": 0.05,
    },
    {
        "label": "fixed_support_eps005_d0075",
        "geometry": "fixed_support",
        "epsilon": 0.005,
        "outer_shells": 3,
        "outer_shells_d": 0.075,
    },
    {
        "label": "fixed_support_eps005_d009",
        "geometry": "fixed_support",
        "epsilon": 0.005,
        "outer_shells": 3,
        "outer_shells_d": 0.09,
    },
    {
        "label": "fixed_support_eps005_d009_trace_reconstructed",
        "geometry": "fixed_support",
        "epsilon": 0.005,
        "outer_shells": 3,
        "outer_shells_d": 0.09,
        "interface_divergence_mode": "trace_reconstructed_v1",
    },
    {
        "label": "fixed_support_eps005_d009_inner_trace_boundary",
        "geometry": "fixed_support",
        "epsilon": 0.005,
        "outer_shells": 3,
        "outer_shells_d": 0.09,
        "inner_scaffold_shape_stencil_mode": "trace_boundary_v1",
    },
    {
        "label": "fixed_support_eps005_d009_preserve_trace",
        "geometry": "fixed_support",
        "epsilon": 0.005,
        "outer_shells": 3,
        "outer_shells_d": 0.09,
        "inner_scaffold_shape_stencil_mode": "trace_boundary_v1",
        "scaffold_mesh_operation_mode": "preserve_trace_v1",
    },
    {
        "label": "fixed_support_eps005_d009_preserve_trace_z",
        "geometry": "fixed_support",
        "epsilon": 0.005,
        "outer_shells": 3,
        "outer_shells_d": 0.09,
        "inner_scaffold_shape_stencil_mode": "trace_boundary_v1",
        "scaffold_mesh_operation_mode": "preserve_trace_v1",
        "pin_to_circle_preserve_normal_groups": ["trace_layer"],
    },
    {
        "label": "fixed_support_eps005_d009_trace_z_fallback",
        "geometry": "fixed_support",
        "epsilon": 0.005,
        "outer_shells": 3,
        "outer_shells_d": 0.09,
        "inner_scaffold_shape_stencil_mode": "trace_boundary_v1",
        "scaffold_mesh_operation_mode": "preserve_trace_v1",
        "pin_to_circle_preserve_normal_groups": ["trace_layer"],
        "shape_scaffold_rejected_step_fallback": "trace_z",
    },
    {
        "label": "gapfill_release_eps005_n1",
        "geometry": "gapfill_release",
        "epsilon": 0.005,
        "outer_shells": 1,
    },
    {
        "label": "gapfill_release_eps005_n1_trace_reconstructed",
        "geometry": "gapfill_release",
        "epsilon": 0.005,
        "outer_shells": 1,
        "interface_divergence_mode": "trace_reconstructed_v1",
    },
    {
        "label": "gapfill_release_eps005_n1_inner_trace_boundary",
        "geometry": "gapfill_release",
        "epsilon": 0.005,
        "outer_shells": 1,
        "inner_scaffold_shape_stencil_mode": "trace_boundary_v1",
    },
    {
        "label": "gapfill_release_eps005_n1_preserve_trace",
        "geometry": "gapfill_release",
        "epsilon": 0.005,
        "outer_shells": 1,
        "inner_scaffold_shape_stencil_mode": "trace_boundary_v1",
        "scaffold_mesh_operation_mode": "preserve_trace_v1",
    },
    {
        "label": "gapfill_release_eps005_n1_preserve_trace_z",
        "geometry": "gapfill_release",
        "epsilon": 0.005,
        "outer_shells": 1,
        "inner_scaffold_shape_stencil_mode": "trace_boundary_v1",
        "scaffold_mesh_operation_mode": "preserve_trace_v1",
        "pin_to_circle_preserve_normal_groups": ["trace_layer"],
    },
    {
        "label": "gapfill_release_eps005_n1_trace_z_fallback",
        "geometry": "gapfill_release",
        "epsilon": 0.005,
        "outer_shells": 1,
        "inner_scaffold_shape_stencil_mode": "trace_boundary_v1",
        "scaffold_mesh_operation_mode": "preserve_trace_v1",
        "pin_to_circle_preserve_normal_groups": ["trace_layer"],
        "shape_scaffold_rejected_step_fallback": "trace_z",
    },
    {
        "label": "gapfill_release_eps005_n2",
        "geometry": "gapfill_release",
        "epsilon": 0.005,
        "outer_shells": 2,
    },
    {
        "label": "gapfill_release_eps005_n3",
        "geometry": "gapfill_release",
        "epsilon": 0.005,
        "outer_shells": 3,
    },
)
SCAFFOLD_GEOMETRY_SEED_LABELS = frozenset(
    {
        "trace_only_eps005",
        "fixed_support_eps005_d009",
        "fixed_support_eps005_d009_trace_reconstructed",
        "gapfill_release_eps005_n1",
        "gapfill_release_eps005_n1_trace_reconstructed",
        "fixed_support_eps005_d009_inner_trace_boundary",
        "gapfill_release_eps005_n1_inner_trace_boundary",
        "fixed_support_eps005_d009_preserve_trace",
        "gapfill_release_eps005_n1_preserve_trace",
        "fixed_support_eps005_d009_preserve_trace_z",
        "gapfill_release_eps005_n1_preserve_trace_z",
        "fixed_support_eps005_d009_trace_z_fallback",
        "gapfill_release_eps005_n1_trace_z_fallback",
    }
)
FIXED_THETA_GRID = (0.16, 0.18, 0.20, 0.21, 0.24, 0.27, 0.30)
RELAXATION_BUDGETS = (0, 1, 5, 20, 60, 120)
SCAN_DELTAS = (0.005, 0.01, 0.02)
SCAN_INNER_STEPS = (1, 5, 20, 60, 120)
MAIN_INNER_STEPS = (5, 20, 60, 120)
TRACE_THETAS = (0.18, 0.21, 0.30)
TRACE_PASSES = 12
LINE_SEARCH_REDUCED_STEPS = (0, 1, 5, 20)
TRACE_Z_FALLBACK_ALPHA_GRID = (0.002, 0.0014, 0.0007, 0.000343, 0.000172944)
TRACE_Z_FALLBACK_CONSTRAINT_CONTEXTS = (
    "none",
    "minimize",
    "mesh_operation",
    "finalize",
)
WARM_START_POLICIES = (
    "anchor_optimized",
    "fresh_fixture",
    "zero_outer",
    "previous_theta",
)
STATE_PATH_THETAS = (0.21, 0.30)
STATE_PATH_RELAXATION_BUDGETS = (5, 60)
SURVIVAL_TRACE_THETAS = (0.21, 0.30)
SURVIVAL_TRACE_PASSES = 6
SURVIVAL_TRACE_POLICIES = (
    "anchor_optimized",
    "fresh_fixture",
    "zero_outer",
)
SOLVER_PATH_THETAS = (0.21, 0.30)
SOLVER_PATH_INNER_STEPS = 20
SOLVER_PATH_VARIANTS = (
    {"label": "cg_jacobi", "solver": "cg", "preconditioner": "jacobi"},
    {"label": "cg_none", "solver": "cg", "preconditioner": "none"},
    {"label": "gd", "solver": "gd", "preconditioner": None},
)
ASSEMBLY_THETAS = (0.18, 0.21, 0.30)
RUNTIME_BRIDGE_THETAS = ASSEMBLY_THETAS
OUTER_COUPLING_SWEEP_THETAS = ASSEMBLY_THETAS
OUTER_COUPLING_SIGN_VARIANTS = (
    {
        "label": "production_sign",
        "div_term_sign": 1.0,
        "pullback_sign": 1.0,
        "reference_mode": "fixture",
    },
    {
        "label": "flipped_div_sign",
        "div_term_sign": -1.0,
        "pullback_sign": -1.0,
        "reference_mode": "fixture",
    },
    {
        "label": "flipped_pullback_only",
        "div_term_sign": 1.0,
        "pullback_sign": -1.0,
        "reference_mode": "fixture",
    },
    {
        "label": "production_sign_current_geometry",
        "div_term_sign": 1.0,
        "pullback_sign": 1.0,
        "reference_mode": "current_geometry",
    },
    {
        "label": "production_sign_flat_reference",
        "div_term_sign": 1.0,
        "pullback_sign": 1.0,
        "reference_mode": "flat_reference_zero_J0",
    },
)
BASE_TERM_REFERENCE_VARIANTS = (
    {
        "label": "global_flat_reference",
        "set": {"bending_tilt_base_term_reference_mode": "flat_reference_zero_J0"},
        "unset": (
            "bending_tilt_base_term_reference_mode_in",
            "bending_tilt_base_term_reference_mode_out",
        ),
    },
    {
        "label": "inner_flat_outer_current",
        "set": {
            "bending_tilt_base_term_reference_mode": "current_geometry",
            "bending_tilt_base_term_reference_mode_in": "flat_reference_zero_J0",
            "bending_tilt_base_term_reference_mode_out": "current_geometry",
        },
        "unset": (),
    },
    {
        "label": "global_current_geometry",
        "set": {"bending_tilt_base_term_reference_mode": "current_geometry"},
        "unset": (
            "bending_tilt_base_term_reference_mode_in",
            "bending_tilt_base_term_reference_mode_out",
        ),
    },
)


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "label": "ghost",
            "base_fixture": GHOST_FIXTURE,
            "overrides": {},
        },
        {
            "label": "default_current",
            "base_fixture": DEFAULT_FIXTURE,
            "overrides": {},
        },
        {
            "label": "default_no_outer_absence",
            "base_fixture": DEFAULT_FIXTURE,
            "overrides": {"leaflet_out_absent_presets": []},
        },
    ]


def _full_physics_lane_specs() -> list[dict[str, Any]]:
    return [
        {
            "label": "default_current",
            "base_fixture": DEFAULT_FIXTURE,
            "overrides": {},
        },
        {
            "label": "full_coupling_trace",
            "base_fixture": FULL_COUPLING_TRACE_FIXTURE,
            "overrides": {},
        },
        {
            "label": "full_coupling",
            "base_fixture": FULL_COUPLING_FIXTURE,
            "overrides": {},
        },
        {
            "label": "ghost",
            "base_fixture": GHOST_FIXTURE,
            "overrides": {},
        },
    ]


def _write_temp_fixture(doc: dict[str, Any], directory: Path, label: str) -> Path:
    path = directory / f"{label}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def _build_variant_doc(base_fixture: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    doc = yaml.safe_load(base_fixture.read_text(encoding="utf-8")) or {}
    gp = dict(doc.get("global_parameters") or {})
    for key, value in (overrides or {}).items():
        gp[key] = copy.deepcopy(value)
    doc["global_parameters"] = gp
    return doc


def _capture_state(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    return {
        "positions": mesh.positions_view().copy(),
        "tilts_in": mesh.tilts_in_view().copy(order="F"),
        "tilts_out": mesh.tilts_out_view().copy(order="F"),
        "global_params": dict(mesh.global_parameters.to_dict()),
    }


def _restore_state(ctx, snapshot: dict[str, Any]) -> None:
    mesh = ctx.mesh
    positions = np.asarray(snapshot["positions"], dtype=float)
    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = positions[row]
    mesh.set_tilts_in_from_array(np.asarray(snapshot["tilts_in"], dtype=float))
    mesh.set_tilts_out_from_array(np.asarray(snapshot["tilts_out"], dtype=float))
    mesh.global_parameters._params = dict(snapshot["global_params"])
    mesh.increment_version()


def _temporary_overrides(
    gp, overrides: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    present = {key: (key in gp) for key in overrides}
    old = {key: gp.get(key) for key in overrides}
    for key, value in overrides.items():
        gp.set(key, value)
    return present, old


def _restore_overrides(gp, present: dict[str, bool], old: dict[str, Any]) -> None:
    for key, was_present in present.items():
        if was_present:
            gp.set(key, old[key])
        else:
            gp.unset(key)


def _temporary_param_patch(
    gp,
    *,
    set_overrides: dict[str, Any] | None = None,
    unset_keys: tuple[str, ...] | list[str] = (),
) -> tuple[dict[str, bool], dict[str, Any]]:
    keys = list(dict.fromkeys([*(set_overrides or {}).keys(), *tuple(unset_keys)]))
    present = {key: (key in gp) for key in keys}
    old = {key: gp.get(key) for key in keys}
    for key in unset_keys:
        gp.unset(key)
    for key, value in (set_overrides or {}).items():
        gp.set(key, value)
    return present, old


def _mean_and_max(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "max": 0.0}
    return {"mean": float(np.mean(values)), "max": float(np.max(values))}


def _region_l2_summary(
    values: np.ndarray,
    region_masks: dict[str, np.ndarray],
    *,
    shell_rows: np.ndarray | None = None,
) -> dict[str, float]:
    out = {
        region: float(np.linalg.norm(values[mask])) if np.any(mask) else 0.0
        for region, mask in region_masks.items()
    }
    if shell_rows is not None:
        out["outer_shell"] = (
            float(np.linalg.norm(values[np.asarray(shell_rows, dtype=int)]))
            if len(shell_rows)
            else 0.0
        )
    return out


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_flat = np.asarray(lhs, dtype=float).reshape(-1)
    rhs_flat = np.asarray(rhs, dtype=float).reshape(-1)
    lhs_norm = float(np.linalg.norm(lhs_flat))
    rhs_norm = float(np.linalg.norm(rhs_flat))
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return 0.0
    return float(np.dot(lhs_flat, rhs_flat) / (lhs_norm * rhs_norm))


def _shell_vector_summary(
    mesh, vectors: np.ndarray, shell_rows: np.ndarray
) -> dict[str, float]:
    shell_rows = np.asarray(shell_rows, dtype=int)
    if shell_rows.size == 0:
        return {
            "norm": 0.0,
            "radial_norm": 0.0,
            "tangential_norm": 0.0,
            "radial_mean": 0.0,
            "radial_abs_mean": 0.0,
            "z_norm": 0.0,
        }
    positions = mesh.positions_view()
    _radii, r_hat = radial_unit_vectors(positions)
    shell_vec = np.asarray(vectors, dtype=float)[shell_rows]
    shell_r_hat = r_hat[shell_rows]
    radial_scalar = np.einsum("ij,ij->i", shell_vec, shell_r_hat)
    radial_vec = radial_scalar[:, None] * shell_r_hat
    tangential_vec = shell_vec - radial_vec
    return {
        "norm": float(np.linalg.norm(shell_vec)),
        "radial_norm": float(np.linalg.norm(radial_vec)),
        "tangential_norm": float(np.linalg.norm(tangential_vec)),
        "radial_mean": float(np.mean(radial_scalar)),
        "radial_abs_mean": float(np.mean(np.abs(radial_scalar))),
        "z_norm": float(np.linalg.norm(shell_vec[:, 2])),
    }


def _accumulate_tri_gradient(
    *,
    n_vertices: int,
    tri_rows: np.ndarray,
    factor: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
) -> np.ndarray:
    grad = np.zeros((n_vertices, 3), dtype=float)
    np.add.at(grad, tri_rows[:, 0], factor[:, None] * g0)
    np.add.at(grad, tri_rows[:, 1], factor[:, None] * g1)
    np.add.at(grad, tri_rows[:, 2], factor[:, None] * g2)
    return grad


def _field_stats_by_region(mesh) -> dict[str, Any]:
    masks = row_region_mask_dict(mesh)
    tin = np.asarray(mesh.tilts_in_view(), dtype=float)
    tout = np.asarray(mesh.tilts_out_view(), dtype=float)
    tin_norm = np.linalg.norm(tin, axis=1)
    tout_norm = np.linalg.norm(tout, axis=1)
    tin_rad = radial_projection(mesh, tin)
    tout_rad = radial_projection(mesh, tout)
    out: dict[str, Any] = {}
    for region, mask in masks.items():
        out[region] = {
            "count": int(np.sum(mask)),
            "tilt_in_norm": _mean_and_max(tin_norm[mask]),
            "tilt_out_norm": _mean_and_max(tout_norm[mask]),
            "tilt_in_radial": _mean_and_max(np.abs(tin_rad[mask])),
            "tilt_out_radial": _mean_and_max(np.abs(tout_rad[mask])),
        }
    return out


def _region_energy_splits(mesh) -> dict[str, Any]:
    return {
        "tilt_in": _tilt_in_region_split(mesh),
        "tilt_out": _tilt_out_region_split(mesh),
        "bending_tilt_in": _bending_tilt_leaflet_region_split(mesh, leaflet="in"),
        "bending_tilt_out": _bending_tilt_leaflet_region_split(mesh, leaflet="out"),
    }


def _gradient_norm(ctx) -> float:
    _energy, grad_arr = ctx.minimizer.compute_energy_and_gradient_array()
    return float(np.linalg.norm(grad_arr))


def _collect_live_summary(
    *,
    ctx,
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> dict[str, Any]:
    report = _collect_report_from_context(
        ctx=ctx,
        mesh_path=mesh_path,
        protocol=protocol,
    )
    breakdown = _runtime_breakdown_summary(report)
    reduced = _reduced_terms_summary(report)
    elastic = float(
        breakdown.get("tilt_in", 0.0)
        + breakdown.get("tilt_out", 0.0)
        + breakdown.get("bending_tilt_in", 0.0)
        + breakdown.get("bending_tilt_out", 0.0)
    )
    return {
        "thetaB_value": _as_float(report.get("metrics", {}).get("thetaB_value")),
        "report_metadata": {
            "model_intent": str(report.get("metrics", {}).get("model_intent") or ""),
            "reference_mode": str(
                report.get("metrics", {}).get("reference_mode") or ""
            ),
        },
        "tex_total_ratio": _as_float(
            report.get("metrics", {})
            .get("tex_benchmark", {})
            .get("ratios", {})
            .get("total_ratio")
        ),
        "tex_ratio_summary": {
            "theta_ratio": _as_float(
                report.get("metrics", {})
                .get("tex_benchmark", {})
                .get("ratios", {})
                .get("theta_ratio")
            ),
            "elastic_ratio": _as_float(
                report.get("metrics", {})
                .get("tex_benchmark", {})
                .get("ratios", {})
                .get("elastic_ratio")
            ),
            "contact_ratio": _as_float(
                report.get("metrics", {})
                .get("tex_benchmark", {})
                .get("ratios", {})
                .get("contact_ratio")
            ),
            "total_ratio": _as_float(
                report.get("metrics", {})
                .get("tex_benchmark", {})
                .get("ratios", {})
                .get("total_ratio")
            ),
        },
        "energy_breakdown": breakdown,
        "reduced_terms": reduced,
        "elastic_total_from_breakdown": elastic,
        "interface_summary": _interface_summary(report),
        "outer_shell_geometry": dict(
            report.get("metrics", {})
            .get("diagnostics", {})
            .get("outer_shell_geometry", {})
        ),
        "scaffold_boundary_driver": dict(
            report.get("metrics", {})
            .get("diagnostics", {})
            .get("scaffold_boundary_driver", {})
        ),
        "thetaB_scan_summary": {
            "scan_count": len(
                report.get("metrics", {})
                .get("diagnostics", {})
                .get("thetaB_scan_trace", [])
                or []
            )
        },
        "field_stats_by_region": _field_stats_by_region(ctx.mesh),
        "region_energy_splits": _region_energy_splits(ctx.mesh),
        "gradient_norm": _gradient_norm(ctx),
        "projection_stats": dict(
            getattr(ctx.minimizer, "_last_tilt_projection_stats", {})
        ),
        "inner_update_mode_stats": dict(
            getattr(ctx.minimizer, "_last_inner_coupled_update_mode_stats", {})
        ),
        "leaflet_relaxation_stats": dict(
            getattr(ctx.minimizer, "_last_leaflet_relaxation_stats", {})
        ),
        "shape_scaffold_rejected_step_fallback_stats": dict(
            getattr(
                ctx.minimizer, "_last_shape_scaffold_rejected_step_fallback_stats", {}
            )
        ),
    }


def _run_protocol_summary(
    *,
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> tuple[Any, dict[str, Any]]:
    ctx = _build_context(mesh_path)
    _run_protocol_with_parity_activation(ctx, protocol=protocol)
    return ctx, _collect_live_summary(ctx=ctx, mesh_path=mesh_path, protocol=protocol)


def _relax_leaflets_for_steps(ctx, steps: int) -> None:
    if steps <= 0:
        return
    gp = ctx.mesh.global_parameters
    overrides = {
        "tilt_inner_steps": int(steps),
        "tilt_coupled_steps": int(steps),
        "tilt_cg_max_iters": int(steps),
    }
    present, old = _temporary_overrides(gp, overrides)
    try:
        ctx.minimizer._relax_leaflet_tilts(
            positions=ctx.mesh.positions_view(),
            mode=str(gp.get("tilt_solve_mode", "coupled") or "coupled"),
        )
    finally:
        _restore_overrides(gp, present, old)


def _one_step_shell_update_summary(
    *,
    ctx,
    theta: float,
) -> dict[str, float]:
    shell_rows = _outer_shell_rows(ctx.mesh)
    before = _capture_state(ctx)
    gp = ctx.mesh.global_parameters
    overrides = {
        "tilt_thetaB_optimize": False,
        "tilt_thetaB_value": float(theta),
        "tilt_inner_steps": 1,
        "tilt_coupled_steps": 1,
        "tilt_cg_max_iters": 1,
    }
    present, old = _temporary_overrides(gp, overrides)
    try:
        ctx.minimizer._relax_leaflet_tilts(
            positions=ctx.mesh.positions_view(),
            mode=str(gp.get("tilt_solve_mode", "coupled") or "coupled"),
        )
        after = _capture_state(ctx)
    finally:
        _restore_overrides(gp, present, old)
        _restore_state(ctx, before)
    delta_out = np.asarray(after["tilts_out"], dtype=float) - np.asarray(
        before["tilts_out"], dtype=float
    )
    return _shell_vector_summary(ctx.mesh, delta_out, shell_rows)


def _fixed_theta_row_classification(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "unknown"
    elastic = np.asarray(
        [_as_float(row.get("elastic_total_from_breakdown")) for row in rows],
        dtype=float,
    )
    outer = np.asarray(
        [
            _as_float(row.get("energy_breakdown", {}).get("tilt_out"))
            + _as_float(row.get("energy_breakdown", {}).get("bending_tilt_out"))
            for row in rows
        ],
        dtype=float,
    )
    if elastic.size >= 2 and elastic[-1] > elastic[0] * 1.10:
        return "under_relaxed"
    peak_idx = int(np.argmax(outer))
    if peak_idx < (len(outer) - 1) and outer[-1] < outer[peak_idx] * 0.75:
        return "outer_canceled_by_inner"
    diffs = np.diff(elastic)
    if np.any(diffs > 0.0) and np.any(diffs < 0.0):
        return "nonmonotone"
    return "stable"


def _local_vs_wide_classification(
    *,
    local_best_theta: float,
    wide_best_theta: float,
    local_best_energy: float,
    wide_best_energy: float,
) -> str:
    if (
        wide_best_energy < (local_best_energy - 1.0e-8)
        and abs(wide_best_theta - local_best_theta) > 1.0e-9
    ):
        return "local_thetaB_scan_trap"
    if abs(wide_best_theta - local_best_theta) > 1.0e-9:
        return "wide_grid_disagreement"
    return "wide_grid_agreement"


def _collect_candidate_row(
    *,
    ctx,
    snapshot: dict[str, Any],
    mesh_path: Path,
    protocol: tuple[str, ...],
    theta: float,
    relax_steps: int,
) -> dict[str, Any]:
    _restore_state(ctx, snapshot)
    gp = ctx.mesh.global_parameters
    gp.set("tilt_thetaB_optimize", False)
    gp.set("tilt_thetaB_value", float(theta))
    _relax_leaflets_for_steps(ctx, relax_steps)
    row = _collect_live_summary(ctx=ctx, mesh_path=mesh_path, protocol=protocol)
    row["requested_thetaB_value"] = float(theta)
    row["relax_steps"] = int(relax_steps)
    row["outer_shell_field"] = _outer_shell_field_snapshot(ctx.mesh)
    row["outer_participation"] = _outer_participation_snapshot(ctx.mesh, ctx=ctx)
    row["leaflet_relaxation_stats"] = _relaxation_stats_snapshot(ctx)
    return row


def _optimized_trace_replay(
    *, tmpdir: Path, protocol: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in _variant_specs():
        doc = _build_variant_doc(spec["base_fixture"], spec["overrides"])
        mesh_path = _write_temp_fixture(doc, tmpdir, spec["label"])
        ctx, summary = _run_protocol_summary(mesh_path=mesh_path, protocol=protocol)
        trace = getattr(ctx.mesh, "_thetaB_scan_trace", [])
        summary.update(
            {
                "label": spec["label"],
                "thetaB_scan_count": int(len(trace)),
                "thetaB_scan_tail": trace[-5:] if isinstance(trace, list) else [],
            }
        )
        rows.append(summary)
    return rows


def _build_default_anchor(
    *, tmpdir: Path, protocol: tuple[str, ...]
) -> tuple[Path, Any, dict[str, Any], dict[str, Any]]:
    doc = _build_variant_doc(DEFAULT_FIXTURE, {})
    mesh_path = _write_temp_fixture(doc, tmpdir, "default_current_anchor")
    ctx, summary = _run_protocol_summary(mesh_path=mesh_path, protocol=protocol)
    snapshot = _capture_state(ctx)
    return mesh_path, ctx, snapshot, summary


def _build_default_fresh_snapshot(
    *, mesh_path: Path, anchor_snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Return a same-topology fresh tilt state for warm-start comparisons.

    The protocol may refine or rebuild the mesh before the optimized anchor is
    captured, so a raw pre-protocol fixture snapshot is not shape-compatible.
    For the cadence audit we compare against the same post-protocol geometry
    with leaflet tilts reinitialized, which isolates state-history effects
    without introducing topology mismatches.
    """
    fresh_snapshot = {
        "positions": np.asarray(anchor_snapshot["positions"], dtype=float).copy(),
        "tilts_in": np.zeros_like(np.asarray(anchor_snapshot["tilts_in"], dtype=float)),
        "tilts_out": np.zeros_like(
            np.asarray(anchor_snapshot["tilts_out"], dtype=float)
        ),
        "global_params": dict(anchor_snapshot["global_params"]),
    }
    # Preserve the current thetaB settings but disable optimizer feedback while
    # the audit drives explicit fixed-theta relaxations.
    fresh_snapshot["global_params"]["tilt_thetaB_optimize"] = False
    return fresh_snapshot


def _warm_start_policy_specs() -> list[dict[str, str]]:
    return [{"label": str(label)} for label in WARM_START_POLICIES]


def _apply_warm_start_policy(
    *,
    ctx,
    policy: str,
    anchor_snapshot: dict[str, Any],
    fresh_snapshot: dict[str, Any],
    previous_snapshot: dict[str, Any] | None = None,
) -> None:
    if policy == "anchor_optimized":
        _restore_state(ctx, anchor_snapshot)
        return
    if policy == "fresh_fixture":
        _restore_state(ctx, fresh_snapshot)
        return
    if policy == "zero_outer":
        _restore_state(ctx, anchor_snapshot)
        ctx.mesh.set_tilts_out_from_array(np.zeros_like(ctx.mesh.tilts_out_view()))
        ctx.mesh.increment_version()
        return
    if policy == "previous_theta":
        _restore_state(ctx, previous_snapshot or anchor_snapshot)
        return
    raise ValueError(f"unknown warm-start policy: {policy}")


def _outer_shell_rows(mesh) -> np.ndarray:
    try:
        shell_data = build_local_interface_shell_data(
            mesh, positions=mesh.positions_view()
        )
    except AssertionError:
        return np.asarray([], dtype=int)
    return np.asarray(shell_data.outer_rows, dtype=int)


def _outer_shell_field_snapshot(mesh) -> dict[str, Any]:
    rows = _outer_shell_rows(mesh)
    if rows.size == 0:
        return {
            "count": 0,
            "tilt_in_norm_mean": 0.0,
            "tilt_out_norm_mean": 0.0,
            "tilt_in_radial_mean": 0.0,
            "tilt_out_radial_mean": 0.0,
        }
    tin = np.asarray(mesh.tilts_in_view(), dtype=float)
    tout = np.asarray(mesh.tilts_out_view(), dtype=float)
    tin_norm = np.linalg.norm(tin[rows], axis=1)
    tout_norm = np.linalg.norm(tout[rows], axis=1)
    tin_rad = np.abs(radial_projection(mesh, tin)[rows])
    tout_rad = np.abs(radial_projection(mesh, tout)[rows])
    return {
        "count": int(rows.size),
        "tilt_in_norm_mean": float(np.mean(tin_norm)),
        "tilt_out_norm_mean": float(np.mean(tout_norm)),
        "tilt_in_radial_mean": float(np.mean(tin_rad)),
        "tilt_out_radial_mean": float(np.mean(tout_rad)),
    }


def _outer_participation_snapshot(mesh, *, ctx=None) -> dict[str, Any]:
    tri_rows, _ = mesh.triangle_row_cache()
    tri_rows = (
        np.asarray(tri_rows, dtype=np.int32)
        if tri_rows is not None and len(tri_rows) > 0
        else np.zeros((0, 3), dtype=np.int32)
    )
    absent = leaflet_absent_vertex_mask(mesh, mesh.global_parameters, leaflet="out")
    tri_keep = (
        leaflet_present_triangle_mask(mesh, tri_rows, absent_vertex_mask=absent)
        if tri_rows.size
        else np.asarray([], dtype=bool)
    )
    if ctx is not None:
        fixed_mask_out = np.asarray(ctx.minimizer._tilt_fixed_mask_out(), dtype=bool)
    else:
        fixed_mask_out = np.zeros(len(mesh.vertex_ids), dtype=bool)
    shell_rows = _outer_shell_rows(mesh)
    shell_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    if shell_rows.size:
        shell_mask[shell_rows] = True
    return {
        "outer_absent_vertex_count": int(np.sum(absent)),
        "kept_triangle_count": int(np.sum(tri_keep))
        if tri_keep.size
        else int(len(tri_rows)),
        "outer_shell_row_count": int(shell_rows.size),
        "outer_shell_absent_count": int(np.sum(absent[shell_rows]))
        if shell_rows.size
        else 0,
        "outer_shell_fixed_count": int(np.sum(fixed_mask_out[shell_rows]))
        if shell_rows.size
        else 0,
        "outer_shell_free_count": int(np.sum((~fixed_mask_out) & shell_mask)),
    }


def _relaxation_stats_snapshot(ctx) -> dict[str, Any]:
    return dict(getattr(ctx.minimizer, "_last_leaflet_relaxation_stats", {}))


def _state_path_row(
    *,
    ctx,
    mesh_path: Path,
    protocol: tuple[str, ...],
    theta: float,
    relax_steps: int,
    warm_start_policy: str,
    pass_index: int | None = None,
) -> dict[str, Any]:
    row = _collect_live_summary(ctx=ctx, mesh_path=mesh_path, protocol=protocol)
    row["requested_thetaB_value"] = float(theta)
    row["relax_steps"] = int(relax_steps)
    row["warm_start_policy"] = str(warm_start_policy)
    if pass_index is not None:
        row["pass_index"] = int(pass_index)
    row["outer_shell_field"] = _outer_shell_field_snapshot(ctx.mesh)
    row["outer_participation"] = _outer_participation_snapshot(ctx.mesh, ctx=ctx)
    row["leaflet_relaxation_stats"] = _relaxation_stats_snapshot(ctx)
    return row


def _state_path_comparison_matrix(
    *,
    ctx,
    anchor_snapshot: dict[str, Any],
    fresh_snapshot: dict[str, Any],
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for policy_row in _warm_start_policy_specs():
        policy = str(policy_row["label"])
        previous_snapshot: dict[str, Any] | None = None
        for theta in STATE_PATH_THETAS:
            for steps in STATE_PATH_RELAXATION_BUDGETS:
                _apply_warm_start_policy(
                    ctx=ctx,
                    policy=policy,
                    anchor_snapshot=anchor_snapshot,
                    fresh_snapshot=fresh_snapshot,
                    previous_snapshot=previous_snapshot,
                )
                ctx.mesh.global_parameters.set("tilt_thetaB_optimize", False)
                ctx.mesh.global_parameters.set("tilt_thetaB_value", float(theta))
                _relax_leaflets_for_steps(ctx, int(steps))
                row = _state_path_row(
                    ctx=ctx,
                    mesh_path=mesh_path,
                    protocol=protocol,
                    theta=float(theta),
                    relax_steps=int(steps),
                    warm_start_policy=policy,
                )
                rows.append(row)
                previous_snapshot = _capture_state(ctx)
    return {"rows": rows}


def _single_pass_outer_survival_trace(
    *,
    ctx,
    anchor_snapshot: dict[str, Any],
    fresh_snapshot: dict[str, Any],
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in SURVIVAL_TRACE_POLICIES:
        for theta in SURVIVAL_TRACE_THETAS:
            _apply_warm_start_policy(
                ctx=ctx,
                policy=str(policy),
                anchor_snapshot=anchor_snapshot,
                fresh_snapshot=fresh_snapshot,
            )
            ctx.mesh.global_parameters.set("tilt_thetaB_optimize", False)
            ctx.mesh.global_parameters.set("tilt_thetaB_value", float(theta))
            rows.append(
                _state_path_row(
                    ctx=ctx,
                    mesh_path=mesh_path,
                    protocol=protocol,
                    theta=float(theta),
                    relax_steps=0,
                    warm_start_policy=str(policy),
                    pass_index=0,
                )
            )
            for pass_index in range(1, SURVIVAL_TRACE_PASSES + 1):
                _relax_leaflets_for_steps(ctx, 1)
                rows.append(
                    _state_path_row(
                        ctx=ctx,
                        mesh_path=mesh_path,
                        protocol=protocol,
                        theta=float(theta),
                        relax_steps=1,
                        warm_start_policy=str(policy),
                        pass_index=pass_index,
                    )
                )
    return rows


def _thetaB_candidate_state_delta(
    *,
    ctx,
    anchor_snapshot: dict[str, Any],
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> list[dict[str, Any]]:
    _restore_state(ctx, anchor_snapshot)
    base_theta = float(anchor_snapshot["global_params"].get("tilt_thetaB_value") or 0.0)
    scan_delta = float(
        ctx.mesh.global_parameters.get("tilt_thetaB_optimize_delta", SCAN_DELTAS[0])
        or SCAN_DELTAS[0]
    )
    scan_steps = int(
        ctx.mesh.global_parameters.get(
            "tilt_thetaB_optimize_inner_steps", SCAN_INNER_STEPS[0]
        )
        or SCAN_INNER_STEPS[0]
    )
    rows: list[dict[str, Any]] = []
    base_outer = _outer_shell_field_snapshot(ctx.mesh)
    for theta in (base_theta - scan_delta, base_theta, base_theta + scan_delta):
        row = _collect_candidate_row(
            ctx=ctx,
            snapshot=anchor_snapshot,
            mesh_path=mesh_path,
            protocol=protocol,
            theta=float(theta),
            relax_steps=int(scan_steps),
        )
        outer_after = row.get("outer_shell_field", {})
        row["candidate_delta"] = {
            "tilt_out_norm_mean_delta": _as_float(outer_after.get("tilt_out_norm_mean"))
            - _as_float(base_outer.get("tilt_out_norm_mean")),
            "tilt_out_radial_mean_delta": _as_float(
                outer_after.get("tilt_out_radial_mean")
            )
            - _as_float(base_outer.get("tilt_out_radial_mean")),
        }
        rows.append(row)
    return rows


def _fixed_theta_relaxation_matrix(
    *,
    ctx,
    anchor_snapshot: dict[str, Any],
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per_theta: dict[str, list[dict[str, Any]]] = {}
    for theta in FIXED_THETA_GRID:
        theta_rows: list[dict[str, Any]] = []
        for steps in RELAXATION_BUDGETS:
            row = _collect_candidate_row(
                ctx=ctx,
                snapshot=anchor_snapshot,
                mesh_path=mesh_path,
                protocol=protocol,
                theta=float(theta),
                relax_steps=int(steps),
            )
            row["theta_label"] = f"{theta:.2f}"
            theta_rows.append(row)
            rows.append(row)
        per_theta[f"{theta:.2f}"] = theta_rows
    summaries = []
    for key, theta_rows in per_theta.items():
        summaries.append(
            {
                "theta_label": key,
                "classification": _fixed_theta_row_classification(theta_rows),
                "budget_rows": [
                    {
                        "relax_steps": int(row["relax_steps"]),
                        "elastic_total_from_breakdown": _as_float(
                            row.get("elastic_total_from_breakdown")
                        ),
                        "tilt_out": _as_float(
                            row.get("energy_breakdown", {}).get("tilt_out")
                        ),
                        "bending_tilt_out": _as_float(
                            row.get("energy_breakdown", {}).get("bending_tilt_out")
                        ),
                    }
                    for row in theta_rows
                ],
            }
        )
    return {"rows": rows, "theta_summaries": summaries}


def _thetaB_scan_sensitivity_matrix(
    *,
    ctx,
    anchor_snapshot: dict[str, Any],
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> dict[str, Any]:
    base_theta = float(anchor_snapshot["global_params"].get("tilt_thetaB_value") or 0.0)
    rows: list[dict[str, Any]] = []
    for main_steps in MAIN_INNER_STEPS:
        _restore_state(ctx, anchor_snapshot)
        gp = ctx.mesh.global_parameters
        gp.set("tilt_thetaB_optimize", False)
        gp.set("tilt_thetaB_value", float(base_theta))
        _relax_leaflets_for_steps(ctx, int(main_steps))
        warm_snapshot = _capture_state(ctx)
        warm_base = _collect_live_summary(
            ctx=ctx, mesh_path=mesh_path, protocol=protocol
        )
        for delta in SCAN_DELTAS:
            for scan_steps in SCAN_INNER_STEPS:
                local_rows = [
                    _collect_candidate_row(
                        ctx=ctx,
                        snapshot=warm_snapshot,
                        mesh_path=mesh_path,
                        protocol=protocol,
                        theta=base_theta + offset,
                        relax_steps=int(scan_steps),
                    )
                    for offset in (-delta, 0.0, delta)
                ]
                wide_rows = [
                    _collect_candidate_row(
                        ctx=ctx,
                        snapshot=warm_snapshot,
                        mesh_path=mesh_path,
                        protocol=protocol,
                        theta=float(theta),
                        relax_steps=int(scan_steps),
                    )
                    for theta in FIXED_THETA_GRID
                ]
                local_best = min(
                    local_rows,
                    key=lambda row: _as_float(
                        row.get("reduced_terms", {}).get("total_measured")
                    ),
                )
                wide_best = min(
                    wide_rows,
                    key=lambda row: _as_float(
                        row.get("reduced_terms", {}).get("total_measured")
                    ),
                )
                rows.append(
                    {
                        "main_inner_steps": int(main_steps),
                        "scan_delta": float(delta),
                        "scan_inner_steps": int(scan_steps),
                        "warm_start": {
                            "thetaB_value": _as_float(warm_base.get("thetaB_value")),
                            "elastic_total_from_breakdown": _as_float(
                                warm_base.get("elastic_total_from_breakdown")
                            ),
                            "tilt_out": _as_float(
                                warm_base.get("energy_breakdown", {}).get("tilt_out")
                            ),
                            "bending_tilt_out": _as_float(
                                warm_base.get("energy_breakdown", {}).get(
                                    "bending_tilt_out"
                                )
                            ),
                        },
                        "local_candidates": [
                            {
                                "thetaB_value": _as_float(row.get("thetaB_value")),
                                "total_measured": _as_float(
                                    row.get("reduced_terms", {}).get("total_measured")
                                ),
                                "elastic_total_from_breakdown": _as_float(
                                    row.get("elastic_total_from_breakdown")
                                ),
                            }
                            for row in local_rows
                        ],
                        "wide_candidates": [
                            {
                                "thetaB_value": _as_float(row.get("thetaB_value")),
                                "total_measured": _as_float(
                                    row.get("reduced_terms", {}).get("total_measured")
                                ),
                                "elastic_total_from_breakdown": _as_float(
                                    row.get("elastic_total_from_breakdown")
                                ),
                            }
                            for row in wide_rows
                        ],
                        "local_best_thetaB": _as_float(local_best.get("thetaB_value")),
                        "local_best_total": _as_float(
                            local_best.get("reduced_terms", {}).get("total_measured")
                        ),
                        "wide_best_thetaB": _as_float(wide_best.get("thetaB_value")),
                        "wide_best_total": _as_float(
                            wide_best.get("reduced_terms", {}).get("total_measured")
                        ),
                        "classification": _local_vs_wide_classification(
                            local_best_theta=_as_float(local_best.get("thetaB_value")),
                            wide_best_theta=_as_float(wide_best.get("thetaB_value")),
                            local_best_energy=_as_float(
                                local_best.get("reduced_terms", {}).get(
                                    "total_measured"
                                )
                            ),
                            wide_best_energy=_as_float(
                                wide_best.get("reduced_terms", {}).get("total_measured")
                            ),
                        ),
                    }
                )
    return {"base_thetaB": float(base_theta), "rows": rows}


def _single_step_relaxation_trace(
    *,
    ctx,
    anchor_snapshot: dict[str, Any],
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for theta in TRACE_THETAS:
        _restore_state(ctx, anchor_snapshot)
        ctx.mesh.global_parameters.set("tilt_thetaB_optimize", False)
        ctx.mesh.global_parameters.set("tilt_thetaB_value", float(theta))
        rows.append(
            {
                "theta_label": f"{theta:.2f}",
                "pass_index": 0,
                **_collect_live_summary(
                    ctx=ctx, mesh_path=mesh_path, protocol=protocol
                ),
            }
        )
        for pass_index in range(1, TRACE_PASSES + 1):
            _relax_leaflets_for_steps(ctx, 1)
            rows.append(
                {
                    "theta_label": f"{theta:.2f}",
                    "pass_index": int(pass_index),
                    **_collect_live_summary(
                        ctx=ctx, mesh_path=mesh_path, protocol=protocol
                    ),
                }
            )
    return rows


def _line_search_interaction(
    *,
    tmpdir: Path,
    protocol: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reduced_steps in LINE_SEARCH_REDUCED_STEPS:
        doc = _build_variant_doc(
            DEFAULT_FIXTURE,
            {"line_search_reduced_tilt_inner_steps": int(reduced_steps)},
        )
        mesh_path = _write_temp_fixture(
            doc, tmpdir, f"default_line_search_reduced_{reduced_steps}"
        )
        ctx, summary = _run_protocol_summary(mesh_path=mesh_path, protocol=protocol)
        trace = getattr(ctx.mesh, "_thetaB_scan_trace", [])
        summary.update(
            {
                "label": f"reduced_{reduced_steps}",
                "line_search_reduced_tilt_inner_steps": int(reduced_steps),
                "thetaB_scan_tail": trace[-5:] if isinstance(trace, list) else [],
            }
        )
        rows.append(summary)
    return rows


def _relaxation_solver_path(
    *,
    ctx,
    anchor_snapshot: dict[str, Any],
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gp = ctx.mesh.global_parameters
    for theta in SOLVER_PATH_THETAS:
        for variant in SOLVER_PATH_VARIANTS:
            _restore_state(ctx, anchor_snapshot)
            overrides = {
                "tilt_thetaB_optimize": False,
                "tilt_thetaB_value": float(theta),
                "tilt_solver": str(variant["solver"]),
                "tilt_inner_steps": int(SOLVER_PATH_INNER_STEPS),
                "tilt_coupled_steps": int(SOLVER_PATH_INNER_STEPS),
                "tilt_cg_max_iters": int(SOLVER_PATH_INNER_STEPS),
            }
            if variant.get("preconditioner") is not None:
                overrides["tilt_cg_preconditioner"] = str(variant["preconditioner"])
            present, old = _temporary_overrides(gp, overrides)
            try:
                ctx.minimizer._relax_leaflet_tilts(
                    positions=ctx.mesh.positions_view(),
                    mode=str(gp.get("tilt_solve_mode", "coupled") or "coupled"),
                )
                row = _collect_live_summary(
                    ctx=ctx, mesh_path=mesh_path, protocol=protocol
                )
            finally:
                _restore_overrides(gp, present, old)
            row["requested_thetaB_value"] = float(theta)
            row["solver_path_label"] = str(variant["label"])
            rows.append(row)
    return rows


def _outer_bending_tilt_gradient_components(
    *,
    ctx,
    div_term_sign: float,
    pullback_sign: float,
) -> dict[str, Any]:
    mesh = ctx.mesh
    gp = mesh.global_parameters
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    resolver = ctx.minimizer.param_resolver
    tilts_out = mesh.tilts_out_view()
    shell_rows = _outer_shell_rows(mesh)

    payload = bending_tilt_out_module._leaflet._leaflet_triangle_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        cache_tag="out",
        ctx=ctx.minimizer.energy_context(),
    )
    tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
    tri_rows_full = np.asarray(payload["tri_rows_full"], dtype=np.int32)
    weights = np.asarray(payload["weights"], dtype=float)
    k_vecs = np.asarray(payload["k_vecs"], dtype=float)
    vertex_areas_vor = np.asarray(payload["vertex_areas_vor"], dtype=float)
    if tri_rows.size == 0:
        zero = np.zeros((len(mesh.vertex_ids), 3), dtype=float)
        return {
            "base_gradient": zero,
            "divergence_gradient": zero,
            "combined_gradient": zero,
            "tilt_gradient": zero,
            "bending_gradient": zero,
            "summary": {
                "base_term_outer_shell_mean": 0.0,
                "div_eval_outer_shell_mean": 0.0,
                "dE_ddiv_base_outer_shell_mean": 0.0,
                "dE_ddiv_div_outer_shell_mean": 0.0,
                "triangle_count": 0,
                "shell_triangle_count": 0,
            },
        }

    g0_use = payload["g0"]
    g1_use = payload["g1"]
    g2_use = payload["g2"]
    transport_model = _resolve_transport_model(
        gp.get("tilt_transport_model", "ambient_v1") if gp is not None else "ambient_v1"
    )
    if g0_use is not None and g1_use is not None and g2_use is not None:
        g0 = np.asarray(g0_use, dtype=float)
        g1 = np.asarray(g1_use, dtype=float)
        g2 = np.asarray(g2_use, dtype=float)
        div_tri = compute_divergence_from_basis(
            mesh=mesh,
            tilts=tilts_out,
            tri_rows=tri_rows,
            g0=g0,
            g1=g1,
            g2=g2,
            positions=positions,
            transport_model=transport_model,
        )
    else:
        div_tri, _, g0, g1, g2 = p1_triangle_divergence(
            mesh=mesh,
            positions=positions,
            tilts=tilts_out,
            tri_rows=tri_rows,
            transport_model=transport_model,
        )
        g0 = np.asarray(g0, dtype=float)
        g1 = np.asarray(g1, dtype=float)
        g2 = np.asarray(g2, dtype=float)

    div_eval_tri = float(div_term_sign) * np.asarray(div_tri, dtype=float)
    _vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token="thetaB_cadence_outer_coupling",
        compute_vertex_areas=False,
    )
    static_payload = bending_tilt_out_module._leaflet._leaflet_static_tilt_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        k_vecs=k_vecs,
        vertex_areas_vor=vertex_areas_vor,
        tri_rows=tri_rows,
        kappa_key="bending_modulus_out",
        cache_tag="out",
    )
    base_term = np.asarray(static_payload["base_term"], dtype=float)
    base_tri = np.asarray(static_payload["base_tri"], dtype=float)
    kappa_tri = np.asarray(static_payload["kappa_tri"], dtype=float)
    coeff_base = (
        (kappa_tri[:, 0] * base_tri[:, 0] * va0_eff)
        + (kappa_tri[:, 1] * base_tri[:, 1] * va1_eff)
        + (kappa_tri[:, 2] * base_tri[:, 2] * va2_eff)
    )
    coeff_div = (
        (kappa_tri[:, 0] * div_eval_tri * va0_eff)
        + (kappa_tri[:, 1] * div_eval_tri * va1_eff)
        + (kappa_tri[:, 2] * div_eval_tri * va2_eff)
    )
    base_gradient = _accumulate_tri_gradient(
        n_vertices=len(mesh.vertex_ids),
        tri_rows=tri_rows,
        factor=float(pullback_sign) * coeff_base,
        g0=g0,
        g1=g1,
        g2=g2,
    )
    divergence_gradient = _accumulate_tri_gradient(
        n_vertices=len(mesh.vertex_ids),
        tri_rows=tri_rows,
        factor=float(pullback_sign) * coeff_div,
        g0=g0,
        g1=g1,
        g2=g2,
    )
    tilt_gradient = np.zeros_like(tilts_out)
    tilt_out_module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=np.zeros_like(positions),
        ctx=ctx.minimizer.energy_context(),
        tilts_out=tilts_out,
        tilt_out_grad_arr=tilt_gradient,
    )
    bending_gradient = base_gradient + divergence_gradient
    shell_tri_mask = (
        np.any(np.isin(tri_rows, shell_rows), axis=1)
        if shell_rows.size and tri_rows.size
        else np.zeros(len(tri_rows), dtype=bool)
    )
    return {
        "base_gradient": base_gradient,
        "divergence_gradient": divergence_gradient,
        "combined_gradient": tilt_gradient + bending_gradient,
        "tilt_gradient": tilt_gradient,
        "bending_gradient": bending_gradient,
        "summary": {
            "base_term_outer_shell_mean": float(np.mean(base_term[shell_rows]))
            if shell_rows.size
            else 0.0,
            "div_eval_outer_shell_mean": float(
                np.mean(
                    [
                        np.mean(div_eval_tri[np.any(tri_rows == row, axis=1)])
                        for row in shell_rows
                        if np.any(tri_rows == row)
                    ]
                )
            )
            if shell_rows.size
            else 0.0,
            "dE_ddiv_base_mean": float(np.mean(coeff_base)) if coeff_base.size else 0.0,
            "dE_ddiv_div_mean": float(np.mean(coeff_div)) if coeff_div.size else 0.0,
            "triangle_count": int(len(tri_rows_full)),
            "shell_triangle_count": int(np.sum(shell_tri_mask)),
        },
    }


def _outer_coupling_sign_sweep(
    *,
    ctx,
    anchor_snapshot: dict[str, Any],
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shell_rows = _outer_shell_rows(ctx.mesh)
    gp = ctx.mesh.global_parameters
    for theta in OUTER_COUPLING_SWEEP_THETAS:
        for variant in OUTER_COUPLING_SIGN_VARIANTS:
            _restore_state(ctx, anchor_snapshot)
            present, old = _temporary_param_patch(
                gp,
                set_overrides=(
                    {"bending_tilt_base_term_reference_mode": variant["reference_mode"]}
                    if variant["reference_mode"] != "fixture"
                    else {}
                ),
            )
            try:
                gp.set("tilt_thetaB_optimize", False)
                gp.set("tilt_thetaB_value", float(theta))
                live = _collect_live_summary(
                    ctx=ctx, mesh_path=mesh_path, protocol=protocol
                )
                coupling = _outer_bending_tilt_gradient_components(
                    ctx=ctx,
                    div_term_sign=float(variant["div_term_sign"]),
                    pullback_sign=float(variant["pullback_sign"]),
                )
            finally:
                _restore_overrides(gp, present, old)

            tilt_gradient = np.asarray(coupling["tilt_gradient"], dtype=float)
            bending_gradient = np.asarray(coupling["bending_gradient"], dtype=float)
            base_gradient = np.asarray(coupling["base_gradient"], dtype=float)
            div_gradient = np.asarray(coupling["divergence_gradient"], dtype=float)
            combined_gradient = np.asarray(coupling["combined_gradient"], dtype=float)
            rows.append(
                {
                    "requested_thetaB_value": float(theta),
                    "variant_label": str(variant["label"]),
                    "thetaB_value": _as_float(live.get("thetaB_value")),
                    "energy_breakdown": dict(live.get("energy_breakdown", {})),
                    "tilt_shell_summary": _shell_vector_summary(
                        ctx.mesh, tilt_gradient, shell_rows
                    ),
                    "bending_shell_summary": _shell_vector_summary(
                        ctx.mesh, bending_gradient, shell_rows
                    ),
                    "base_shell_summary": _shell_vector_summary(
                        ctx.mesh, base_gradient, shell_rows
                    ),
                    "divergence_shell_summary": _shell_vector_summary(
                        ctx.mesh, div_gradient, shell_rows
                    ),
                    "combined_shell_summary": _shell_vector_summary(
                        ctx.mesh, combined_gradient, shell_rows
                    ),
                    "tilt_vs_bending_cosine": _cosine_similarity(
                        tilt_gradient[shell_rows]
                        if shell_rows.size
                        else np.zeros((0, 3)),
                        bending_gradient[shell_rows]
                        if shell_rows.size
                        else np.zeros((0, 3)),
                    ),
                    "base_vs_divergence_cosine": _cosine_similarity(
                        base_gradient[shell_rows]
                        if shell_rows.size
                        else np.zeros((0, 3)),
                        div_gradient[shell_rows]
                        if shell_rows.size
                        else np.zeros((0, 3)),
                    ),
                    "descent_shell_update_summary": _shell_vector_summary(
                        ctx.mesh, -combined_gradient, shell_rows
                    ),
                    "bending_coupling_summary": dict(coupling["summary"]),
                }
            )
    return rows


def _outer_energy_gradient_assembly(
    *,
    ctx,
    anchor_snapshot: dict[str, Any],
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = ctx.mesh.positions_view()
    index_map = ctx.mesh.vertex_index_to_row
    resolver = ctx.minimizer.param_resolver
    shell_rows = _outer_shell_rows(ctx.mesh)
    region_masks = row_region_mask_dict(ctx.mesh)

    for theta in ASSEMBLY_THETAS:
        _restore_state(ctx, anchor_snapshot)
        gp = ctx.mesh.global_parameters
        gp.set("tilt_thetaB_optimize", False)
        gp.set("tilt_thetaB_value", float(theta))
        live = _collect_live_summary(ctx=ctx, mesh_path=mesh_path, protocol=protocol)

        tilt_shape_grad = np.zeros_like(positions)
        tilt_out_grad = np.zeros_like(ctx.mesh.tilts_out_view())
        tilt_energy = tilt_out_module.compute_energy_and_gradient_array(
            ctx.mesh,
            gp,
            resolver,
            positions=positions,
            index_map=index_map,
            grad_arr=tilt_shape_grad,
            ctx=ctx.minimizer.energy_context(),
            tilts_out=ctx.mesh.tilts_out_view(),
            tilt_out_grad_arr=tilt_out_grad,
        )

        btl_shape_grad = np.zeros_like(positions)
        btl_out_grad = np.zeros_like(ctx.mesh.tilts_out_view())
        btl_energy = bending_tilt_out_module.compute_energy_and_gradient_array(
            ctx.mesh,
            gp,
            resolver,
            positions=positions,
            index_map=index_map,
            grad_arr=btl_shape_grad,
            ctx=ctx.minimizer.energy_context(),
            tilts_out=ctx.mesh.tilts_out_view(),
            tilt_out_grad_arr=btl_out_grad,
        )

        payload = bending_tilt_out_module._leaflet._leaflet_triangle_payload(
            ctx.mesh,
            gp,
            positions=positions,
            index_map=index_map,
            cache_tag="out",
            ctx=ctx.minimizer.energy_context(),
        )
        tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
        tri_rows_full = np.asarray(payload["tri_rows_full"], dtype=np.int32)
        tri_keep = np.asarray(payload["tri_keep"], dtype=bool)
        static_payload = bending_tilt_out_module._leaflet._leaflet_static_tilt_payload(
            ctx.mesh,
            gp,
            positions=positions,
            index_map=index_map,
            k_vecs=np.asarray(payload["k_vecs"], dtype=float),
            vertex_areas_vor=np.asarray(payload["vertex_areas_vor"], dtype=float),
            tri_rows=tri_rows,
            kappa_key="bending_modulus_out",
            cache_tag="out",
        )
        active_weights = tilt_out_module._active_row_weights(ctx.mesh, resolver)
        shell_weight_mean = (
            float(np.mean(active_weights[shell_rows]))
            if active_weights is not None and shell_rows.size
            else 1.0
        )

        rows.append(
            {
                "requested_thetaB_value": float(theta),
                "thetaB_value": _as_float(live.get("thetaB_value")),
                "elastic_total_from_breakdown": _as_float(
                    live.get("elastic_total_from_breakdown")
                ),
                "energy_breakdown": dict(live.get("energy_breakdown", {})),
                "tilt_out_module": {
                    "energy": float(tilt_energy),
                    "tilt_grad_norm_by_region": _region_l2_summary(
                        tilt_out_grad, region_masks, shell_rows=shell_rows
                    ),
                    "shape_grad_norm_by_region": _region_l2_summary(
                        tilt_shape_grad, region_masks, shell_rows=shell_rows
                    ),
                    "active_row_weight_mean_outer_shell": float(shell_weight_mean),
                },
                "bending_tilt_out_module": {
                    "energy": float(btl_energy),
                    "tilt_grad_norm_by_region": _region_l2_summary(
                        btl_out_grad, region_masks, shell_rows=shell_rows
                    ),
                    "shape_grad_norm_by_region": _region_l2_summary(
                        btl_shape_grad, region_masks, shell_rows=shell_rows
                    ),
                    "triangle_counts": {
                        "full": int(len(tri_rows_full)),
                        "kept": int(len(tri_rows)),
                        "kept_touching_outer_shell": int(
                            np.sum(np.any(np.isin(tri_rows, shell_rows), axis=1))
                        )
                        if tri_rows.size and shell_rows.size
                        else 0,
                        "full_touching_outer_shell": int(
                            np.sum(np.any(np.isin(tri_rows_full, shell_rows), axis=1))
                        )
                        if tri_rows_full.size and shell_rows.size
                        else 0,
                        "removed_by_absence_or_transition": int(np.sum(~tri_keep))
                        if tri_keep.size
                        else 0,
                    },
                    "base_term_outer_shell_abs_mean": float(
                        np.mean(
                            np.abs(
                                np.asarray(static_payload["base_term"], dtype=float)[
                                    shell_rows
                                ]
                            )
                        )
                    )
                    if shell_rows.size
                    else 0.0,
                    "base_term_outer_shell_mean": float(
                        np.mean(
                            np.asarray(static_payload["base_term"], dtype=float)[
                                shell_rows
                            ]
                        )
                    )
                    if shell_rows.size
                    else 0.0,
                },
                "combined_outer_shell_gradient": {
                    "norm": float(
                        np.linalg.norm((tilt_out_grad + btl_out_grad)[shell_rows])
                    )
                    if shell_rows.size
                    else 0.0,
                    "cosine": _cosine_similarity(
                        tilt_out_grad[shell_rows]
                        if shell_rows.size
                        else np.zeros((0, 3)),
                        btl_out_grad[shell_rows]
                        if shell_rows.size
                        else np.zeros((0, 3)),
                    ),
                },
            }
        )
    return rows


def _runtime_gradient_bridge(
    *,
    ctx,
    anchor_snapshot: dict[str, Any],
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mesh = ctx.mesh
    minim = ctx.minimizer
    resolver = minim.param_resolver
    shell_rows = _outer_shell_rows(mesh)
    region_masks = row_region_mask_dict(mesh)
    outer_shell_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    if shell_rows.size:
        outer_shell_mask[shell_rows] = True

    for theta in RUNTIME_BRIDGE_THETAS:
        _restore_state(ctx, anchor_snapshot)
        gp = mesh.global_parameters
        gp.set("tilt_thetaB_optimize", False)
        gp.set("tilt_thetaB_value", float(theta))
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row
        tilts_in = mesh.tilts_in_view().copy(order="F")
        tilts_out = mesh.tilts_out_view().copy(order="F")

        tri_rows, _ = minim._triangle_rows()
        tri_rows = (
            np.asarray(tri_rows, dtype=np.int32)
            if tri_rows is not None and len(tri_rows) > 0
            else np.zeros((0, 3), dtype=np.int32)
        )
        tilt_vertex_areas_in = mesh.barycentric_vertex_areas(positions=positions)
        absent_mask_out = leaflet_absent_vertex_mask(mesh, gp, leaflet="out")
        if not np.any(absent_mask_out):
            tilt_vertex_areas_out = tilt_vertex_areas_in
        else:
            tri_keep_out = leaflet_present_triangle_mask(
                mesh, tri_rows, absent_vertex_mask=absent_mask_out
            )
            tri_rows_out = tri_rows[tri_keep_out] if tri_keep_out.size else tri_rows
            tilt_vertex_areas_out = (
                np.zeros(len(mesh.vertex_ids), dtype=float)
                if tri_rows_out.size == 0
                else minim._tilt_vertex_areas_from_triangles(
                    n_vertices=len(mesh.vertex_ids),
                    tri_rows=tri_rows_out,
                    positions=positions,
                )
            )

        live = _collect_live_summary(ctx=ctx, mesh_path=mesh_path, protocol=protocol)
        energy_ctx = minim.energy_context()
        tilt_shape_grad = np.zeros_like(positions)
        tilt_out_grad = np.zeros_like(tilts_out)
        tilt_out_module.compute_energy_and_gradient_array(
            mesh,
            gp,
            resolver,
            positions=positions,
            index_map=index_map,
            grad_arr=tilt_shape_grad,
            ctx=energy_ctx,
            tilts_out=tilts_out,
            tilt_out_grad_arr=tilt_out_grad,
        )
        btl_shape_grad = np.zeros_like(positions)
        btl_out_grad = np.zeros_like(tilts_out)
        bending_tilt_out_module.compute_energy_and_gradient_array(
            mesh,
            gp,
            resolver,
            positions=positions,
            index_map=index_map,
            grad_arr=btl_shape_grad,
            ctx=energy_ctx,
            tilts_out=tilts_out,
            tilt_out_grad_arr=btl_out_grad,
        )
        direct_outer_grad = tilt_out_grad + btl_out_grad

        runtime_grad_in = np.zeros_like(tilts_in)
        runtime_grad_out = np.zeros_like(tilts_out)
        grad_dummy = np.zeros_like(positions)
        minim._compute_energy_and_leaflet_tilt_gradients_array(
            positions=positions,
            tilts_in=tilts_in,
            tilts_out=tilts_out,
            tilt_in_grad_arr=runtime_grad_in,
            tilt_out_grad_arr=runtime_grad_out,
            tilt_vertex_areas_in=tilt_vertex_areas_in,
            tilt_vertex_areas_out=tilt_vertex_areas_out,
            grad_dummy=grad_dummy,
            tilt_only=True,
        )
        runtime_before_out = np.array(runtime_grad_out, copy=True)
        runtime_after_out = np.array(runtime_grad_out, copy=True)
        runtime_after_in = np.array(runtime_grad_in, copy=True)
        minim.constraint_manager.apply_tilt_gradient_modifications_array(
            runtime_after_in,
            runtime_after_out,
            mesh,
            gp,
            positions=positions,
            tilts_in=tilts_in,
            tilts_out=tilts_out,
        )
        fixed_mask_out = np.asarray(minim._tilt_fixed_mask_out(), dtype=bool)
        if np.any(fixed_mask_out):
            runtime_after_out[fixed_mask_out] = 0.0

        bridge_snapshot = _capture_state(ctx)
        overrides = {
            "tilt_thetaB_optimize": False,
            "tilt_thetaB_value": float(theta),
            "tilt_inner_steps": 1,
            "tilt_coupled_steps": 1,
            "tilt_cg_max_iters": 1,
        }
        present, old = _temporary_overrides(gp, overrides)
        try:
            minim._relax_leaflet_tilts(
                positions=mesh.positions_view(),
                mode=str(gp.get("tilt_solve_mode", "coupled") or "coupled"),
            )
            after_state = _capture_state(ctx)
            relax_stats = _relaxation_stats_snapshot(ctx)
        finally:
            _restore_overrides(gp, present, old)
            _restore_state(ctx, bridge_snapshot)

        delta_out = np.asarray(after_state["tilts_out"], dtype=float) - np.asarray(
            bridge_snapshot["tilts_out"], dtype=float
        )
        direct_shell = (
            direct_outer_grad[shell_rows] if shell_rows.size else np.zeros((0, 3))
        )
        runtime_before_shell = (
            runtime_before_out[shell_rows] if shell_rows.size else np.zeros((0, 3))
        )
        runtime_after_shell = (
            runtime_after_out[shell_rows] if shell_rows.size else np.zeros((0, 3))
        )
        update_shell = delta_out[shell_rows] if shell_rows.size else np.zeros((0, 3))
        rows.append(
            {
                "requested_thetaB_value": float(theta),
                "thetaB_value": _as_float(live.get("thetaB_value")),
                "energy_breakdown": live.get("energy_breakdown", {}),
                "elastic_total_from_breakdown": _as_float(
                    live.get("elastic_total_from_breakdown")
                ),
                "direct_module_outer_gradient": {
                    "tilt_out_shell_norm": float(
                        np.linalg.norm(tilt_out_grad[shell_rows])
                    )
                    if shell_rows.size
                    else 0.0,
                    "bending_tilt_out_shell_norm": float(
                        np.linalg.norm(btl_out_grad[shell_rows])
                    )
                    if shell_rows.size
                    else 0.0,
                    "tilt_grad_norm_by_region": _region_l2_summary(
                        direct_outer_grad,
                        region_masks,
                        shell_rows=shell_rows,
                    ),
                    "tilt_vs_bending_cosine": _cosine_similarity(
                        tilt_out_grad[shell_rows]
                        if shell_rows.size
                        else np.zeros((0, 3)),
                        btl_out_grad[shell_rows]
                        if shell_rows.size
                        else np.zeros((0, 3)),
                    ),
                },
                "runtime_aggregated_gradient_before_constraints": {
                    "tilt_grad_norm_by_region": _region_l2_summary(
                        runtime_before_out,
                        region_masks,
                        shell_rows=shell_rows,
                    ),
                },
                "runtime_aggregated_gradient_after_constraints": {
                    "tilt_grad_norm_by_region": _region_l2_summary(
                        runtime_after_out,
                        region_masks,
                        shell_rows=shell_rows,
                    ),
                },
                "accepted_update": {
                    "tilt_grad_norm_by_region": _region_l2_summary(
                        delta_out,
                        region_masks,
                        shell_rows=shell_rows,
                    ),
                },
                "shell_vector_comparison": {
                    "direct_vs_runtime_before_cosine": _cosine_similarity(
                        direct_shell, runtime_before_shell
                    ),
                    "direct_vs_runtime_before_norm_ratio": (
                        float(np.linalg.norm(runtime_before_shell))
                        / max(float(np.linalg.norm(direct_shell)), 1.0e-30)
                    ),
                    "runtime_before_vs_after_cosine": _cosine_similarity(
                        runtime_before_shell, runtime_after_shell
                    ),
                    "runtime_before_vs_after_norm_ratio": (
                        float(np.linalg.norm(runtime_after_shell))
                        / max(float(np.linalg.norm(runtime_before_shell)), 1.0e-30)
                    ),
                    "runtime_after_vs_update_cosine": _cosine_similarity(
                        runtime_after_shell, update_shell
                    ),
                    "runtime_after_vs_update_norm_ratio": (
                        float(np.linalg.norm(update_shell))
                        / max(float(np.linalg.norm(runtime_after_shell)), 1.0e-30)
                    ),
                },
                "leaflet_relaxation_stats": relax_stats,
            }
        )
    return rows


def _base_term_reference_sweep(
    *,
    ctx,
    anchor_snapshot: dict[str, Any],
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = ctx.mesh.positions_view()
    index_map = ctx.mesh.vertex_index_to_row
    resolver = ctx.minimizer.param_resolver
    shell_rows = _outer_shell_rows(ctx.mesh)
    region_masks = row_region_mask_dict(ctx.mesh)
    gp = ctx.mesh.global_parameters

    for variant in BASE_TERM_REFERENCE_VARIANTS:
        for theta in ASSEMBLY_THETAS:
            _restore_state(ctx, anchor_snapshot)
            present, old = _temporary_param_patch(
                gp,
                set_overrides=dict(variant.get("set") or {}),
                unset_keys=tuple(variant.get("unset") or ()),
            )
            try:
                gp.set("tilt_thetaB_optimize", False)
                gp.set("tilt_thetaB_value", float(theta))
                live = _collect_live_summary(
                    ctx=ctx, mesh_path=mesh_path, protocol=protocol
                )

                tilt_out_grad = np.zeros_like(ctx.mesh.tilts_out_view())
                tilt_out_module.compute_energy_and_gradient_array(
                    ctx.mesh,
                    gp,
                    resolver,
                    positions=positions,
                    index_map=index_map,
                    grad_arr=np.zeros_like(positions),
                    ctx=ctx.minimizer.energy_context(),
                    tilts_out=ctx.mesh.tilts_out_view(),
                    tilt_out_grad_arr=tilt_out_grad,
                )
                btl_out_grad = np.zeros_like(ctx.mesh.tilts_out_view())
                bending_tilt_out_module.compute_energy_and_gradient_array(
                    ctx.mesh,
                    gp,
                    resolver,
                    positions=positions,
                    index_map=index_map,
                    grad_arr=np.zeros_like(positions),
                    ctx=ctx.minimizer.energy_context(),
                    tilts_out=ctx.mesh.tilts_out_view(),
                    tilt_out_grad_arr=btl_out_grad,
                )
                payload = bending_tilt_out_module._leaflet._leaflet_triangle_payload(
                    ctx.mesh,
                    gp,
                    positions=positions,
                    index_map=index_map,
                    cache_tag="out",
                    ctx=ctx.minimizer.energy_context(),
                )
                static_payload = (
                    bending_tilt_out_module._leaflet._leaflet_static_tilt_payload(
                        ctx.mesh,
                        gp,
                        positions=positions,
                        index_map=index_map,
                        k_vecs=np.asarray(payload["k_vecs"], dtype=float),
                        vertex_areas_vor=np.asarray(
                            payload["vertex_areas_vor"], dtype=float
                        ),
                        tri_rows=np.asarray(payload["tri_rows"], dtype=np.int32),
                        kappa_key="bending_modulus_out",
                        cache_tag="out",
                    )
                )

                bridge_snapshot = _capture_state(ctx)
                inner_steps = int(
                    gp.get("tilt_thetaB_optimize_inner_steps", SCAN_INNER_STEPS[0])
                    or SCAN_INNER_STEPS[0]
                )
                step_present, step_old = _temporary_overrides(
                    gp,
                    {
                        "tilt_inner_steps": int(inner_steps),
                        "tilt_coupled_steps": int(inner_steps),
                        "tilt_cg_max_iters": int(inner_steps),
                    },
                )
                try:
                    ctx.minimizer._relax_leaflet_tilts(
                        positions=ctx.mesh.positions_view(),
                        mode=str(gp.get("tilt_solve_mode", "coupled") or "coupled"),
                    )
                    after_state = _capture_state(ctx)
                finally:
                    _restore_overrides(gp, step_present, step_old)
                    _restore_state(ctx, bridge_snapshot)

                delta_out = np.asarray(
                    after_state["tilts_out"], dtype=float
                ) - np.asarray(bridge_snapshot["tilts_out"], dtype=float)
                rows.append(
                    {
                        "variant_label": str(variant["label"]),
                        "requested_thetaB_value": float(theta),
                        "thetaB_value": _as_float(live.get("thetaB_value")),
                        "tex_ratio_summary": dict(live.get("tex_ratio_summary", {})),
                        "energy_breakdown": dict(live.get("energy_breakdown", {})),
                        "elastic_total_from_breakdown": _as_float(
                            live.get("elastic_total_from_breakdown")
                        ),
                        "tilt_out_shell_gradient": float(
                            np.linalg.norm(tilt_out_grad[shell_rows])
                        )
                        if shell_rows.size
                        else 0.0,
                        "bending_tilt_out_shell_gradient": float(
                            np.linalg.norm(btl_out_grad[shell_rows])
                        )
                        if shell_rows.size
                        else 0.0,
                        "combined_outer_shell_gradient": {
                            "norm": float(
                                np.linalg.norm(
                                    (tilt_out_grad + btl_out_grad)[shell_rows]
                                )
                            )
                            if shell_rows.size
                            else 0.0,
                            "cosine": _cosine_similarity(
                                tilt_out_grad[shell_rows]
                                if shell_rows.size
                                else np.zeros((0, 3)),
                                btl_out_grad[shell_rows]
                                if shell_rows.size
                                else np.zeros((0, 3)),
                            ),
                            "by_region": _region_l2_summary(
                                tilt_out_grad + btl_out_grad,
                                region_masks,
                                shell_rows=shell_rows,
                            ),
                        },
                        "outer_shell_base_term_mean": float(
                            np.mean(
                                np.asarray(static_payload["base_term"], dtype=float)[
                                    shell_rows
                                ]
                            )
                        )
                        if shell_rows.size
                        else 0.0,
                        "outer_shell_base_term_abs_mean": float(
                            np.mean(
                                np.abs(
                                    np.asarray(
                                        static_payload["base_term"], dtype=float
                                    )[shell_rows]
                                )
                            )
                        )
                        if shell_rows.size
                        else 0.0,
                        "first_accepted_shell_update_norm": float(
                            np.linalg.norm(delta_out[shell_rows])
                        )
                        if shell_rows.size
                        else 0.0,
                    }
                )
            finally:
                _restore_overrides(gp, present, old)
    return rows


def _full_physics_lane_matrix(
    *, tmpdir: Path, protocol: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in _full_physics_lane_specs():
        doc = _build_variant_doc(spec["base_fixture"], spec["overrides"])
        mesh_path = _write_temp_fixture(doc, tmpdir, f"{spec['label']}_lane")
        ctx, summary = _run_protocol_summary(mesh_path=mesh_path, protocol=protocol)
        shell_rows = _outer_shell_rows(ctx.mesh)
        theta = float(summary.get("thetaB_value") or 0.0)
        coupling = _outer_bending_tilt_gradient_components(
            ctx=ctx,
            div_term_sign=1.0,
            pullback_sign=1.0,
        )
        base_term_summary = _base_term_summary_for_fixture(
            mesh_path, str(spec["label"])
        )
        rows.append(
            {
                "label": str(spec["label"]),
                "thetaB_value": _as_float(summary.get("thetaB_value")),
                "tex_total_ratio": _as_float(summary.get("tex_total_ratio")),
                "tex_ratio_summary": dict(summary.get("tex_ratio_summary", {})),
                "energy_breakdown": dict(summary.get("energy_breakdown", {})),
                "elastic_total_from_breakdown": _as_float(
                    summary.get("elastic_total_from_breakdown")
                ),
                "gradient_norm": _as_float(summary.get("gradient_norm")),
                "leaflet_relaxation_stats": dict(
                    summary.get("leaflet_relaxation_stats", {})
                ),
                "outer_participation": _outer_participation_snapshot(ctx.mesh, ctx=ctx),
                "combined_shell_summary": _shell_vector_summary(
                    ctx.mesh, coupling["combined_gradient"], shell_rows
                ),
                "tilt_shell_summary": _shell_vector_summary(
                    ctx.mesh, coupling["tilt_gradient"], shell_rows
                ),
                "bending_shell_summary": _shell_vector_summary(
                    ctx.mesh, coupling["bending_gradient"], shell_rows
                ),
                "first_shell_update_summary": _one_step_shell_update_summary(
                    ctx=ctx, theta=theta
                ),
                "bending_coupling_summary": dict(coupling["summary"]),
                "model_intent": str(
                    summary.get("report_metadata", {}).get("model_intent") or ""
                ),
                "reference_mode": str(
                    summary.get("report_metadata", {}).get("reference_mode") or ""
                ),
                "base_term_summary": base_term_summary,
            }
        )
    return rows


def _trace_convergence_current_geometry_doc(
    *,
    geometry: str,
    epsilon: float | None,
) -> tuple[dict[str, Any], str]:
    base_doc = yaml.safe_load(DEFAULT_FIXTURE.read_text(encoding="utf-8")) or {}
    if geometry == "no_trace_current":
        doc = _build_variant_doc(FULL_COUPLING_FIXTURE, {})
        return doc, "full_coupling_no_trace_current"
    if geometry == "trace_only":
        if epsilon is None:
            raise ValueError("trace_only convergence row requires epsilon")
        doc = build_full_physics_trace_fixture(
            base_doc=base_doc,
            lane=f"physical_edge_full_coupling_trace_eps{str(epsilon).replace('.', '')}_conv",
            trace_radius=(7.0 / 15.0) + float(epsilon),
            planar_geometry=False,
        )
        return doc, f"full_coupling_trace_eps{str(epsilon).replace('.', '')}"
    if geometry == "fixed_support":
        if epsilon is None:
            raise ValueError("fixed_support convergence row requires epsilon")
        doc = build_outer_shell_scaffold_fixture(
            base_doc=base_doc,
            label=f"physical_edge_full_coupling_fixed_support_eps{str(epsilon).replace('.', '')}_n3_d005",
            trace_radius=(7.0 / 15.0) + float(epsilon),
            outer_shells=3,
            outer_shells_d=0.05,
            planar_geometry=False,
        )
        gp = dict(doc.get("global_parameters") or {})
        gp["theory_parity_lane"] = str(
            f"physical_edge_full_coupling_fixed_support_eps{str(epsilon).replace('.', '')}_n3_d005"
        )
        gp["bending_tilt_base_term_reference_mode"] = "current_geometry"
        doc["global_parameters"] = gp
        return doc, f"full_coupling_fixed_support_eps{str(epsilon).replace('.', '')}"
    if geometry == "gapfill_release":
        if epsilon is None:
            raise ValueError("gapfill_release convergence row requires epsilon")
        doc = build_gap_filled_outer_shell_scaffold_fixture(
            base_doc=base_doc,
            label=f"physical_edge_full_coupling_gapfill_release_eps{str(epsilon).replace('.', '')}_n3",
            trace_radius=(7.0 / 15.0) + float(epsilon),
            outer_shells=3,
            planar_geometry=False,
        )
        gp = dict(doc.get("global_parameters") or {})
        gp["theory_parity_lane"] = str(
            f"physical_edge_full_coupling_gapfill_release_eps{str(epsilon).replace('.', '')}_n3"
        )
        gp["bending_tilt_base_term_reference_mode"] = "current_geometry"
        doc["global_parameters"] = gp
        return doc, f"full_coupling_gapfill_release_eps{str(epsilon).replace('.', '')}"
    raise ValueError(f"unknown convergence geometry: {geometry}")


def _trace_convergence_row_classification(row: dict[str, Any]) -> str:
    if str(row.get("geometry")) == "no_trace_current":
        return "no_trace_control"
    if not np.isfinite(float(row.get("thetaB_value") or 0.0)):
        return "unstable_or_rejected"
    if (
        str(row.get("stop_reason") or "") == "line_search_rejected"
        and float(row.get("shell_update_norm") or 0.0) <= 0.0
    ):
        return "unstable_or_rejected"
    if (
        float(row.get("direct_t_out") or 0.0) > 0.1
        and float(row.get("direct_phi") or 0.0) > 0.1
    ):
        return "develops_trace"
    return "suppressed_trace"


def _trace_convergence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trace_only = [row for row in rows if str(row.get("geometry")) == "trace_only"]
    support_rows = [
        row
        for row in rows
        if str(row.get("geometry")) in {"fixed_support", "gapfill_release"}
    ]
    if trace_only and all(
        row.get("classification") == "develops_trace" for row in trace_only
    ):
        classification = "trace_only_robust"
    elif support_rows and any(
        row.get("classification") == "develops_trace" for row in support_rows
    ):
        classification = "support_needed"
    else:
        classification = "not_converged"
    return {
        "classification": classification,
        "trace_only_success_count": int(
            sum(
                1 for row in trace_only if row.get("classification") == "develops_trace"
            )
        ),
        "support_success_count": int(
            sum(
                1
                for row in support_rows
                if row.get("classification") == "develops_trace"
            )
        ),
    }


def _apply_current_geometry_scaffold_probe_options(
    doc: dict[str, Any],
    *,
    lane: str,
    variant: str,
) -> dict[str, Any]:
    result = copy.deepcopy(doc)
    gp = dict(result.get("global_parameters") or {})
    gp["theory_parity_lane"] = lane
    gp["rim_slope_match_mode"] = "physical_edge_staggered_v1"
    gp["bending_tilt_base_term_reference_mode"] = "current_geometry"
    if variant in {"projector_only", "runtime_options", "runtime_options_gd_fallback"}:
        gp["rim_slope_match_scaffold_projector_mode"] = "continuity_v2"
    if variant in {"runtime_options", "runtime_options_gd_fallback"}:
        gp["tilt_thetaB_contact_work_mode"] = "field_linear"
        gp["tilt_solver"] = "cg"
        gp["tilt_cg_max_iters"] = 120
        gp["tilt_mass_mode_in"] = "consistent"
        if variant == "runtime_options_gd_fallback":
            gp["tilt_cg_rejection_fallback"] = "gd"
        constraints = [str(x) for x in (result.get("constraint_modules") or [])]
        result["constraint_modules"] = [
            name for name in constraints if name != "tilt_thetaB_boundary_in"
        ]
    result["global_parameters"] = gp
    return result


def _scaffold_collapse_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    projector_rescue = any(
        row.get("variant") == "projector_only"
        and row.get("classification") == "develops_trace"
        for row in rows
    )
    runtime_rescue = any(
        row.get("variant") == "runtime_options"
        and row.get("classification") == "develops_trace"
        for row in rows
    )
    runtime_partial = any(
        row.get("variant") == "runtime_options"
        and row.get("classification") == "partial_trace_revival"
        for row in rows
    )
    fallback_rescue = any(
        row.get("variant") == "runtime_options_gd_fallback"
        and row.get("classification") == "develops_trace"
        for row in rows
    )
    fallback_partial = any(
        row.get("variant") == "runtime_options_gd_fallback"
        and row.get("classification") == "partial_trace_revival"
        for row in rows
    )
    if runtime_rescue:
        classification = "runtime_options_rescue"
    elif fallback_rescue:
        classification = "gd_fallback_rescue"
    elif fallback_partial:
        classification = "gd_fallback_partial_revival"
    elif runtime_partial:
        classification = "runtime_options_partial_revival"
    elif projector_rescue:
        classification = "projector_rescue"
    else:
        classification = "scaffold_suppression_persists"
    return {
        "classification": classification,
        "plain_success_count": int(
            sum(
                1
                for row in rows
                if row.get("variant") == "plain"
                and row.get("classification") == "develops_trace"
            )
        ),
        "projector_success_count": int(
            sum(
                1
                for row in rows
                if row.get("variant") == "projector_only"
                and row.get("classification") == "develops_trace"
            )
        ),
        "runtime_success_count": int(
            sum(
                1
                for row in rows
                if row.get("variant") == "runtime_options"
                and row.get("classification") == "develops_trace"
            )
        ),
        "runtime_partial_count": int(
            sum(
                1
                for row in rows
                if row.get("variant") == "runtime_options"
                and row.get("classification") == "partial_trace_revival"
            )
        ),
        "fallback_success_count": int(
            sum(
                1
                for row in rows
                if row.get("variant") == "runtime_options_gd_fallback"
                and row.get("classification") == "develops_trace"
            )
        ),
        "fallback_partial_count": int(
            sum(
                1
                for row in rows
                if row.get("variant") == "runtime_options_gd_fallback"
                and row.get("classification") == "partial_trace_revival"
            )
        ),
        "negative_branch_count": int(
            sum(
                1
                for row in rows
                if row.get("classification") == "negative_trace_branch"
            )
        ),
    }


def _scaffold_collapse_row_classification(row: dict[str, Any]) -> str:
    direct_t = float(row.get("direct_t_out") or 0.0)
    direct_phi = float(row.get("direct_phi") or 0.0)
    if direct_t < -0.02 and direct_phi < -0.02:
        return "negative_trace_branch"
    if direct_t > 0.1 and direct_phi > 0.1:
        return "develops_trace"
    if direct_t > 0.03 and direct_phi > 0.03:
        return "partial_trace_revival"
    if (
        float(row.get("shell_grad_norm") or 0.0) > 1.0e-3
        and float(row.get("shell_update_norm") or 0.0) > 1.0e-5
    ):
        return "active_but_not_trace_aligned"
    return "suppressed_trace"


def _rows_from_vertex_options(mesh, predicate) -> np.ndarray:
    rows: list[int] = []
    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", {}) or {}
        if predicate(opts):
            rows.append(int(row))
    return np.asarray(rows, dtype=int)


def _scaffold_role_rows(mesh) -> dict[str, np.ndarray]:
    try:
        shell_data = build_local_interface_shell_data(
            mesh, positions=mesh.positions_view()
        )
    except AssertionError:
        shell_data = None
    trace = _rows_from_vertex_options(
        mesh,
        lambda opts: str(opts.get("pin_to_circle_group") or "") == "trace_layer",
    )
    support = _rows_from_vertex_options(
        mesh, lambda opts: opts.get("outer_shell_scaffold_index") is not None
    )
    release = _rows_from_vertex_options(
        mesh, lambda opts: bool(opts.get("outer_shell_release_ring", False))
    )
    return {
        "disk": np.asarray(getattr(shell_data, "disk_rows", []), dtype=int)
        if shell_data is not None
        else np.zeros(0, dtype=int),
        "rim": np.asarray(getattr(shell_data, "rim_rows", []), dtype=int)
        if shell_data is not None
        else np.zeros(0, dtype=int),
        "trace": trace,
        "support": support,
        "release": release,
        "outer": np.asarray(getattr(shell_data, "outer_rows", []), dtype=int)
        if shell_data is not None
        else np.zeros(0, dtype=int),
    }


def _row_radial_mean(mesh, vectors: np.ndarray, rows: np.ndarray) -> float:
    if rows.size == 0:
        return 0.0
    vals = radial_projection(mesh, vectors)
    return float(np.mean(vals[rows]))


def _scaffold_role_field_summary(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    roles = _scaffold_role_rows(mesh)
    tin = np.asarray(mesh.tilts_in_view(), dtype=float)
    tout = np.asarray(mesh.tilts_out_view(), dtype=float)
    fixed_in = np.asarray(ctx.minimizer._tilt_fixed_mask_in(), dtype=bool)
    fixed_out = np.asarray(ctx.minimizer._tilt_fixed_mask_out(), dtype=bool)
    out: dict[str, Any] = {}
    for name, rows in roles.items():
        phi = _role_phi_to_inner_summary(ctx, roles=roles, role=name)
        out[name] = {
            "count": int(rows.size),
            "tilt_in_radial_mean": _row_radial_mean(mesh, tin, rows),
            "tilt_out_radial_mean": _row_radial_mean(mesh, tout, rows),
            "phi_to_inner": float(phi),
            "tilt_in_norm_mean": float(np.mean(np.linalg.norm(tin[rows], axis=1)))
            if rows.size
            else 0.0,
            "tilt_out_norm_mean": float(np.mean(np.linalg.norm(tout[rows], axis=1)))
            if rows.size
            else 0.0,
            "free_in_count": int(np.sum(~fixed_in[rows])) if rows.size else 0,
            "free_out_count": int(np.sum(~fixed_out[rows])) if rows.size else 0,
        }
    return out


def _role_phi_to_inner_summary(
    ctx, *, roles: dict[str, np.ndarray], role: str
) -> float:
    previous_role = {
        "trace": "disk",
        "support": "trace",
        "release": "support",
        "outer": "trace",
    }.get(str(role))
    if previous_role is None:
        return 0.0
    rows = np.asarray(roles.get(str(role), np.zeros(0, dtype=int)), dtype=int)
    prev = np.asarray(roles.get(previous_role, np.zeros(0, dtype=int)), dtype=int)
    if rows.size == 0 or prev.size == 0:
        return 0.0
    pos = ctx.mesh.positions_view()
    r = np.linalg.norm(pos[:, :2], axis=1)
    dr = float(np.mean(r[rows]) - np.mean(r[prev]))
    if abs(dr) <= 1.0e-12:
        return 0.0
    return float((np.mean(pos[rows, 2]) - np.mean(pos[prev, 2])) / dr)


def _one_step_outer_update_array(ctx, *, theta: float) -> np.ndarray:
    before = _capture_state(ctx)
    gp = ctx.mesh.global_parameters
    overrides = {
        "tilt_thetaB_optimize": False,
        "tilt_thetaB_value": float(theta),
        "tilt_inner_steps": 1,
        "tilt_coupled_steps": 1,
        "tilt_cg_max_iters": 1,
    }
    present, old = _temporary_overrides(gp, overrides)
    try:
        ctx.minimizer._relax_leaflet_tilts(
            positions=ctx.mesh.positions_view(),
            mode=str(gp.get("tilt_solve_mode", "coupled") or "coupled"),
        )
        after = _capture_state(ctx)
    finally:
        _restore_overrides(gp, present, old)
        _restore_state(ctx, before)
    return np.asarray(after["tilts_out"], dtype=float) - np.asarray(
        before["tilts_out"], dtype=float
    )


def _role_vector_summaries(ctx, vectors: np.ndarray) -> dict[str, dict[str, float]]:
    roles = _scaffold_role_rows(ctx.mesh)
    return {
        role: _shell_vector_summary(ctx.mesh, vectors, rows)
        for role, rows in roles.items()
        if role in {"trace", "support", "release"}
    }


def _leaflet_tilt_vertex_areas(
    ctx, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mesh = ctx.mesh
    minim = ctx.minimizer
    gp = mesh.global_parameters
    tri_rows, _ = minim._triangle_rows()
    tri_rows = (
        np.asarray(tri_rows, dtype=np.int32)
        if tri_rows is not None and len(tri_rows) > 0
        else np.zeros((0, 3), dtype=np.int32)
    )
    areas_in = mesh.barycentric_vertex_areas(positions=positions)
    absent_out = leaflet_absent_vertex_mask(mesh, gp, leaflet="out")
    if not np.any(absent_out):
        return areas_in, areas_in
    tri_keep_out = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_out
    )
    tri_rows_out = tri_rows[tri_keep_out] if tri_keep_out.size else tri_rows
    areas_out = (
        np.zeros(len(mesh.vertex_ids), dtype=float)
        if tri_rows_out.size == 0
        else minim._tilt_vertex_areas_from_triangles(
            n_vertices=len(mesh.vertex_ids),
            tri_rows=tri_rows_out,
            positions=positions,
        )
    )
    return areas_in, areas_out


def _energy_breakdown_with_leaflet_tilts(
    ctx,
    *,
    tilts_in: np.ndarray,
    tilts_out: np.ndarray,
) -> dict[str, float]:
    mesh = ctx.mesh
    original_in = mesh.tilts_in_view().copy(order="F")
    original_out = mesh.tilts_out_view().copy(order="F")
    try:
        mesh.set_tilts_in_from_array(tilts_in)
        mesh.set_tilts_out_from_array(tilts_out)
        return {
            k: float(v) for k, v in ctx.minimizer.compute_energy_breakdown().items()
        }
    finally:
        mesh.set_tilts_in_from_array(original_in)
        mesh.set_tilts_out_from_array(original_out)


def _tilt_dependent_total_from_breakdown(breakdown: dict[str, float]) -> float:
    return float(
        breakdown.get("tilt_in", 0.0)
        + breakdown.get("tilt_out", 0.0)
        + breakdown.get("bending_tilt_in", 0.0)
        + breakdown.get("bending_tilt_out", 0.0)
    )


def _vertex_option_role(opts: dict[str, Any]) -> str:
    if str(opts.get("pin_to_circle_group") or "") == "trace_layer":
        return "trace"
    if opts.get("outer_shell_scaffold_index") is not None:
        return "support"
    if bool(opts.get("outer_shell_release_ring", False)):
        return "release"
    return ""


def _freeze_doc_support_roles(
    doc: dict[str, Any],
    *,
    freeze_tilt_out: bool = False,
    freeze_geometry: bool = False,
) -> dict[str, Any]:
    result = copy.deepcopy(doc)
    vertices = result.get("vertices") or []
    if not isinstance(vertices, list):
        return result
    for vertex in vertices:
        if not isinstance(vertex, list) or len(vertex) < 4:
            continue
        opts = vertex[3]
        if not isinstance(opts, dict):
            continue
        if _vertex_option_role(opts) not in {"support", "release"}:
            continue
        if freeze_tilt_out:
            opts["tilt_fixed_out"] = True
        if freeze_geometry:
            opts["fixed"] = True
    return result


def _apply_support_ownership_probe_options(
    doc: dict[str, Any],
    *,
    lane: str,
    variant: str,
) -> dict[str, Any]:
    base_variant = (
        "runtime_options_gd_fallback"
        if variant
        in {
            "runtime_options_gd_fallback",
            "support_tilt_frozen",
            "support_geometry_frozen",
            "support_passive_trace_only",
        }
        else "runtime_options"
    )
    result = _apply_current_geometry_scaffold_probe_options(
        doc, lane=lane, variant=base_variant
    )
    if variant == "support_tilt_frozen":
        result = _freeze_doc_support_roles(result, freeze_tilt_out=True)
    elif variant == "support_geometry_frozen":
        result = _freeze_doc_support_roles(result, freeze_geometry=True)
    return result


def _support_continuation_probe(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    roles = _scaffold_role_rows(mesh)
    support_rows = np.unique(
        np.concatenate(
            [
                roles.get("support", np.zeros(0, dtype=int)),
                roles.get("release", np.zeros(0, dtype=int)),
            ]
        )
    )
    trace_rows = roles.get("trace", np.zeros(0, dtype=int))
    if support_rows.size == 0 or trace_rows.size == 0:
        return {"available": False, "samples": [], "best_sample": {}}

    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")
    base_breakdown = _energy_breakdown_with_leaflet_tilts(
        ctx, tilts_in=tilts_in, tilts_out=tilts_out
    )
    base_total = _tilt_dependent_total_from_breakdown(base_breakdown)
    trace_rad = _row_radial_mean(mesh, tilts_out, trace_rows)
    if abs(trace_rad) <= 1.0e-12:
        return {
            "available": True,
            "trace_radial_target": float(trace_rad),
            "samples": [],
            "best_sample": {},
            "energy_lowering_positive_branch": False,
        }

    positions = mesh.positions_view()
    _radii, r_hat = radial_unit_vectors(positions)
    samples: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for alpha in (0.25, 0.5, 1.0):
        trial_out = tilts_out.copy(order="F")
        current_rad = np.einsum(
            "ij,ij->i", trial_out[support_rows], r_hat[support_rows]
        )
        target_rad = (1.0 - float(alpha)) * current_rad + float(alpha) * float(
            trace_rad
        )
        delta_rad = target_rad - current_rad
        trial_out[support_rows] += delta_rad[:, None] * r_hat[support_rows]
        breakdown = _energy_breakdown_with_leaflet_tilts(
            ctx, tilts_in=tilts_in, tilts_out=trial_out
        )
        total = _tilt_dependent_total_from_breakdown(breakdown)
        sample = {
            "alpha": float(alpha),
            "tilt_dependent_delta": float(total - base_total),
            "tilt_out_delta": float(
                breakdown.get("tilt_out", 0.0) - base_breakdown.get("tilt_out", 0.0)
            ),
            "bending_tilt_out_delta": float(
                breakdown.get("bending_tilt_out", 0.0)
                - base_breakdown.get("bending_tilt_out", 0.0)
            ),
            "support_radial_before": float(np.mean(current_rad)),
            "support_radial_after": float(np.mean(target_rad)),
            "trace_radial_target": float(trace_rad),
            "positive_branch": bool(np.sign(np.mean(target_rad)) == np.sign(trace_rad)),
        }
        samples.append(sample)
        if best is None or sample["tilt_dependent_delta"] < float(
            best["tilt_dependent_delta"]
        ):
            best = sample
    return {
        "available": True,
        "trace_radial_target": float(trace_rad),
        "samples": samples,
        "best_sample": best or {},
        "energy_lowering_positive_branch": bool(
            best is not None
            and float(best.get("tilt_dependent_delta", 0.0)) < 0.0
            and bool(best.get("positive_branch", False))
        ),
    }


def _energy_breakdown_for_trial_state(
    ctx,
    *,
    positions: np.ndarray | None = None,
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
) -> dict[str, float]:
    snapshot = _capture_state(ctx)
    try:
        if positions is not None:
            pos = np.asarray(positions, dtype=float)
            for row, vid in enumerate(ctx.mesh.vertex_ids):
                ctx.mesh.vertices[int(vid)].position[:] = pos[row]
            ctx.mesh.increment_version()
        if tilts_in is not None:
            ctx.mesh.set_tilts_in_from_array(np.asarray(tilts_in, dtype=float))
        if tilts_out is not None:
            ctx.mesh.set_tilts_out_from_array(np.asarray(tilts_out, dtype=float))
        return {
            k: float(v) for k, v in ctx.minimizer.compute_energy_breakdown().items()
        }
    finally:
        _restore_state(ctx, snapshot)


def _apply_radial_tilt_delta(
    mesh,
    tilts: np.ndarray,
    rows: np.ndarray,
    delta: float,
) -> np.ndarray:
    out = np.asarray(tilts, dtype=float).copy(order="F")
    rows = np.asarray(rows, dtype=int)
    if rows.size == 0 or abs(float(delta)) <= 0.0:
        return out
    _radii, r_hat = radial_unit_vectors(mesh.positions_view())
    out[rows] += float(delta) * r_hat[rows]
    return out


def _apply_trace_height_delta(
    ctx,
    positions: np.ndarray,
    *,
    delta_phi: float,
) -> np.ndarray:
    roles = _scaffold_role_rows(ctx.mesh)
    trace_rows = np.asarray(roles.get("trace", np.zeros(0, dtype=int)), dtype=int)
    disk_rows = np.asarray(roles.get("disk", np.zeros(0, dtype=int)), dtype=int)
    out = np.asarray(positions, dtype=float).copy(order="F")
    if trace_rows.size == 0 or disk_rows.size == 0:
        return out
    radii = np.linalg.norm(out[:, :2], axis=1)
    dr = float(np.mean(radii[trace_rows]) - np.mean(radii[disk_rows]))
    if abs(dr) <= 1.0e-12:
        return out
    out[trace_rows, 2] += float(delta_phi) * dr
    return out


def _breakdown_delta(
    base: dict[str, float],
    trial: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(base) | set(trial))
    return {key: float(trial.get(key, 0.0) - base.get(key, 0.0)) for key in keys}


def _dominant_positive_term(delta: dict[str, float]) -> str:
    if not delta:
        return ""
    key, value = max(delta.items(), key=lambda item: float(item[1]))
    return str(key) if float(value) > 0.0 else ""


def _trace_continuation_landscape_for_context(
    ctx,
    *,
    variant: str,
    geometry: str,
) -> dict[str, Any]:
    roles = _scaffold_role_rows(ctx.mesh)
    trace_rows = np.asarray(roles.get("trace", np.zeros(0, dtype=int)), dtype=int)
    support_rows = np.unique(
        np.concatenate(
            [
                roles.get("support", np.zeros(0, dtype=int)),
                roles.get("release", np.zeros(0, dtype=int)),
            ]
        )
    )
    if support_rows.size == 0:
        support_rows = np.asarray(roles.get("outer", np.zeros(0, dtype=int)), dtype=int)
    base_positions = ctx.mesh.positions_view().copy(order="F")
    base_in = ctx.mesh.tilts_in_view().copy(order="F")
    base_out = ctx.mesh.tilts_out_view().copy(order="F")
    base_breakdown = _energy_breakdown_for_trial_state(
        ctx, positions=base_positions, tilts_in=base_in, tilts_out=base_out
    )
    base_total = _tilt_dependent_total_from_breakdown(base_breakdown)
    trace_rad = _row_radial_mean(ctx.mesh, base_out, trace_rows)
    support_rad = _row_radial_mean(ctx.mesh, base_out, support_rows)
    trace_phi = _role_phi_to_inner_summary(ctx, roles=roles, role="trace")
    scale = max(abs(trace_rad), abs(trace_phi), 1.0e-3)
    sign = 1.0 if trace_rad >= 0.0 else -1.0
    samples: list[dict[str, Any]] = []
    for mode in TRACE_CONTINUATION_LANDSCAPE_MODES:
        for alpha in TRACE_CONTINUATION_LANDSCAPE_ALPHAS:
            delta = sign * float(alpha) * scale
            trial_pos = base_positions
            trial_out = base_out
            if mode in {"trace_tilt", "trace_support_tilt", "trace_tilt_height"}:
                trial_out = _apply_radial_tilt_delta(
                    ctx.mesh, trial_out, trace_rows, delta
                )
            if mode in {"support_tilt", "trace_support_tilt"}:
                trial_out = _apply_radial_tilt_delta(
                    ctx.mesh, trial_out, support_rows, delta
                )
            if mode == "trace_tilt_height":
                trial_pos = _apply_trace_height_delta(ctx, trial_pos, delta_phi=delta)
            trial_breakdown = _energy_breakdown_for_trial_state(
                ctx,
                positions=trial_pos,
                tilts_in=base_in,
                tilts_out=trial_out,
            )
            delta_terms = _breakdown_delta(base_breakdown, trial_breakdown)
            sample = {
                "variant": variant,
                "geometry": geometry,
                "mode": mode,
                "alpha": float(alpha),
                "trace_radial_before": float(trace_rad),
                "support_radial_before": float(support_rad),
                "trace_phi_before": float(trace_phi),
                "tilt_dependent_delta": float(
                    _tilt_dependent_total_from_breakdown(trial_breakdown) - base_total
                ),
                "total_delta": float(sum(delta_terms.values())),
                "term_deltas": delta_terms,
                "dominant_positive_term": _dominant_positive_term(delta_terms),
            }
            samples.append(sample)
    return {
        "variant": variant,
        "geometry": geometry,
        "base_tilt_dependent_energy": float(base_total),
        "base_breakdown": base_breakdown,
        "trace_radial_before": float(trace_rad),
        "support_radial_before": float(support_rad),
        "trace_phi_before": float(trace_phi),
        "samples": samples,
    }


def _trace_continuation_landscape_doc(
    *,
    variant: str,
) -> tuple[dict[str, Any], str, str]:
    base_doc = yaml.safe_load(DEFAULT_FIXTURE.read_text(encoding="utf-8")) or {}
    if variant == "trace_only":
        doc = build_full_physics_trace_fixture(
            base_doc=base_doc,
            lane="physical_edge_full_coupling_trace_eps0005_landscape",
            trace_radius=(7.0 / 15.0) + 0.005,
            planar_geometry=False,
        )
        return doc, "trace_only", "full_coupling_trace_landscape"
    if variant == "fixed_support_runtime_options_gd_fallback":
        doc, _label = _trace_convergence_current_geometry_doc(
            geometry="fixed_support", epsilon=0.005
        )
        doc = _apply_support_ownership_probe_options(
            doc,
            lane="physical_edge_full_coupling_fixed_support_landscape",
            variant="runtime_options_gd_fallback",
        )
        return doc, "fixed_support", "full_coupling_fixed_support_landscape"
    if variant == "gapfill_release_runtime_options_gd_fallback":
        doc, _label = _trace_convergence_current_geometry_doc(
            geometry="gapfill_release", epsilon=0.005
        )
        doc = _apply_support_ownership_probe_options(
            doc,
            lane="physical_edge_full_coupling_gapfill_release_landscape",
            variant="runtime_options_gd_fallback",
        )
        return doc, "gapfill_release", "full_coupling_gapfill_release_landscape"
    raise ValueError(f"unknown landscape variant: {variant}")


def _scaffold_geometry_spacing_doc(spec: dict[str, Any]) -> tuple[dict[str, Any], str]:
    base_doc = yaml.safe_load(DEFAULT_FIXTURE.read_text(encoding="utf-8")) or {}
    label = str(spec["label"])
    geometry = str(spec["geometry"])
    epsilon = float(spec.get("epsilon", 0.005))
    trace_radius = (7.0 / 15.0) + epsilon
    if geometry == "trace_only":
        doc = build_full_physics_trace_fixture(
            base_doc=base_doc,
            lane=f"physical_edge_full_coupling_{label}",
            trace_radius=trace_radius,
            planar_geometry=False,
        )
    elif geometry == "fixed_support":
        doc = build_outer_shell_scaffold_fixture(
            base_doc=base_doc,
            label=f"physical_edge_full_coupling_{label}",
            trace_radius=trace_radius,
            outer_shells=int(spec.get("outer_shells", 3)),
            outer_shells_d=float(spec["outer_shells_d"]),
            planar_geometry=False,
        )
    elif geometry == "gapfill_release":
        doc = build_gap_filled_outer_shell_scaffold_fixture(
            base_doc=base_doc,
            label=f"physical_edge_full_coupling_{label}",
            trace_radius=trace_radius,
            outer_shells=int(spec.get("outer_shells", 3)),
            planar_geometry=False,
        )
    else:
        raise ValueError(f"unknown scaffold spacing geometry: {geometry}")
    doc = _apply_support_ownership_probe_options(
        doc,
        lane=f"physical_edge_full_coupling_{label}",
        variant="runtime_options_gd_fallback",
    )
    gp = dict(doc.get("global_parameters") or {})
    gp["bending_tilt_base_term_reference_mode"] = "current_geometry"
    if spec.get("interface_divergence_mode") is not None:
        gp["bending_tilt_out_interface_divergence_mode"] = str(
            spec["interface_divergence_mode"]
        )
    if spec.get("inner_scaffold_shape_stencil_mode") is not None:
        gp["bending_tilt_in_scaffold_shape_stencil_mode"] = str(
            spec["inner_scaffold_shape_stencil_mode"]
        )
    if spec.get("scaffold_mesh_operation_mode") is not None:
        gp["rim_slope_match_scaffold_mesh_operation_mode"] = str(
            spec["scaffold_mesh_operation_mode"]
        )
    if spec.get("pin_to_circle_preserve_normal_groups") is not None:
        gp["pin_to_circle_mesh_operation_preserve_normal_groups"] = list(
            spec["pin_to_circle_preserve_normal_groups"]
        )
    if spec.get("shape_scaffold_rejected_step_fallback") is not None:
        gp["shape_scaffold_rejected_step_fallback"] = str(
            spec["shape_scaffold_rejected_step_fallback"]
        )
    doc["global_parameters"] = gp
    return doc, label


def _trace_continuation_landscape_probe(
    *,
    tmpdir: Path,
    protocol: tuple[str, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    interface_rows: list[dict[str, Any]] = []
    conditioning_rows: list[dict[str, Any]] = []
    for variant in TRACE_CONTINUATION_LANDSCAPE_VARIANTS:
        doc, geometry, label = _trace_continuation_landscape_doc(variant=variant)
        mesh_path = _write_temp_fixture(doc, tmpdir, label)
        ctx, summary = _run_protocol_summary(mesh_path=mesh_path, protocol=protocol)
        landscape = _trace_continuation_landscape_for_context(
            ctx, variant=variant, geometry=geometry
        )
        landscape.update(
            {
                "thetaB_value": _as_float(summary.get("thetaB_value")),
                "interface_summary": dict(summary.get("interface_summary", {})),
                "stop_reason": str(
                    summary.get("leaflet_relaxation_stats", {}).get("stop_reason") or ""
                ),
            }
        )
        rows.append(landscape)
        decompositions = [
            _bending_tilt_out_interface_decomposition(ctx, reference_mode=mode)
            for mode in BENDING_TILT_OUT_INTERFACE_REFERENCE_MODES
        ]
        current = next(
            (
                row
                for row in decompositions
                if row.get("reference_mode") == "current_geometry"
            ),
            {},
        )
        flat = next(
            (
                row
                for row in decompositions
                if str(row.get("reference_mode")).lower() == "flat_reference_zero_j0"
            ),
            {},
        )
        current_total = float(current.get("totals", {}).get("total_energy") or 0.0)
        flat_total = float(flat.get("totals", {}).get("total_energy") or 0.0)
        interface_rows.append(
            {
                "variant": variant,
                "geometry": geometry,
                "thetaB_value": _as_float(summary.get("thetaB_value")),
                "direct_t_out": _as_float(
                    summary.get("interface_summary", {}).get("direct_t_out")
                ),
                "direct_phi": _as_float(
                    summary.get("interface_summary", {}).get("direct_phi")
                ),
                "energy_breakdown": dict(summary.get("energy_breakdown", {})),
                "elastic_total_from_breakdown": _as_float(
                    summary.get("elastic_total_from_breakdown")
                ),
                "stop_reason": str(
                    summary.get("leaflet_relaxation_stats", {}).get("stop_reason") or ""
                ),
                "decompositions": decompositions,
                "current_minus_flat_total": float(current_total - flat_total),
                "current_vs_flat_ratio": float(current_total / flat_total)
                if abs(flat_total) > 1.0e-15
                else 0.0,
            }
        )
        conditioning_rows.append(
            {
                "variant": variant,
                "geometry": geometry,
                "thetaB_value": _as_float(summary.get("thetaB_value")),
                "direct_t_out": _as_float(
                    summary.get("interface_summary", {}).get("direct_t_out")
                ),
                "direct_phi": _as_float(
                    summary.get("interface_summary", {}).get("direct_phi")
                ),
                "energy_breakdown": dict(summary.get("energy_breakdown", {})),
                "elastic_total_from_breakdown": _as_float(
                    summary.get("elastic_total_from_breakdown")
                ),
                "stop_reason": str(
                    summary.get("leaflet_relaxation_stats", {}).get("stop_reason") or ""
                ),
                "conditioning": _bending_tilt_out_divergence_conditioning(ctx),
            }
        )
    suppressing_terms: dict[str, int] = {}
    lowering_count = 0
    for row in rows:
        for sample in row.get("samples", []):
            if float(sample.get("tilt_dependent_delta") or 0.0) < 0.0:
                lowering_count += 1
            term = str(sample.get("dominant_positive_term") or "")
            if term:
                suppressing_terms[term] = suppressing_terms.get(term, 0) + 1
    dominant = (
        max(suppressing_terms.items(), key=lambda item: item[1])[0]
        if suppressing_terms
        else ""
    )
    return {
        "rows": rows,
        "summary": {
            "energy_lowering_sample_count": int(lowering_count),
            "dominant_positive_terms": suppressing_terms,
            "most_common_suppressing_term": dominant,
        },
        "bending_tilt_out_interface_audit": {
            "rows": interface_rows,
            "summary": _bending_tilt_out_interface_summary(interface_rows),
        },
        "bending_tilt_out_divergence_conditioning_audit": {
            "rows": conditioning_rows,
            "summary": _bending_tilt_out_divergence_conditioning_summary(
                conditioning_rows
            ),
        },
    }


def _triangle_role_counts(
    tri_rows: np.ndarray, roles: dict[str, np.ndarray]
) -> dict[str, int]:
    if tri_rows.size == 0:
        return {
            "trace_touching": 0,
            "support_touching": 0,
            "release_touching": 0,
            "far_or_other": 0,
        }
    trace_rows = np.asarray(roles.get("trace", np.zeros(0, dtype=int)), dtype=int)
    support_rows = np.asarray(roles.get("support", np.zeros(0, dtype=int)), dtype=int)
    release_rows = np.asarray(roles.get("release", np.zeros(0, dtype=int)), dtype=int)
    trace_mask = (
        np.any(np.isin(tri_rows, trace_rows), axis=1)
        if trace_rows.size
        else np.zeros(len(tri_rows), dtype=bool)
    )
    support_mask = (
        np.any(np.isin(tri_rows, support_rows), axis=1)
        if support_rows.size
        else np.zeros(len(tri_rows), dtype=bool)
    )
    release_mask = (
        np.any(np.isin(tri_rows, release_rows), axis=1)
        if release_rows.size
        else np.zeros(len(tri_rows), dtype=bool)
    )
    classified = trace_mask | support_mask | release_mask
    return {
        "trace_touching": int(np.sum(trace_mask)),
        "support_touching": int(np.sum(~trace_mask & support_mask)),
        "release_touching": int(np.sum(~trace_mask & ~support_mask & release_mask)),
        "far_or_other": int(np.sum(~classified)),
    }


def _triangle_role_masks(
    tri_rows: np.ndarray, roles: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if tri_rows.size == 0:
        empty = np.zeros(0, dtype=bool)
        return {
            "trace_touching": empty,
            "support_touching": empty,
            "release_touching": empty,
            "far_or_other": empty,
        }
    trace_rows = np.asarray(roles.get("trace", np.zeros(0, dtype=int)), dtype=int)
    support_rows = np.asarray(roles.get("support", np.zeros(0, dtype=int)), dtype=int)
    release_rows = np.asarray(roles.get("release", np.zeros(0, dtype=int)), dtype=int)
    trace_mask = (
        np.any(np.isin(tri_rows, trace_rows), axis=1)
        if trace_rows.size
        else np.zeros(len(tri_rows), dtype=bool)
    )
    support_mask = (
        np.any(np.isin(tri_rows, support_rows), axis=1)
        if support_rows.size
        else np.zeros(len(tri_rows), dtype=bool)
    )
    release_mask = (
        np.any(np.isin(tri_rows, release_rows), axis=1)
        if release_rows.size
        else np.zeros(len(tri_rows), dtype=bool)
    )
    support_only = ~trace_mask & support_mask
    release_only = ~trace_mask & ~support_only & release_mask
    return {
        "trace_touching": trace_mask,
        "support_touching": support_only,
        "release_touching": release_only,
        "far_or_other": ~(trace_mask | support_only | release_only),
    }


def _row_role_labels(n_rows: int, roles: dict[str, np.ndarray]) -> np.ndarray:
    labels = np.full(n_rows, "other", dtype=object)
    for role in ("trace", "support", "release", "disk", "outer"):
        rows = np.asarray(roles.get(role, np.zeros(0, dtype=int)), dtype=int)
        if rows.size:
            labels[rows] = role
    return labels


def _triangle_geometry_arrays(
    positions: np.ndarray, tri_rows: np.ndarray
) -> dict[str, np.ndarray]:
    if tri_rows.size == 0:
        return {
            "area": np.zeros(0, dtype=float),
            "min_edge": np.zeros(0, dtype=float),
            "max_edge": np.zeros(0, dtype=float),
            "aspect": np.zeros(0, dtype=float),
        }
    pts = np.asarray(positions, dtype=float)[tri_rows]
    e01 = np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1)
    e12 = np.linalg.norm(pts[:, 2] - pts[:, 1], axis=1)
    e20 = np.linalg.norm(pts[:, 0] - pts[:, 2], axis=1)
    edges = np.column_stack([e01, e12, e20])
    area = 0.5 * np.linalg.norm(
        np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]), axis=1
    )
    max_edge = np.max(edges, axis=1)
    min_edge = np.min(edges, axis=1)
    aspect = (max_edge**2) / np.maximum(area, 1.0e-15)
    return {
        "area": area,
        "min_edge": min_edge,
        "max_edge": max_edge,
        "aspect": aspect,
    }


def _array_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "abs_mean": 0.0, "max_abs": 0.0, "min": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "abs_mean": float(np.mean(np.abs(arr))),
        "max_abs": float(np.max(np.abs(arr))),
        "min": float(np.min(arr)),
    }


def _bending_tilt_out_interface_decomposition(
    ctx,
    *,
    reference_mode: str,
) -> dict[str, Any]:
    return _bending_tilt_leaflet_interface_decomposition(
        ctx,
        leaflet="out",
        reference_mode=reference_mode,
    )


def _bending_tilt_leaflet_interface_decomposition(
    ctx,
    *,
    leaflet: str,
    reference_mode: str | None = None,
) -> dict[str, Any]:
    mesh = ctx.mesh
    gp = mesh.global_parameters
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    roles = _scaffold_role_rows(mesh)
    leaflet = str(leaflet)
    if leaflet == "in":
        module = bending_tilt_in_module
        cache_tag = "in"
        tilts = mesh.tilts_in_view()
        kappa_key = "bending_modulus_in"
        div_sign = -1.0
    elif leaflet == "out":
        module = bending_tilt_out_module
        cache_tag = "out"
        tilts = mesh.tilts_out_view()
        kappa_key = "bending_modulus_out"
        div_sign = 1.0
    else:
        raise ValueError(f"unknown bending-tilt leaflet: {leaflet}")

    overrides = {}
    if reference_mode is not None:
        overrides[f"bending_tilt_base_term_reference_mode_{cache_tag}"] = reference_mode
    present, old = _temporary_overrides(gp, overrides)
    try:
        payload = module._leaflet._leaflet_triangle_payload(
            mesh,
            gp,
            positions=positions,
            index_map=index_map,
            cache_tag=cache_tag,
            ctx=ctx.minimizer.energy_context(),
        )
        tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
        tri_rows_full = np.asarray(payload["tri_rows_full"], dtype=np.int32)
        tri_keep = np.asarray(payload["tri_keep"], dtype=bool)
        if tri_rows.size == 0:
            return {
                "leaflet": cache_tag,
                "reference_mode": reference_mode or "",
                "triangle_counts": {
                    "full": int(len(tri_rows_full)),
                    "kept": 0,
                    "removed": int(np.sum(~tri_keep)) if tri_keep.size else 0,
                    **_triangle_role_counts(tri_rows, roles),
                },
                "role_rows": {
                    role: int(np.asarray(rows, dtype=int).size)
                    for role, rows in roles.items()
                },
                "roles": {},
                "totals": {
                    "base_energy": 0.0,
                    "divergence_energy": 0.0,
                    "cross_energy": 0.0,
                    "total_energy": 0.0,
                    "area": 0.0,
                },
            }

        g0_use = payload["g0"]
        g1_use = payload["g1"]
        g2_use = payload["g2"]
        transport_model = _resolve_transport_model(
            gp.get("tilt_transport_model", "ambient_v1")
            if gp is not None
            else "ambient_v1"
        )
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
        div_eval_tri = float(div_sign) * np.asarray(div_tri, dtype=float)
        div_eval_tri = module._leaflet._apply_inner_divergence_update_mode(
            mesh,
            gp,
            positions=positions,
            tri_rows=tri_rows,
            cache_tag=cache_tag,
            div_term=div_eval_tri,
        )
        if not module._leaflet._use_inner_recovered_divergence(gp, cache_tag=cache_tag):
            div_eval_tri, _reconstruction_stats = (
                module._leaflet._outer_trace_reconstructed_divergence(
                    mesh,
                    gp,
                    cache_tag=cache_tag,
                    tri_rows=tri_rows,
                    div_term=div_eval_tri,
                )
            )
        _vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
            mesh,
            positions,
            tri_rows,
            np.asarray(payload["weights"], dtype=float),
            index_map,
            cache_token=(
                f"thetaB_cadence_btl_{cache_tag}_interface_"
                f"{reference_mode or 'current'}"
            ),
            compute_vertex_areas=False,
        )
        static_payload = module._leaflet._leaflet_static_tilt_payload(
            mesh,
            gp,
            positions=positions,
            index_map=index_map,
            k_vecs=np.asarray(payload["k_vecs"], dtype=float),
            vertex_areas_vor=np.asarray(payload["vertex_areas_vor"], dtype=float),
            tri_rows=tri_rows,
            kappa_key=kappa_key,
            cache_tag=cache_tag,
        )
        base_tri = np.asarray(static_payload["base_tri"], dtype=float)
        base_term = np.asarray(static_payload["base_term"], dtype=float)
        kappa_tri = np.asarray(static_payload["kappa_tri"], dtype=float)
        area_cols = np.column_stack([va0_eff, va1_eff, va2_eff])
        div_cols = div_eval_tri[:, None]
        base_energy_cols = 0.5 * kappa_tri * (base_tri**2) * area_cols
        div_energy_cols = 0.5 * kappa_tri * (div_cols**2) * area_cols
        cross_energy_cols = kappa_tri * base_tri * div_cols * area_cols
        total_cols = base_energy_cols + div_energy_cols + cross_energy_cols
        role_masks = {
            "trace_touching": np.any(
                np.isin(tri_rows, np.asarray(roles.get("trace", []), dtype=int)),
                axis=1,
            )
            if np.asarray(roles.get("trace", []), dtype=int).size
            else np.zeros(len(tri_rows), dtype=bool),
            "support_touching": np.any(
                np.isin(tri_rows, np.asarray(roles.get("support", []), dtype=int)),
                axis=1,
            )
            if np.asarray(roles.get("support", []), dtype=int).size
            else np.zeros(len(tri_rows), dtype=bool),
            "release_touching": np.any(
                np.isin(tri_rows, np.asarray(roles.get("release", []), dtype=int)),
                axis=1,
            )
            if np.asarray(roles.get("release", []), dtype=int).size
            else np.zeros(len(tri_rows), dtype=bool),
        }
        role_masks["support_touching"] = (
            ~role_masks["trace_touching"] & role_masks["support_touching"]
        )
        role_masks["release_touching"] = (
            ~role_masks["trace_touching"]
            & ~role_masks["support_touching"]
            & role_masks["release_touching"]
        )
        classified = (
            role_masks["trace_touching"]
            | role_masks["support_touching"]
            | role_masks["release_touching"]
        )
        role_masks["far_or_other"] = ~classified

        role_rows: dict[str, Any] = {}
        for role, mask in role_masks.items():
            if not np.any(mask):
                role_rows[role] = {
                    "triangle_count": 0,
                    "area": 0.0,
                    "base_energy": 0.0,
                    "divergence_energy": 0.0,
                    "cross_energy": 0.0,
                    "total_energy": 0.0,
                    "base_term_mean": 0.0,
                    "divergence_mean": 0.0,
                    "base_divergence_product_mean": 0.0,
                }
                continue
            base_vals = base_tri[mask].reshape(-1)
            div_vals = np.repeat(div_eval_tri[mask], 3)
            role_rows[role] = {
                "triangle_count": int(np.sum(mask)),
                "area": float(np.sum(area_cols[mask])),
                "base_energy": float(np.sum(base_energy_cols[mask])),
                "divergence_energy": float(np.sum(div_energy_cols[mask])),
                "cross_energy": float(np.sum(cross_energy_cols[mask])),
                "total_energy": float(np.sum(total_cols[mask])),
                "base_term_mean": float(np.mean(base_vals)) if base_vals.size else 0.0,
                "divergence_mean": float(np.mean(div_vals)) if div_vals.size else 0.0,
                "base_divergence_product_mean": float(np.mean(base_vals * div_vals))
                if base_vals.size
                else 0.0,
            }
        totals = {
            "base_energy": float(np.sum(base_energy_cols)),
            "divergence_energy": float(np.sum(div_energy_cols)),
            "cross_energy": float(np.sum(cross_energy_cols)),
            "total_energy": float(np.sum(total_cols)),
            "area": float(np.sum(area_cols)),
            "base_term_abs_mean": float(np.mean(np.abs(base_term)))
            if base_term.size
            else 0.0,
            "divergence_mean": float(np.mean(div_eval_tri))
            if div_eval_tri.size
            else 0.0,
        }
        return {
            "leaflet": cache_tag,
            "reference_mode": reference_mode or "",
            "triangle_counts": {
                "full": int(len(tri_rows_full)),
                "kept": int(len(tri_rows)),
                "removed": int(np.sum(~tri_keep)) if tri_keep.size else 0,
                **_triangle_role_counts(tri_rows, roles),
            },
            "role_rows": {
                role: int(np.asarray(rows, dtype=int).size)
                for role, rows in roles.items()
            },
            "roles": role_rows,
            "totals": totals,
        }
    finally:
        _restore_overrides(gp, present, old)


def _bending_tilt_out_interface_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    largest_cross_role = ""
    largest_cross_value = 0.0
    for row in rows:
        for decomposition in row.get("decompositions", []):
            if decomposition.get("reference_mode") != "current_geometry":
                continue
            for role, role_data in decomposition.get("roles", {}).items():
                value = abs(float(role_data.get("cross_energy") or 0.0))
                if value > largest_cross_value:
                    largest_cross_value = value
                    largest_cross_role = str(role)
    return {
        "reference_modes": list(BENDING_TILT_OUT_INTERFACE_REFERENCE_MODES),
        "largest_current_geometry_cross_role": largest_cross_role,
        "largest_current_geometry_cross_abs": float(largest_cross_value),
    }


def _bending_tilt_out_divergence_conditioning(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    gp = mesh.global_parameters
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    roles = _scaffold_role_rows(mesh)
    payload = bending_tilt_out_module._leaflet._leaflet_triangle_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        cache_tag="out",
        ctx=ctx.minimizer.energy_context(),
    )
    tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
    tri_rows_full = np.asarray(payload["tri_rows_full"], dtype=np.int32)
    tri_keep = np.asarray(payload["tri_keep"], dtype=bool)
    if tri_rows.size == 0:
        return {
            "triangle_counts": {
                "full": int(len(tri_rows_full)),
                "kept": 0,
                "removed": int(np.sum(~tri_keep)) if tri_keep.size else 0,
            },
            "roles": {},
        }

    transport_model = _resolve_transport_model(
        gp.get("tilt_transport_model", "ambient_v1") if gp is not None else "ambient_v1"
    )
    tilts_out = mesh.tilts_out_view()
    g0_use = payload["g0"]
    g1_use = payload["g1"]
    g2_use = payload["g2"]
    if g0_use is not None and g1_use is not None and g2_use is not None:
        g0 = np.asarray(g0_use, dtype=float)
        g1 = np.asarray(g1_use, dtype=float)
        g2 = np.asarray(g2_use, dtype=float)
        div_tri = compute_divergence_from_basis(
            mesh=mesh,
            tilts=tilts_out,
            tri_rows=tri_rows,
            g0=g0,
            g1=g1,
            g2=g2,
            positions=positions,
            transport_model=transport_model,
        )
    else:
        div_tri, _, g0, g1, g2 = p1_triangle_divergence(
            mesh=mesh,
            positions=positions,
            tilts=tilts_out,
            tri_rows=tri_rows,
            transport_model=transport_model,
        )
        g0 = np.asarray(g0, dtype=float)
        g1 = np.asarray(g1, dtype=float)
        g2 = np.asarray(g2, dtype=float)

    geom = _triangle_geometry_arrays(positions, tri_rows)
    basis_norms = np.column_stack(
        [
            np.linalg.norm(g0, axis=1),
            np.linalg.norm(g1, axis=1),
            np.linalg.norm(g2, axis=1),
        ]
    )
    corner_components = np.column_stack(
        [
            np.einsum("ij,ij->i", tilts_out[tri_rows[:, 0]], g0),
            np.einsum("ij,ij->i", tilts_out[tri_rows[:, 1]], g1),
            np.einsum("ij,ij->i", tilts_out[tri_rows[:, 2]], g2),
        ]
    )
    corner_tilt_norms = np.linalg.norm(tilts_out[tri_rows], axis=2)
    row_labels = _row_role_labels(len(mesh.vertex_ids), roles)
    tri_masks = _triangle_role_masks(tri_rows, roles)
    role_summaries: dict[str, Any] = {}
    for role, mask in tri_masks.items():
        if not np.any(mask):
            role_summaries[role] = {
                "triangle_count": 0,
                "area": _array_summary(np.zeros(0, dtype=float)),
                "min_edge": _array_summary(np.zeros(0, dtype=float)),
                "aspect": _array_summary(np.zeros(0, dtype=float)),
                "basis_norm": _array_summary(np.zeros(0, dtype=float)),
                "divergence": _array_summary(np.zeros(0, dtype=float)),
                "corner_component": _array_summary(np.zeros(0, dtype=float)),
                "corner_tilt_norm": _array_summary(np.zeros(0, dtype=float)),
                "corner_components_by_row_role": {},
            }
            continue
        selected_rows = tri_rows[mask]
        selected_components = corner_components[mask]
        selected_labels = row_labels[selected_rows]
        by_corner_role: dict[str, dict[str, float]] = {}
        for corner_role in ("trace", "support", "release", "disk", "outer", "other"):
            corner_mask = selected_labels == corner_role
            by_corner_role[corner_role] = _array_summary(
                selected_components[corner_mask]
            )
        role_summaries[role] = {
            "triangle_count": int(np.sum(mask)),
            "area": _array_summary(geom["area"][mask]),
            "min_edge": _array_summary(geom["min_edge"][mask]),
            "aspect": _array_summary(geom["aspect"][mask]),
            "basis_norm": _array_summary(basis_norms[mask]),
            "divergence": _array_summary(div_tri[mask]),
            "corner_component": _array_summary(selected_components),
            "corner_tilt_norm": _array_summary(corner_tilt_norms[mask]),
            "corner_components_by_row_role": by_corner_role,
        }
    return {
        "triangle_counts": {
            "full": int(len(tri_rows_full)),
            "kept": int(len(tri_rows)),
            "removed": int(np.sum(~tri_keep)) if tri_keep.size else 0,
            **_triangle_role_counts(tri_rows, roles),
        },
        "role_rows": {
            role: int(np.asarray(rows, dtype=int).size) for role, rows in roles.items()
        },
        "roles": role_summaries,
    }


def _bending_tilt_out_divergence_conditioning_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    max_basis_role = ""
    max_basis_value = 0.0
    max_div_role = ""
    max_div_value = 0.0
    for row in rows:
        for role, data in row.get("conditioning", {}).get("roles", {}).items():
            basis = float(data.get("basis_norm", {}).get("max_abs") or 0.0)
            div = float(data.get("divergence", {}).get("max_abs") or 0.0)
            if basis > max_basis_value:
                max_basis_value = basis
                max_basis_role = f"{row.get('variant')}:{role}"
            if div > max_div_value:
                max_div_value = div
                max_div_role = f"{row.get('variant')}:{role}"
    return {
        "max_basis_norm_role": max_basis_role,
        "max_basis_norm": float(max_basis_value),
        "max_divergence_role": max_div_role,
        "max_divergence": float(max_div_value),
    }


def _bending_tilt_out_divergence_ablation(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    gp = mesh.global_parameters
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    roles = _scaffold_role_rows(mesh)
    payload = bending_tilt_out_module._leaflet._leaflet_triangle_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        cache_tag="out",
        ctx=ctx.minimizer.energy_context(),
    )
    tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
    if tri_rows.size == 0:
        return {
            "rows": [],
            "summary": {"best_energy_delta_label": "", "best_energy_delta": 0.0},
        }

    transport_model = _resolve_transport_model(
        gp.get("tilt_transport_model", "ambient_v1") if gp is not None else "ambient_v1"
    )
    tilts_out = mesh.tilts_out_view()
    g0_use = payload["g0"]
    g1_use = payload["g1"]
    g2_use = payload["g2"]
    if g0_use is not None and g1_use is not None and g2_use is not None:
        g0 = np.asarray(g0_use, dtype=float)
        g1 = np.asarray(g1_use, dtype=float)
        g2 = np.asarray(g2_use, dtype=float)
        div_tri = compute_divergence_from_basis(
            mesh=mesh,
            tilts=tilts_out,
            tri_rows=tri_rows,
            g0=g0,
            g1=g1,
            g2=g2,
            positions=positions,
            transport_model=transport_model,
        )
    else:
        div_tri, _, g0, g1, g2 = p1_triangle_divergence(
            mesh=mesh,
            positions=positions,
            tilts=tilts_out,
            tri_rows=tri_rows,
            transport_model=transport_model,
        )
        g0 = np.asarray(g0, dtype=float)
        g1 = np.asarray(g1, dtype=float)
        g2 = np.asarray(g2, dtype=float)

    _vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        np.asarray(payload["weights"], dtype=float),
        index_map,
        cache_token="thetaB_cadence_btl_out_divergence_ablation",
        compute_vertex_areas=False,
    )
    static_payload = bending_tilt_out_module._leaflet._leaflet_static_tilt_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        k_vecs=np.asarray(payload["k_vecs"], dtype=float),
        vertex_areas_vor=np.asarray(payload["vertex_areas_vor"], dtype=float),
        tri_rows=tri_rows,
        kappa_key="bending_modulus_out",
        cache_tag="out",
    )
    kappa_tri = np.asarray(static_payload["kappa_tri"], dtype=float)
    area_cols = np.column_stack([va0_eff, va1_eff, va2_eff])
    coeff = np.sum(kappa_tri * area_cols, axis=1)
    role_masks = _triangle_role_masks(tri_rows, roles)
    row_labels = _row_role_labels(len(mesh.vertex_ids), roles)
    corner_components = np.column_stack(
        [
            np.einsum("ij,ij->i", tilts_out[tri_rows[:, 0]], g0),
            np.einsum("ij,ij->i", tilts_out[tri_rows[:, 1]], g1),
            np.einsum("ij,ij->i", tilts_out[tri_rows[:, 2]], g2),
        ]
    )
    corner_labels = row_labels[tri_rows]

    def _self_energy(div_values: np.ndarray) -> tuple[float, float]:
        values = np.asarray(div_values, dtype=float)
        energy = 0.5 * coeff * values**2
        trace_mask = role_masks["trace_touching"]
        return (
            float(np.sum(energy)),
            float(np.sum(energy[trace_mask])) if np.any(trace_mask) else 0.0,
        )

    production_total, production_trace = _self_energy(div_tri)
    samples: list[dict[str, Any]] = [
        {
            "label": "production",
            "total_divergence_energy": production_total,
            "trace_divergence_energy": production_trace,
            "energy_delta": 0.0,
            "trace_energy_delta": 0.0,
        }
    ]

    def _add_sample(label: str, div_values: np.ndarray) -> None:
        total, trace_total = _self_energy(div_values)
        samples.append(
            {
                "label": label,
                "total_divergence_energy": float(total),
                "trace_divergence_energy": float(trace_total),
                "energy_delta": float(total - production_total),
                "trace_energy_delta": float(trace_total - production_trace),
            }
        )

    div_zero_support = np.asarray(div_tri, dtype=float).copy()
    div_zero_support[role_masks["support_touching"]] = 0.0
    _add_sample("zero_support_touching_triangles", div_zero_support)

    div_zero_scaffold = np.asarray(div_tri, dtype=float).copy()
    div_zero_scaffold[
        role_masks["support_touching"] | role_masks["release_touching"]
    ] = 0.0
    _add_sample("zero_support_release_touching_triangles", div_zero_scaffold)

    for omitted_role in ("support", "release", "outer", "other"):
        kept_components = np.where(
            corner_labels == omitted_role, 0.0, corner_components
        )
        _add_sample(
            f"omit_{omitted_role}_row_corner_components",
            np.sum(kept_components, axis=1),
        )

    trace_mask = role_masks["trace_touching"]
    support_mask = role_masks["support_touching"]
    far_mask = role_masks["far_or_other"]
    if np.any(trace_mask) and np.any(support_mask):
        div_support_mean = np.asarray(div_tri, dtype=float).copy()
        div_support_mean[trace_mask] = float(np.mean(div_tri[support_mask]))
        _add_sample("trace_divergence_from_support_mean", div_support_mean)
    if np.any(trace_mask) and np.any(far_mask):
        div_far_mean = np.asarray(div_tri, dtype=float).copy()
        div_far_mean[trace_mask] = float(np.mean(div_tri[far_mask]))
        _add_sample("trace_divergence_from_far_mean", div_far_mean)

    best = min(samples[1:] or samples, key=lambda row: float(row["energy_delta"]))
    return {
        "rows": samples,
        "summary": {
            "production_total_divergence_energy": float(production_total),
            "production_trace_divergence_energy": float(production_trace),
            "best_energy_delta_label": str(best.get("label") or ""),
            "best_energy_delta": _as_float(best.get("energy_delta")),
            "best_trace_energy_delta": _as_float(best.get("trace_energy_delta")),
        },
    }


def _conditioning_role_metric(
    conditioning: dict[str, Any],
    role: str,
    path: tuple[str, str],
) -> float:
    data = conditioning.get("roles", {}).get(role, {})
    node: Any = data
    for key in path:
        if not isinstance(node, dict):
            return 0.0
        node = node.get(key)
    return _as_float(node)


def _outer_tilt_gradient_role_probe(ctx, *, theta: float) -> dict[str, Any]:
    mesh = ctx.mesh
    gp = mesh.global_parameters
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    resolver = ctx.minimizer.param_resolver
    tilt_grad = np.zeros_like(mesh.tilts_out_view())
    btl_grad = np.zeros_like(mesh.tilts_out_view())
    tilt_shape = np.zeros_like(positions)
    btl_shape = np.zeros_like(positions)
    tilt_energy = tilt_out_module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=tilt_shape,
        ctx=ctx.minimizer.energy_context(),
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=tilt_grad,
    )
    btl_energy = bending_tilt_out_module.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=btl_shape,
        ctx=ctx.minimizer.energy_context(),
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=btl_grad,
    )
    combined = tilt_grad + btl_grad
    update = _one_step_outer_update_array(ctx, theta=float(theta))
    return {
        "tilt_out_energy": float(tilt_energy),
        "bending_tilt_out_energy": float(btl_energy),
        "tilt_out_gradient": _role_vector_summaries(ctx, tilt_grad),
        "bending_tilt_out_gradient": _role_vector_summaries(ctx, btl_grad),
        "combined_gradient": _role_vector_summaries(ctx, combined),
        "descent_direction": _role_vector_summaries(ctx, -combined),
        "first_update": _role_vector_summaries(ctx, update),
    }


def _set_radial_tilt_target(
    mesh, tilts: np.ndarray, rows: np.ndarray, target: float
) -> np.ndarray:
    out = np.asarray(tilts, dtype=float).copy(order="F")
    rows = np.asarray(rows, dtype=int)
    if rows.size == 0:
        return out
    _radii, r_hat = radial_unit_vectors(mesh.positions_view())
    current_radial = np.einsum("ij,ij->i", out[rows], r_hat[rows])
    out[rows] += (float(target) - current_radial)[:, None] * r_hat[rows]
    return out


def _high_trace_seed_replay(
    ctx,
    *,
    mesh_path: Path,
    protocol: tuple[str, ...],
    target_radial: float = 0.15,
    steps: int = 60,
) -> dict[str, Any]:
    snapshot = _capture_state(ctx)
    roles = _scaffold_role_rows(ctx.mesh)
    trace_rows = np.asarray(roles.get("trace", np.zeros(0, dtype=int)), dtype=int)
    try:
        seeded_out = _set_radial_tilt_target(
            ctx.mesh,
            ctx.mesh.tilts_out_view(),
            trace_rows,
            target=float(target_radial),
        )
        ctx.mesh.set_tilts_out_from_array(seeded_out)
        seeded_breakdown = {
            key: float(value)
            for key, value in ctx.minimizer.compute_energy_breakdown().items()
        }
        seeded_fields = _scaffold_role_field_summary(ctx)
        _relax_leaflets_for_steps(ctx, int(steps))
        relaxed = _collect_live_summary(ctx=ctx, mesh_path=mesh_path, protocol=protocol)
        return {
            "target_radial": float(target_radial),
            "steps": int(steps),
            "seeded_trace_t_out": _as_float(
                seeded_fields.get("trace", {}).get("tilt_out_radial_mean")
            ),
            "seeded_bending_tilt_out": _as_float(
                seeded_breakdown.get("bending_tilt_out")
            ),
            "seeded_tilt_out": _as_float(seeded_breakdown.get("tilt_out")),
            "relaxed_thetaB": _as_float(relaxed.get("thetaB_value")),
            "relaxed_direct_t_out": _as_float(
                relaxed.get("interface_summary", {}).get("direct_t_out")
            ),
            "relaxed_direct_phi": _as_float(
                relaxed.get("interface_summary", {}).get("direct_phi")
            ),
            "relaxed_bending_tilt_out": _as_float(
                relaxed.get("energy_breakdown", {}).get("bending_tilt_out")
            ),
            "relaxed_tilt_out": _as_float(
                relaxed.get("energy_breakdown", {}).get("tilt_out")
            ),
            "stop_reason": str(
                relaxed.get("leaflet_relaxation_stats", {}).get("stop_reason") or ""
            ),
        }
    finally:
        _restore_state(ctx, snapshot)


def _high_trace_constraint_projection_probe(
    ctx,
    *,
    target_radial: float = 0.15,
) -> dict[str, Any]:
    snapshot = _capture_state(ctx)
    roles = _scaffold_role_rows(ctx.mesh)
    trace_rows = np.asarray(roles.get("trace", np.zeros(0, dtype=int)), dtype=int)
    try:
        seeded_out = _set_radial_tilt_target(
            ctx.mesh,
            ctx.mesh.tilts_out_view(),
            trace_rows,
            target=float(target_radial),
        )
        ctx.mesh.set_tilts_out_from_array(seeded_out)
        before = _scaffold_role_field_summary(ctx)
        if hasattr(ctx.minimizer.constraint_manager, "enforce_tilt_constraints"):
            ctx.minimizer.constraint_manager.enforce_tilt_constraints(
                ctx.mesh,
                global_params=ctx.mesh.global_parameters,
            )
        after = _scaffold_role_field_summary(ctx)
        before_trace = before.get("trace", {})
        after_trace = after.get("trace", {})
        return {
            "target_radial": float(target_radial),
            "before_trace_t_out": _as_float(before_trace.get("tilt_out_radial_mean")),
            "before_trace_phi": _as_float(before_trace.get("phi_to_inner")),
            "after_trace_t_out": _as_float(after_trace.get("tilt_out_radial_mean")),
            "after_trace_phi": _as_float(after_trace.get("phi_to_inner")),
            "projection_delta": _as_float(after_trace.get("tilt_out_radial_mean"))
            - _as_float(before_trace.get("tilt_out_radial_mean")),
            "after_gap_to_phi": _as_float(after_trace.get("tilt_out_radial_mean"))
            - _as_float(after_trace.get("phi_to_inner")),
        }
    finally:
        _restore_state(ctx, snapshot)


def _role_constraint_tag_summary(ctx) -> dict[str, Any]:
    roles = _scaffold_role_rows(ctx.mesh)
    out: dict[str, Any] = {}
    for role, rows in roles.items():
        rows = np.asarray(rows, dtype=int)
        constraints: dict[str, int] = {}
        fixed = 0
        tilt_fixed_out = 0
        for row in rows:
            vertex = ctx.mesh.vertices[int(ctx.mesh.vertex_ids[int(row)])]
            opts = getattr(vertex, "options", {}) or {}
            for constraint in list(opts.get("constraints") or []):
                key = str(constraint)
                constraints[key] = constraints.get(key, 0) + 1
            fixed += int(bool(getattr(vertex, "fixed", False)))
            tilt_fixed_out += int(bool(getattr(vertex, "tilt_fixed_out", False)))
        out[role] = {
            "count": int(rows.size),
            "fixed_count": int(fixed),
            "tilt_fixed_out_count": int(tilt_fixed_out),
            "constraints": constraints,
        }
    return out


def _shape_gradient_role_probe(ctx) -> dict[str, Any]:
    roles = _scaffold_role_rows(ctx.mesh)
    energy, grad = ctx.minimizer.compute_energy_and_gradient_array()
    grad = np.asarray(grad, dtype=float)
    pos = ctx.mesh.positions_view()
    radial = np.zeros_like(pos)
    radii = np.linalg.norm(pos[:, :2], axis=1)
    nonzero = radii > 1.0e-12
    radial[nonzero, :2] = pos[nonzero, :2] / radii[nonzero, None]
    out: dict[str, Any] = {"energy": _as_float(energy), "roles": {}}
    for role, rows in roles.items():
        rows = np.asarray(rows, dtype=int)
        if rows.size == 0:
            out["roles"][role] = {
                "count": 0,
                "norm": 0.0,
                "z_mean": 0.0,
                "descent_z_mean": 0.0,
                "radial_mean": 0.0,
                "descent_radial_mean": 0.0,
            }
            continue
        role_grad = grad[rows]
        radial_grad = np.sum(role_grad[:, :2] * radial[rows, :2], axis=1)
        out["roles"][role] = {
            "count": int(rows.size),
            "norm": float(np.linalg.norm(role_grad)),
            "z_mean": float(np.mean(role_grad[:, 2])),
            "descent_z_mean": float(np.mean(-role_grad[:, 2])),
            "radial_mean": float(np.mean(radial_grad)),
            "descent_radial_mean": float(np.mean(-radial_grad)),
        }
    return out


def _shape_gradient_role_metrics(
    *,
    mesh,
    roles: dict[str, np.ndarray],
    grad: np.ndarray,
) -> dict[str, Any]:
    pos = mesh.positions_view()
    radial = np.zeros_like(pos)
    radii = np.linalg.norm(pos[:, :2], axis=1)
    nonzero = radii > 1.0e-12
    radial[nonzero, :2] = pos[nonzero, :2] / radii[nonzero, None]
    out: dict[str, Any] = {}
    for role, rows in roles.items():
        rows = np.asarray(rows, dtype=int)
        if rows.size == 0:
            out[role] = {
                "count": 0,
                "norm": 0.0,
                "z_mean": 0.0,
                "descent_z_mean": 0.0,
                "radial_mean": 0.0,
                "descent_radial_mean": 0.0,
            }
            continue
        role_grad = grad[rows]
        radial_grad = np.sum(role_grad[:, :2] * radial[rows, :2], axis=1)
        out[role] = {
            "count": int(rows.size),
            "norm": float(np.linalg.norm(role_grad)),
            "z_mean": float(np.mean(role_grad[:, 2])),
            "descent_z_mean": float(np.mean(-role_grad[:, 2])),
            "radial_mean": float(np.mean(radial_grad)),
            "descent_radial_mean": float(np.mean(-radial_grad)),
        }
    return out


def _shape_gradient_module_role_probe(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    minim = ctx.minimizer
    roles = _scaffold_role_rows(mesh)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    raw_total = np.zeros_like(positions)
    modules: dict[str, Any] = {}
    for name, module in zip(minim.energy_module_names, minim.energy_modules):
        if not hasattr(module, "compute_energy_and_gradient_array"):
            continue
        grad = np.zeros_like(positions)
        energy = minim._evaluation_manager._call_module_array(
            module,
            positions=positions,
            index_map=index_map,
            grad_arr=grad,
        )
        scale = minim._experimental_energy_scale_for_module(str(name))
        grad *= float(scale)
        raw_total += grad
        role_metrics = _shape_gradient_role_metrics(
            mesh=mesh,
            roles=roles,
            grad=grad,
        )
        modules[str(name)] = {
            "energy": float(scale) * _as_float(energy),
            "roles": role_metrics,
        }
    raw_roles = _shape_gradient_role_metrics(mesh=mesh, roles=roles, grad=raw_total)
    trace_entries = []
    for name, data in modules.items():
        trace = data.get("roles", {}).get("trace", {})
        trace_entries.append(
            {
                "module": str(name),
                "energy": _as_float(data.get("energy")),
                "trace_descent_z_mean": _as_float(trace.get("descent_z_mean")),
                "trace_norm": _as_float(trace.get("norm")),
            }
        )
    trace_entries.sort(
        key=lambda item: abs(float(item.get("trace_descent_z_mean") or 0.0)),
        reverse=True,
    )
    downward = [
        item
        for item in trace_entries
        if float(item.get("trace_descent_z_mean") or 0.0) < 0.0
    ]
    upward = [
        item
        for item in trace_entries
        if float(item.get("trace_descent_z_mean") or 0.0) > 0.0
    ]
    downward.sort(key=lambda item: float(item.get("trace_descent_z_mean") or 0.0))
    upward.sort(
        key=lambda item: float(item.get("trace_descent_z_mean") or 0.0),
        reverse=True,
    )
    return {
        "modules": modules,
        "raw_total_roles": raw_roles,
        "dominant_trace_z_modules": trace_entries[:6],
        "dominant_trace_downward_module": downward[0] if downward else {},
        "dominant_trace_upward_module": upward[0] if upward else {},
    }


def _set_trace_geometry_phi_target(ctx, *, target_phi: float) -> dict[str, Any]:
    roles = _scaffold_role_rows(ctx.mesh)
    trace_rows = np.asarray(roles.get("trace", np.zeros(0, dtype=int)), dtype=int)
    disk_rows = np.asarray(roles.get("disk", np.zeros(0, dtype=int)), dtype=int)
    if trace_rows.size == 0 or disk_rows.size == 0:
        return {"applied": False, "reason": "missing_trace_or_disk_rows"}
    pos = ctx.mesh.positions_view()
    radii = np.linalg.norm(pos[:, :2], axis=1)
    dr = float(np.mean(radii[trace_rows]) - np.mean(radii[disk_rows]))
    if abs(dr) <= 1.0e-12:
        return {"applied": False, "reason": "zero_trace_disk_spacing"}
    current_phi = _role_phi_to_inner_summary(ctx, roles=roles, role="trace")
    target_trace_z = float(np.mean(pos[disk_rows, 2]) + float(target_phi) * dr)
    dz = target_trace_z - float(np.mean(pos[trace_rows, 2]))
    for row in trace_rows:
        vid = int(ctx.mesh.vertex_ids[int(row)])
        ctx.mesh.vertices[vid].position[2] += dz
    ctx.mesh.increment_version()
    updated_roles = _scaffold_role_rows(ctx.mesh)
    return {
        "applied": True,
        "target_phi": float(target_phi),
        "before_phi": _as_float(current_phi),
        "after_phi": _as_float(
            _role_phi_to_inner_summary(ctx, roles=updated_roles, role="trace")
        ),
        "trace_dz": float(dz),
        "trace_disk_dr": float(dr),
    }


def _high_trace_geometry_seed_probe(
    ctx,
    *,
    mesh_path: Path,
    protocol: tuple[str, ...],
    target_phi: float = 0.15,
    shape_steps: int = 20,
) -> dict[str, Any]:
    snapshot = _capture_state(ctx)
    saved_step_size = getattr(ctx.minimizer, "step_size", None)
    roles = _scaffold_role_rows(ctx.mesh)
    trace_rows = np.asarray(roles.get("trace", np.zeros(0, dtype=int)), dtype=int)
    try:
        seed_geometry = _set_trace_geometry_phi_target(
            ctx, target_phi=float(target_phi)
        )
        seeded_out = _set_radial_tilt_target(
            ctx.mesh,
            ctx.mesh.tilts_out_view(),
            trace_rows,
            target=float(target_phi),
        )
        ctx.mesh.set_tilts_out_from_array(seeded_out)
        seeded_fields = _scaffold_role_field_summary(ctx)
        if hasattr(ctx.minimizer.constraint_manager, "enforce_tilt_constraints"):
            ctx.minimizer.constraint_manager.enforce_tilt_constraints(
                ctx.mesh,
                global_params=ctx.mesh.global_parameters,
            )
        projected_fields = _scaffold_role_field_summary(ctx)
        ctx.minimizer.minimize(n_steps=int(shape_steps))
        relaxed = _collect_live_summary(ctx=ctx, mesh_path=mesh_path, protocol=protocol)
        relaxed_fields = _scaffold_role_field_summary(ctx)
        trace_seeded = seeded_fields.get("trace", {})
        trace_projected = projected_fields.get("trace", {})
        trace_relaxed = relaxed_fields.get("trace", {})
        return {
            "target_phi": float(target_phi),
            "shape_steps": int(shape_steps),
            "seed_geometry": seed_geometry,
            "seeded_trace_phi": _as_float(trace_seeded.get("phi_to_inner")),
            "seeded_trace_t_out": _as_float(trace_seeded.get("tilt_out_radial_mean")),
            "projected_trace_phi": _as_float(trace_projected.get("phi_to_inner")),
            "projected_trace_t_out": _as_float(
                trace_projected.get("tilt_out_radial_mean")
            ),
            "relaxed_trace_phi": _as_float(trace_relaxed.get("phi_to_inner")),
            "relaxed_trace_t_out": _as_float(trace_relaxed.get("tilt_out_radial_mean")),
            "relaxed_direct_t_out": _as_float(
                relaxed.get("interface_summary", {}).get("direct_t_out")
            ),
            "relaxed_direct_phi": _as_float(
                relaxed.get("interface_summary", {}).get("direct_phi")
            ),
            "relaxed_elastic_total": _as_float(
                relaxed.get("elastic_total_from_breakdown")
            ),
            "stop_reason": str(
                relaxed.get("leaflet_relaxation_stats", {}).get("stop_reason") or ""
            ),
        }
    finally:
        _restore_state(ctx, snapshot)
        if saved_step_size is not None:
            ctx.minimizer.step_size = saved_step_size


def _trace_stage_snapshot(ctx, *, stage: str, iteration: int) -> dict[str, Any]:
    fields = _scaffold_role_field_summary(ctx)
    modules = _shape_gradient_module_role_probe(ctx)
    breakdown = {
        key: float(value)
        for key, value in ctx.minimizer.compute_energy_breakdown().items()
    }
    trace = fields.get("trace", {})
    return {
        "iteration": int(iteration),
        "stage": str(stage),
        "energy": _as_float(ctx.minimizer.compute_energy()),
        "thetaB_value": float(
            ctx.mesh.global_parameters.get("tilt_thetaB_value") or 0.0
        ),
        "trace_t_out": _as_float(trace.get("tilt_out_radial_mean")),
        "trace_t_in": _as_float(trace.get("tilt_in_radial_mean")),
        "trace_phi": _as_float(trace.get("phi_to_inner")),
        "energy_breakdown": breakdown,
        "dominant_down": dict(modules.get("dominant_trace_downward_module") or {}),
        "dominant_up": dict(modules.get("dominant_trace_upward_module") or {}),
        "trace_shape_gradient": dict(
            modules.get("raw_total_roles", {}).get("trace", {})
        ),
        "inner_scaffold_shape_stencil_stats": dict(
            getattr(
                ctx.mesh,
                "_last_bending_tilt_in_scaffold_shape_stencil_stats",
                {},
            )
        ),
        "leaflet_relaxation_stats": dict(
            getattr(ctx.minimizer, "_last_leaflet_relaxation_stats", {})
        ),
    }


def _high_trace_stage_replay_probe(
    ctx,
    *,
    target_phi: float = 0.15,
    iterations: int = 4,
) -> dict[str, Any]:
    snapshot = _capture_state(ctx)
    saved_step_size = getattr(ctx.minimizer, "step_size", None)
    rows: list[dict[str, Any]] = []
    try:
        _set_trace_geometry_phi_target(ctx, target_phi=float(target_phi))
        trace_rows = _scaffold_role_rows(ctx.mesh)["trace"]
        seeded_out = _set_radial_tilt_target(
            ctx.mesh,
            ctx.mesh.tilts_out_view(),
            trace_rows,
            target=float(target_phi),
        )
        ctx.mesh.set_tilts_out_from_array(seeded_out)
        if hasattr(ctx.minimizer.constraint_manager, "enforce_tilt_constraints"):
            ctx.minimizer.constraint_manager.enforce_tilt_constraints(
                ctx.mesh,
                global_params=ctx.mesh.global_parameters,
            )
        rows.append(_trace_stage_snapshot(ctx, stage="seed_projected", iteration=0))
        ctx.minimizer.enforce_constraints_after_mesh_ops(ctx.mesh)
        ctx.mesh.project_tilts_to_tangent()
        ctx.mesh.increment_version()
        rows.append(
            _trace_stage_snapshot(
                ctx, stage="after_initial_mesh_constraints", iteration=0
            )
        )
        tilt_mode = str(
            ctx.mesh.global_parameters.get("tilt_solve_mode", "fixed") or "fixed"
        )
        for i in range(int(iterations)):
            ctx.minimizer._relax_leaflet_tilts(
                positions=ctx.mesh.positions_view(),
                mode=tilt_mode,
            )
            ctx.mesh.project_tilts_to_tangent()
            ctx.mesh.increment_version()
            ctx.minimizer._update_scalar_params()
            _optimize_thetaB_scalar(ctx.minimizer, tilt_mode=tilt_mode, iteration=i)
            rows.append(
                _trace_stage_snapshot(ctx, stage="after_tilt_solve", iteration=i)
            )

            energy_before, grad_arr = ctx.minimizer.compute_energy_and_gradient_array()
            ctx.minimizer.project_constraints_array(grad_arr)
            project_curved_free_disk_shape_dofs(
                ctx.mesh, ctx.mesh.global_parameters, grad_arr
            )
            rows.append(
                {
                    **_trace_stage_snapshot(
                        ctx, stage="after_shape_gradient", iteration=i
                    ),
                    "shape_energy_before_step": _as_float(energy_before),
                    "shape_gradient_norm": float(np.linalg.norm(grad_arr)),
                }
            )

            step_mode = str(
                ctx.mesh.global_parameters.get("step_size_mode", "adaptive")
                or "adaptive"
            ).lower()
            fixed_step = float(
                ctx.mesh.global_parameters.get("step_size", ctx.minimizer.step_size)
                or ctx.minimizer.step_size
            )
            step_size_in = (
                fixed_step if step_mode == "fixed" else ctx.minimizer.step_size
            )
            energy_fn = ctx.minimizer._line_search_energy_fn()
            trial_energy_fn = ctx.minimizer._line_search_trial_energy_fn()
            reduced_flag = (
                bool(
                    ctx.mesh.global_parameters.get("line_search_reduced_energy", False)
                )
                and int(
                    ctx.mesh.global_parameters.get(
                        "line_search_reduced_tilt_inner_steps", 0
                    )
                    or 0
                )
                > 0
            )
            setattr(ctx.mesh, "_line_search_reduced_energy", reduced_flag)
            if reduced_flag:
                accept_rule = str(
                    ctx.mesh.global_parameters.get(
                        "line_search_reduced_accept_rule", "armijo"
                    )
                    or "armijo"
                )
                setattr(ctx.mesh, "_line_search_reduced_accept_rule", accept_rule)
            step_kwargs = {}
            if ctx.minimizer._stepper_supports_trial_energy_fn():
                step_kwargs["trial_energy_fn"] = trial_energy_fn
            try:
                step_success, new_step, accepted_energy = ctx.minimizer.stepper.step(
                    ctx.mesh,
                    grad_arr,
                    step_size_in,
                    energy_fn,
                    constraint_enforcer=ctx.minimizer._enforce_constraints
                    if ctx.minimizer._has_enforceable_constraints
                    else None,
                    **step_kwargs,
                )
            finally:
                if hasattr(ctx.mesh, "_line_search_reduced_energy"):
                    delattr(ctx.mesh, "_line_search_reduced_energy")
                if hasattr(ctx.mesh, "_line_search_reduced_accept_rule"):
                    delattr(ctx.mesh, "_line_search_reduced_accept_rule")
            ctx.minimizer.step_size = new_step
            ctx.mesh.project_tilts_to_tangent()
            ctx.mesh.increment_version()
            rows.append(
                {
                    **_trace_stage_snapshot(ctx, stage="after_shape_step", iteration=i),
                    "shape_step_success": bool(step_success),
                    "shape_step_size_in": float(step_size_in),
                    "shape_step_size_out": float(new_step),
                    "shape_accepted_energy": _as_float(accepted_energy),
                }
            )
        if ctx.minimizer._has_enforceable_constraints:
            ctx.minimizer.constraint_manager.enforce_all(
                ctx.mesh,
                global_params=ctx.mesh.global_parameters,
                context="finalize",
            )
            ctx.mesh.project_tilts_to_tangent()
            ctx.mesh.increment_version()
            rows.append(
                _trace_stage_snapshot(
                    ctx, stage="after_finalize_constraints", iteration=int(iterations)
                )
            )
        return {
            "target_phi": float(target_phi),
            "iterations": int(iterations),
            "rows": rows,
        }
    finally:
        _restore_state(ctx, snapshot)
        if saved_step_size is not None:
            ctx.minimizer.step_size = saved_step_size


def _branch_access_probe(
    ctx,
    *,
    target_phi: float = 0.15,
) -> dict[str, Any]:
    snapshot = _capture_state(ctx)

    def _summary(label: str) -> dict[str, Any]:
        theta = float(ctx.mesh.global_parameters.get("tilt_thetaB_value") or 0.0)
        fields = _scaffold_role_field_summary(ctx)
        trace = fields.get("trace", {})
        tilt_probe = _outer_tilt_gradient_role_probe(ctx, theta=theta)
        shape_probe = _shape_gradient_module_role_probe(ctx)
        breakdown = {
            key: float(value)
            for key, value in ctx.minimizer.compute_energy_breakdown().items()
        }
        return {
            "label": str(label),
            "thetaB_value": float(theta),
            "trace_t_out": _as_float(trace.get("tilt_out_radial_mean")),
            "trace_t_in": _as_float(trace.get("tilt_in_radial_mean")),
            "trace_phi": _as_float(trace.get("phi_to_inner")),
            "energy": _as_float(ctx.minimizer.compute_energy()),
            "energy_breakdown": breakdown,
            "tilt_descent_trace": dict(
                tilt_probe.get("descent_direction", {}).get("trace", {})
            ),
            "first_update_trace": dict(
                tilt_probe.get("first_update", {}).get("trace", {})
            ),
            "shape_trace": dict(
                shape_probe.get("raw_total_roles", {}).get("trace", {})
            ),
            "shape_dominant_down": dict(
                shape_probe.get("dominant_trace_downward_module") or {}
            ),
            "shape_dominant_up": dict(
                shape_probe.get("dominant_trace_upward_module") or {}
            ),
        }

    roles = _scaffold_role_rows(ctx.mesh)
    trace_rows = np.asarray(roles.get("trace", np.zeros(0, dtype=int)), dtype=int)
    rows: list[dict[str, Any]] = []
    try:
        rows.append(_summary("fresh_optimized"))

        seeded_out = _set_radial_tilt_target(
            ctx.mesh,
            ctx.mesh.tilts_out_view(),
            trace_rows,
            target=float(target_phi),
        )
        ctx.mesh.set_tilts_out_from_array(seeded_out)
        if hasattr(ctx.minimizer.constraint_manager, "enforce_tilt_constraints"):
            ctx.minimizer.constraint_manager.enforce_tilt_constraints(
                ctx.mesh,
                global_params=ctx.mesh.global_parameters,
            )
        rows.append(_summary("tilt_seed_projected"))

        _restore_state(ctx, snapshot)
        _set_trace_geometry_phi_target(ctx, target_phi=float(target_phi))
        seeded_out = _set_radial_tilt_target(
            ctx.mesh,
            ctx.mesh.tilts_out_view(),
            trace_rows,
            target=float(target_phi),
        )
        ctx.mesh.set_tilts_out_from_array(seeded_out)
        if hasattr(ctx.minimizer.constraint_manager, "enforce_tilt_constraints"):
            ctx.minimizer.constraint_manager.enforce_tilt_constraints(
                ctx.mesh,
                global_params=ctx.mesh.global_parameters,
            )
        rows.append(_summary("geometry_tilt_seed_projected"))

        return {
            "target_phi": float(target_phi),
            "rows": rows,
        }
    finally:
        _restore_state(ctx, snapshot)


def _set_positions_from_array(ctx, positions: np.ndarray) -> None:
    pos = np.asarray(positions, dtype=float)
    for row, vid in enumerate(ctx.mesh.vertex_ids):
        ctx.mesh.vertices[int(vid)].position[:] = pos[row]
    ctx.mesh.increment_version()


def _breakdown_delta(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    keys = sorted(set(before) | set(after))
    out = {key: float(after.get(key, 0.0) - before.get(key, 0.0)) for key in keys}
    out["__total__"] = float(sum(out.values()))
    return out


def _dominant_positive_delta(delta: dict[str, float]) -> dict[str, Any]:
    terms = {k: v for k, v in delta.items() if not str(k).startswith("__")}
    if not terms:
        return {"module": "", "delta": 0.0}
    module, value = max(terms.items(), key=lambda item: float(item[1]))
    return {"module": str(module), "delta": float(value)}


def _bt_leaflet_role_total_deltas(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    role_names = sorted(
        set((before.get("roles") or {}).keys()) | set((after.get("roles") or {}).keys())
    )
    roles: dict[str, Any] = {}
    for role in role_names:
        b_role = (before.get("roles") or {}).get(role, {})
        a_role = (after.get("roles") or {}).get(role, {})
        roles[role] = {
            "total_energy_delta": float(
                a_role.get("total_energy", 0.0) - b_role.get("total_energy", 0.0)
            ),
            "base_energy_delta": float(
                a_role.get("base_energy", 0.0) - b_role.get("base_energy", 0.0)
            ),
            "divergence_energy_delta": float(
                a_role.get("divergence_energy", 0.0)
                - b_role.get("divergence_energy", 0.0)
            ),
            "cross_energy_delta": float(
                a_role.get("cross_energy", 0.0) - b_role.get("cross_energy", 0.0)
            ),
        }
    dominant_role = (
        max(
            roles.items(),
            key=lambda item: float(item[1].get("total_energy_delta", 0.0)),
        )[0]
        if roles
        else ""
    )
    return {
        "roles": roles,
        "dominant_positive_role": str(dominant_role),
        "dominant_positive_delta": float(
            roles.get(dominant_role, {}).get("total_energy_delta", 0.0)
        )
        if dominant_role
        else 0.0,
    }


def _apply_trace_z_probe_constraints(ctx, context: str) -> None:
    context = str(context)
    if context == "none":
        return
    if context == "minimize":
        ctx.minimizer._enforce_constraints(ctx.mesh)
        ctx.mesh.project_tilts_to_tangent()
        ctx.mesh.increment_version()
        return
    if context == "mesh_operation":
        ctx.minimizer.enforce_constraints_after_mesh_ops(ctx.mesh)
        ctx.mesh.project_tilts_to_tangent()
        ctx.mesh.increment_version()
        return
    if context == "finalize":
        if ctx.minimizer._has_enforceable_constraints:
            ctx.minimizer.constraint_manager.enforce_all(
                ctx.mesh,
                global_params=ctx.mesh.global_parameters,
                context="finalize",
            )
        if hasattr(ctx.minimizer.constraint_manager, "enforce_tilt_constraints"):
            ctx.minimizer.constraint_manager.enforce_tilt_constraints(
                ctx.mesh,
                global_params=ctx.mesh.global_parameters,
            )
        ctx.mesh.project_tilts_to_tangent()
        ctx.mesh.increment_version()
        return
    raise ValueError(f"unknown trace-z fallback constraint context: {context}")


def _trace_z_fallback_trial_decomposition_probe(ctx) -> dict[str, Any]:
    roles = _scaffold_role_rows(ctx.mesh)
    trace_rows = np.asarray(roles.get("trace", np.zeros(0, dtype=int)), dtype=int)
    if trace_rows.size == 0:
        return {"available": False, "reason": "no_trace_rows", "samples": []}

    snapshot = _capture_state(ctx)
    try:
        energy, grad_arr = ctx.minimizer.compute_energy_and_gradient_array()
        ctx.minimizer.project_constraints_array(grad_arr)
        project_curved_free_disk_shape_dofs(
            ctx.mesh, ctx.mesh.global_parameters, grad_arr
        )
        direction = np.zeros_like(grad_arr)
        direction[trace_rows, 2] = -np.asarray(grad_arr[trace_rows, 2], dtype=float)
        descent_z_mean = float(np.mean(direction[trace_rows, 2]))
        if not np.isfinite(descent_z_mean) or descent_z_mean <= 0.0:
            return {
                "available": False,
                "reason": "non_positive_trace_z_descent",
                "trace_descent_z_mean": descent_z_mean,
                "samples": [],
            }

        base_positions = ctx.mesh.positions_view().copy(order="F")
        base_breakdown = {
            key: float(value)
            for key, value in ctx.minimizer.compute_energy_breakdown().items()
        }
        base_bt_decomposition = {
            "bending_tilt_in": _bending_tilt_leaflet_interface_decomposition(
                ctx, leaflet="in"
            ),
            "bending_tilt_out": _bending_tilt_leaflet_interface_decomposition(
                ctx, leaflet="out"
            ),
        }
        base_fields = _scaffold_role_field_summary(ctx)
        samples: list[dict[str, Any]] = []
        for alpha in TRACE_Z_FALLBACK_ALPHA_GRID:
            trial_positions = base_positions + float(alpha) * direction
            for context in TRACE_Z_FALLBACK_CONSTRAINT_CONTEXTS:
                _restore_state(ctx, snapshot)
                _set_positions_from_array(ctx, trial_positions)
                pre_constraint_positions = ctx.mesh.positions_view().copy(order="F")
                _apply_trace_z_probe_constraints(ctx, context)
                post_positions = ctx.mesh.positions_view().copy(order="F")
                breakdown = {
                    key: float(value)
                    for key, value in ctx.minimizer.compute_energy_breakdown().items()
                }
                bt_decomposition = {
                    "bending_tilt_in": _bt_leaflet_role_total_deltas(
                        base_bt_decomposition["bending_tilt_in"],
                        _bending_tilt_leaflet_interface_decomposition(
                            ctx, leaflet="in"
                        ),
                    ),
                    "bending_tilt_out": _bt_leaflet_role_total_deltas(
                        base_bt_decomposition["bending_tilt_out"],
                        _bending_tilt_leaflet_interface_decomposition(
                            ctx, leaflet="out"
                        ),
                    ),
                }
                delta = _breakdown_delta(base_breakdown, breakdown)
                fields = _scaffold_role_field_summary(ctx)
                trace = fields.get("trace", {})
                support = fields.get("support", {})
                trace_dz_applied = (
                    pre_constraint_positions[trace_rows, 2]
                    - base_positions[trace_rows, 2]
                )
                trace_dz_final = (
                    post_positions[trace_rows, 2] - base_positions[trace_rows, 2]
                )
                samples.append(
                    {
                        "alpha": float(alpha),
                        "constraint_context": str(context),
                        "base_energy": float(sum(base_breakdown.values())),
                        "trial_energy": float(sum(breakdown.values())),
                        "energy_delta": float(delta["__total__"]),
                        "module_deltas": delta,
                        "dominant_positive_delta": _dominant_positive_delta(delta),
                        "bending_tilt_role_deltas": bt_decomposition,
                        "trace_dz_applied_mean": float(np.mean(trace_dz_applied)),
                        "trace_dz_final_mean": float(np.mean(trace_dz_final)),
                        "trace_dz_preserved_ratio": float(
                            np.mean(trace_dz_final) / np.mean(trace_dz_applied)
                        )
                        if abs(float(np.mean(trace_dz_applied))) > 1.0e-15
                        else 0.0,
                        "trace_t_out_before": _as_float(
                            base_fields.get("trace", {}).get("tilt_out_radial_mean")
                        ),
                        "trace_t_out_after": _as_float(
                            trace.get("tilt_out_radial_mean")
                        ),
                        "trace_phi_before": _as_float(
                            base_fields.get("trace", {}).get("phi_to_inner")
                        ),
                        "trace_phi_after": _as_float(trace.get("phi_to_inner")),
                        "support_phi_after": _as_float(support.get("phi_to_inner")),
                    }
                )

        lowering = [
            sample
            for sample in samples
            if float(sample.get("energy_delta") or 0.0) < 0.0
        ]
        best = min(
            samples,
            key=lambda sample: float(sample.get("energy_delta") or 0.0),
        )
        return {
            "available": True,
            "energy_before": float(energy),
            "trace_count": int(trace_rows.size),
            "trace_descent_z_mean": float(descent_z_mean),
            "samples": samples,
            "best_sample": dict(best),
            "lowering_count": int(len(lowering)),
        }
    finally:
        _restore_state(ctx, snapshot)


def _scaffold_geometry_spacing_probe(
    *,
    tmpdir: Path,
    protocol: tuple[str, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in SCAFFOLD_GEOMETRY_SPACING_VARIANTS:
        doc, label = _scaffold_geometry_spacing_doc(spec)
        mesh_path = _write_temp_fixture(doc, tmpdir, label)
        ctx, summary = _run_protocol_summary(mesh_path=mesh_path, protocol=protocol)
        conditioning = _bending_tilt_out_divergence_conditioning(ctx)
        ablation = _bending_tilt_out_divergence_ablation(ctx)
        theta = _as_float(summary.get("thetaB_value"))
        gradient_update = _outer_tilt_gradient_role_probe(ctx, theta=theta)
        high_seed = _high_trace_seed_replay(
            ctx,
            mesh_path=mesh_path,
            protocol=protocol,
            target_radial=0.15,
            steps=60,
        )
        high_seed_projection = _high_trace_constraint_projection_probe(
            ctx, target_radial=0.15
        )
        shape_gradient = _shape_gradient_role_probe(ctx)
        shape_gradient_modules = _shape_gradient_module_role_probe(ctx)
        geometry_seed = (
            _high_trace_geometry_seed_probe(
                ctx,
                mesh_path=mesh_path,
                protocol=protocol,
                target_phi=0.15,
                shape_steps=20,
            )
            if str(label) in SCAFFOLD_GEOMETRY_SEED_LABELS
            else {}
        )
        stage_replay = (
            _high_trace_stage_replay_probe(ctx, target_phi=0.15, iterations=4)
            if str(label).endswith("_inner_trace_boundary")
            or str(spec.get("scaffold_mesh_operation_mode") or "")
            == "preserve_trace_v1"
            else {}
        )
        branch_access = (
            _branch_access_probe(ctx, target_phi=0.15)
            if str(spec.get("scaffold_mesh_operation_mode") or "")
            == "preserve_trace_v1"
            else {}
        )
        trace_z_trials = (
            _trace_z_fallback_trial_decomposition_probe(ctx)
            if str(spec.get("scaffold_mesh_operation_mode") or "")
            == "preserve_trace_v1"
            and "trace_layer"
            in set(spec.get("pin_to_circle_preserve_normal_groups") or [])
            else {}
        )
        rows.append(
            {
                "label": label,
                "geometry": str(spec["geometry"]),
                "epsilon": float(spec.get("epsilon", 0.005)),
                "outer_shells": int(spec.get("outer_shells", 0)),
                "outer_shells_d": float(spec.get("outer_shells_d", 0.0)),
                "interface_divergence_mode": str(
                    spec.get("interface_divergence_mode") or "p1_triangle"
                ),
                "inner_scaffold_shape_stencil_mode": str(
                    spec.get("inner_scaffold_shape_stencil_mode") or "off"
                ),
                "scaffold_mesh_operation_mode": str(
                    spec.get("scaffold_mesh_operation_mode") or "project"
                ),
                "pin_to_circle_preserve_normal_groups": list(
                    spec.get("pin_to_circle_preserve_normal_groups") or []
                ),
                "shape_scaffold_rejected_step_fallback": str(
                    spec.get("shape_scaffold_rejected_step_fallback") or "off"
                ),
                "thetaB_value": theta,
                "direct_t_out": _as_float(
                    summary.get("interface_summary", {}).get("direct_t_out")
                ),
                "direct_phi": _as_float(
                    summary.get("interface_summary", {}).get("direct_phi")
                ),
                "energy_breakdown": dict(summary.get("energy_breakdown", {})),
                "elastic_total_from_breakdown": _as_float(
                    summary.get("elastic_total_from_breakdown")
                ),
                "stop_reason": str(
                    summary.get("leaflet_relaxation_stats", {}).get("stop_reason") or ""
                ),
                "conditioning": conditioning,
                "divergence_ablation": ablation,
                "gradient_update_probe": gradient_update,
                "high_trace_seed_replay": high_seed,
                "high_trace_constraint_projection": high_seed_projection,
                "shape_gradient_probe": shape_gradient,
                "shape_gradient_module_probe": shape_gradient_modules,
                "shape_scaffold_rejected_step_fallback_stats": dict(
                    summary.get("shape_scaffold_rejected_step_fallback_stats", {})
                ),
                "high_trace_geometry_seed_probe": geometry_seed,
                "high_trace_stage_replay_probe": stage_replay,
                "branch_access_probe": branch_access,
                "trace_z_fallback_trial_decomposition_probe": trace_z_trials,
                "role_constraint_tags": _role_constraint_tag_summary(ctx),
                "trace_div_abs_mean": _conditioning_role_metric(
                    conditioning, "trace_touching", ("divergence", "abs_mean")
                ),
                "trace_basis_max": _conditioning_role_metric(
                    conditioning, "trace_touching", ("basis_norm", "max_abs")
                ),
                "trace_min_edge": _conditioning_role_metric(
                    conditioning, "trace_touching", ("min_edge", "min")
                ),
                "trace_area_mean": _conditioning_role_metric(
                    conditioning, "trace_touching", ("area", "mean")
                ),
                "support_div_abs_mean": _conditioning_role_metric(
                    conditioning, "support_touching", ("divergence", "abs_mean")
                ),
                "support_basis_max": _conditioning_role_metric(
                    conditioning, "support_touching", ("basis_norm", "max_abs")
                ),
                "interface_divergence_stats": dict(
                    getattr(
                        ctx.mesh,
                        "_last_bending_tilt_out_interface_divergence_stats",
                        {},
                    )
                ),
                "inner_scaffold_shape_stencil_stats": dict(
                    getattr(
                        ctx.mesh,
                        "_last_bending_tilt_in_scaffold_shape_stencil_stats",
                        {},
                    )
                ),
                "rim_slope_match_scaffold_mesh_operation_stats": dict(
                    getattr(
                        ctx.mesh,
                        "_last_rim_slope_match_scaffold_mesh_operation_stats",
                        {},
                    )
                ),
            }
        )
    best_by_trace_div = (
        min(
            rows,
            key=lambda row: float(row.get("trace_div_abs_mean") or float("inf")),
        )
        if rows
        else {}
    )
    best_by_direct_t = (
        max(
            rows,
            key=lambda row: float(row.get("direct_t_out") or float("-inf")),
        )
        if rows
        else {}
    )
    best_ablation = (
        min(
            rows,
            key=lambda row: float(
                row.get("divergence_ablation", {})
                .get("summary", {})
                .get("best_energy_delta")
                or 0.0
            ),
        )
        if rows
        else {}
    )
    best_high_seed = (
        max(
            rows,
            key=lambda row: float(
                row.get("high_trace_seed_replay", {}).get("relaxed_direct_t_out")
                or float("-inf")
            ),
        )
        if rows
        else {}
    )
    seeded_geometry_rows = [
        row for row in rows if row.get("high_trace_geometry_seed_probe")
    ]
    best_geometry_seed = (
        max(
            seeded_geometry_rows,
            key=lambda row: float(
                row.get("high_trace_geometry_seed_probe", {}).get(
                    "relaxed_direct_t_out"
                )
                or float("-inf")
            ),
        )
        if seeded_geometry_rows
        else {}
    )
    return {
        "rows": rows,
        "summary": {
            "best_trace_divergence_label": str(best_by_trace_div.get("label") or ""),
            "best_trace_divergence": _as_float(
                best_by_trace_div.get("trace_div_abs_mean")
            ),
            "best_direct_t_out_label": str(best_by_direct_t.get("label") or ""),
            "best_direct_t_out": _as_float(best_by_direct_t.get("direct_t_out")),
            "best_ablation_label": str(best_ablation.get("label") or ""),
            "best_ablation_mode": str(
                best_ablation.get("divergence_ablation", {})
                .get("summary", {})
                .get("best_energy_delta_label")
                or ""
            ),
            "best_ablation_energy_delta": _as_float(
                best_ablation.get("divergence_ablation", {})
                .get("summary", {})
                .get("best_energy_delta")
            ),
            "best_high_seed_label": str(best_high_seed.get("label") or ""),
            "best_high_seed_direct_t_out": _as_float(
                best_high_seed.get("high_trace_seed_replay", {}).get(
                    "relaxed_direct_t_out"
                )
            ),
            "best_geometry_seed_label": str(best_geometry_seed.get("label") or ""),
            "best_geometry_seed_direct_t_out": _as_float(
                best_geometry_seed.get("high_trace_geometry_seed_probe", {}).get(
                    "relaxed_direct_t_out"
                )
            ),
            "geometry_seed_labels": [
                str(row.get("label") or "") for row in seeded_geometry_rows
            ],
        },
    }


def _scaffold_gd_line_search_probe(ctx) -> dict[str, Any]:
    mesh = ctx.mesh
    minim = ctx.minimizer
    gp = mesh.global_parameters
    positions = mesh.positions_view()
    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")
    normals = mesh.vertex_normals(positions=positions)
    fixed_in = np.asarray(minim._tilt_fixed_mask_in(), dtype=bool)
    fixed_out = np.asarray(minim._tilt_fixed_mask_out(), dtype=bool)
    fixed_vals_in = tilts_in[fixed_in].copy() if np.any(fixed_in) else None
    fixed_vals_out = tilts_out[fixed_out].copy() if np.any(fixed_out) else None
    areas_in, areas_out = _leaflet_tilt_vertex_areas(ctx, positions)
    grad_in = np.zeros_like(tilts_in)
    grad_out = np.zeros_like(tilts_out)
    grad_dummy = np.zeros_like(positions)
    e0_tilt = minim._compute_energy_and_leaflet_tilt_gradients_array(
        positions=positions,
        tilts_in=tilts_in,
        tilts_out=tilts_out,
        tilt_in_grad_arr=grad_in,
        tilt_out_grad_arr=grad_out,
        tilt_vertex_areas_in=areas_in,
        tilt_vertex_areas_out=areas_out,
        grad_dummy=grad_dummy,
        tilt_only=True,
    )
    raw_grad_out = grad_out.copy(order="F")
    raw_grad_in = grad_in.copy(order="F")
    minim.constraint_manager.apply_tilt_gradient_modifications_array(
        grad_in,
        grad_out,
        mesh,
        gp,
        positions=positions,
        tilts_in=tilts_in,
        tilts_out=tilts_out,
    )
    if np.any(fixed_in):
        grad_in[fixed_in] = 0.0
    if np.any(fixed_out):
        grad_out[fixed_out] = 0.0
    base_breakdown = _energy_breakdown_with_leaflet_tilts(
        ctx, tilts_in=tilts_in, tilts_out=tilts_out
    )
    base_total = _tilt_dependent_total_from_breakdown(base_breakdown)
    step0 = float(gp.get("tilt_step_size", 0.0) or 0.0)
    samples: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    roles = _scaffold_role_rows(mesh)
    for i in range(12):
        step = step0 * (0.5**i)
        trial_in, trial_out = build_leaflet_trial_tilts(
            base_in=tilts_in,
            base_out=tilts_out,
            delta_in=-step * grad_in,
            delta_out=-step * grad_out,
            normals=normals,
            fixed_mask_in=fixed_in,
            fixed_mask_out=fixed_out,
            fixed_vals_in=fixed_vals_in,
            fixed_vals_out=fixed_vals_out,
        )
        breakdown = _energy_breakdown_with_leaflet_tilts(
            ctx, tilts_in=trial_in, tilts_out=trial_out
        )
        total = _tilt_dependent_total_from_breakdown(breakdown)
        delta_total = float(total - base_total)
        delta_out = trial_out - tilts_out
        sample = {
            "backtrack": int(i),
            "step": float(step),
            "tilt_dependent_delta": delta_total,
            "tilt_out_delta": float(
                breakdown.get("tilt_out", 0.0) - base_breakdown.get("tilt_out", 0.0)
            ),
            "bending_tilt_out_delta": float(
                breakdown.get("bending_tilt_out", 0.0)
                - base_breakdown.get("bending_tilt_out", 0.0)
            ),
            "trace_update_radial_mean": _row_radial_mean(
                mesh, delta_out, roles.get("trace", np.zeros(0, dtype=int))
            ),
            "support_update_radial_mean": _row_radial_mean(
                mesh, delta_out, roles.get("support", np.zeros(0, dtype=int))
            ),
            "release_update_radial_mean": _row_radial_mean(
                mesh, delta_out, roles.get("release", np.zeros(0, dtype=int))
            ),
        }
        samples.append(sample)
        if best is None or delta_total < float(best["tilt_dependent_delta"]):
            best = sample
    accepted = next(
        (sample for sample in samples if float(sample["tilt_dependent_delta"]) <= 0.0),
        None,
    )
    role_rows = np.unique(
        np.concatenate(
            [
                roles.get("trace", np.zeros(0, dtype=int)),
                roles.get("support", np.zeros(0, dtype=int)),
                roles.get("release", np.zeros(0, dtype=int)),
            ]
        )
    )
    return {
        "tilt_dependent_energy": float(e0_tilt),
        "breakdown_tilt_dependent_energy": float(base_total),
        "raw_outer_grad_norm": float(np.linalg.norm(raw_grad_out[role_rows]))
        if role_rows.size
        else 0.0,
        "projected_outer_grad_norm": float(np.linalg.norm(grad_out[role_rows]))
        if role_rows.size
        else 0.0,
        "raw_vs_projected_outer_grad_cosine": _cosine_similarity(
            raw_grad_out[role_rows] if role_rows.size else np.zeros((0, 3)),
            grad_out[role_rows] if role_rows.size else np.zeros((0, 3)),
        ),
        "raw_inner_grad_norm": float(np.linalg.norm(raw_grad_in[role_rows]))
        if role_rows.size
        else 0.0,
        "projected_inner_grad_norm": float(np.linalg.norm(grad_in[role_rows]))
        if role_rows.size
        else 0.0,
        "first_sample": samples[0] if samples else {},
        "best_sample": best or {},
        "accepted_sample": accepted or {},
        "has_gd_descent_step": accepted is not None,
    }


def _full_physics_scaffold_collapse_probe(
    *,
    tmpdir: Path,
    protocol: tuple[str, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for geometry in SCAFFOLD_COLLAPSE_GEOMETRIES:
        base_doc, base_label = _trace_convergence_current_geometry_doc(
            geometry=geometry,
            epsilon=0.005,
        )
        for variant in SCAFFOLD_COLLAPSE_VARIANTS:
            label = f"{base_label}_{variant}"
            doc = _apply_current_geometry_scaffold_probe_options(
                base_doc,
                lane=f"physical_edge_full_coupling_{geometry}_collapse_{variant}",
                variant=variant,
            )
            mesh_path = _write_temp_fixture(doc, tmpdir, label)
            ctx, summary = _run_protocol_summary(mesh_path=mesh_path, protocol=protocol)
            shell_rows = _outer_shell_rows(ctx.mesh)
            theta = float(summary.get("thetaB_value") or 0.0)
            coupling = _outer_bending_tilt_gradient_components(
                ctx=ctx,
                div_term_sign=1.0,
                pullback_sign=1.0,
            )
            interface = dict(summary.get("interface_summary", {}))
            ratios = dict(summary.get("tex_ratio_summary", {}))
            breakdown = dict(summary.get("energy_breakdown", {}))
            relax = dict(summary.get("leaflet_relaxation_stats", {}))
            driver = dict(summary.get("scaffold_boundary_driver", {}))
            part = _outer_participation_snapshot(ctx.mesh, ctx=ctx)
            update = _one_step_shell_update_summary(ctx=ctx, theta=theta)
            role_fields = _scaffold_role_field_summary(ctx)
            gd_probe = _scaffold_gd_line_search_probe(ctx)
            gp = ctx.mesh.global_parameters
            constraints = [str(x) for x in (doc.get("constraint_modules") or [])]
            row = {
                "label": label,
                "geometry": geometry,
                "variant": variant,
                "epsilon": 0.005,
                "thetaB_value": _as_float(summary.get("thetaB_value")),
                "model_intent": str(
                    summary.get("report_metadata", {}).get("model_intent") or ""
                ),
                "reference_mode": str(
                    summary.get("report_metadata", {}).get("reference_mode") or ""
                ),
                "projector_mode": str(
                    gp.get("rim_slope_match_scaffold_projector_mode") or ""
                ),
                "has_thetaB_boundary_constraint": "tilt_thetaB_boundary_in"
                in constraints,
                "constraint_modules": constraints,
                "direct_t_out": _as_float(interface.get("direct_t_out")),
                "direct_phi": _as_float(interface.get("direct_phi")),
                "tex_total_ratio": _as_float(summary.get("tex_total_ratio")),
                "elastic_ratio": _as_float(ratios.get("elastic_ratio")),
                "contact_ratio": _as_float(ratios.get("contact_ratio")),
                "tilt_out": _as_float(breakdown.get("tilt_out")),
                "bending_tilt_out": _as_float(breakdown.get("bending_tilt_out")),
                "shell_grad_norm": _as_float(
                    _shell_vector_summary(
                        ctx.mesh, coupling["combined_gradient"], shell_rows
                    ).get("norm")
                ),
                "shell_update_norm": _as_float(update.get("norm")),
                "shell_base_mean": _as_float(
                    coupling.get("summary", {}).get("base_term_outer_shell_mean")
                ),
                "outer_shell_free_count": int(part.get("outer_shell_free_count", 0)),
                "active_outer_area_rows": int(relax.get("active_outer_area_rows", 0)),
                "stop_reason": str(relax.get("stop_reason") or ""),
                "final_gradient_norm": _as_float(relax.get("final_gradient_norm")),
                "thetaB_scan_count": int(
                    summary.get("thetaB_scan_summary", {}).get("scan_count", 0) or 0
                ),
                "driver_stationarity_residual": _as_float(
                    driver.get("stationarity_residual")
                ),
                "cg_rejection_fallback": str(
                    relax.get("cg_rejection_fallback") or "off"
                ),
                "cg_fallback_attempted_count": int(
                    relax.get("cg_fallback_attempted_count", 0) or 0
                ),
                "cg_fallback_accepted_count": int(
                    relax.get("cg_fallback_accepted_count", 0) or 0
                ),
                "cg_fallback_step_size_last": _as_float(
                    relax.get("cg_fallback_step_size_last")
                ),
                "role_field_summary": role_fields,
                "gd_line_search_probe": gd_probe,
            }
            row["classification"] = _scaffold_collapse_row_classification(row)
            rows.append(row)
    return {"rows": rows, "summary": _scaffold_collapse_summary(rows)}


def _support_ownership_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lowering = [
        row
        for row in rows
        if bool(
            row.get("support_continuation_probe", {}).get(
                "energy_lowering_positive_branch", False
            )
        )
    ]
    best_improved = False
    by_geometry: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_geometry.setdefault(str(row.get("geometry")), {})[
            str(row.get("variant"))
        ] = row
    for variants in by_geometry.values():
        baseline = variants.get("runtime_options_gd_fallback")
        if baseline is None:
            continue
        base_t = float(baseline.get("direct_t_out") or 0.0)
        base_phi = float(baseline.get("direct_phi") or 0.0)
        for name, row in variants.items():
            if name in {"runtime_options", "runtime_options_gd_fallback"}:
                continue
            if float(row.get("direct_t_out") or 0.0) > base_t + 1.0e-4:
                best_improved = True
            if float(row.get("direct_phi") or 0.0) > base_phi + 1.0e-4:
                best_improved = True
    if lowering:
        classification = "support_continuation_energy_lowering"
    elif best_improved:
        classification = "support_ownership_ablation_improves_trace"
    else:
        classification = "support_damping_persists"
    return {
        "classification": classification,
        "energy_lowering_positive_branch_count": int(len(lowering)),
        "ablation_improved_trace": bool(best_improved),
    }


def _full_physics_scaffold_support_ownership_probe(
    *,
    tmpdir: Path,
    protocol: tuple[str, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for geometry in SCAFFOLD_COLLAPSE_GEOMETRIES:
        base_doc, base_label = _trace_convergence_current_geometry_doc(
            geometry=geometry,
            epsilon=0.005,
        )
        for variant in SCAFFOLD_SUPPORT_OWNERSHIP_VARIANTS:
            label = f"{base_label}_ownership_{variant}"
            doc = _apply_support_ownership_probe_options(
                base_doc,
                lane=f"physical_edge_full_coupling_{geometry}_ownership_{variant}",
                variant=variant,
            )
            mesh_path = _write_temp_fixture(doc, tmpdir, label)
            ctx, summary = _run_protocol_summary(mesh_path=mesh_path, protocol=protocol)
            theta = float(summary.get("thetaB_value") or 0.0)
            coupling = _outer_bending_tilt_gradient_components(
                ctx=ctx,
                div_term_sign=1.0,
                pullback_sign=1.0,
            )
            combined_grad = np.asarray(coupling["combined_gradient"], dtype=float)
            update_delta = _one_step_outer_update_array(ctx, theta=theta)
            interface = dict(summary.get("interface_summary", {}))
            ratios = dict(summary.get("tex_ratio_summary", {}))
            breakdown = dict(summary.get("energy_breakdown", {}))
            relax = dict(summary.get("leaflet_relaxation_stats", {}))
            row = {
                "label": label,
                "geometry": geometry,
                "variant": variant,
                "epsilon": 0.005,
                "thetaB_value": _as_float(summary.get("thetaB_value")),
                "direct_t_out": _as_float(interface.get("direct_t_out")),
                "direct_phi": _as_float(interface.get("direct_phi")),
                "tex_total_ratio": _as_float(summary.get("tex_total_ratio")),
                "elastic_ratio": _as_float(ratios.get("elastic_ratio")),
                "contact_ratio": _as_float(ratios.get("contact_ratio")),
                "tilt_out": _as_float(breakdown.get("tilt_out")),
                "bending_tilt_out": _as_float(breakdown.get("bending_tilt_out")),
                "stop_reason": str(relax.get("stop_reason") or ""),
                "cg_fallback_accepted_count": int(
                    relax.get("cg_fallback_accepted_count", 0) or 0
                ),
                "role_field_summary": _scaffold_role_field_summary(ctx),
                "role_gradient_summary": _role_vector_summaries(ctx, combined_grad),
                "role_update_summary": _role_vector_summaries(ctx, update_delta),
                "support_continuation_probe": _support_continuation_probe(ctx),
            }
            row["classification"] = _scaffold_collapse_row_classification(row)
            rows.append(row)
    return {"rows": rows, "summary": _support_ownership_summary(rows)}


def _full_physics_trace_convergence(
    *,
    tmpdir: Path,
    protocol: tuple[str, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    variants: list[tuple[str, float | None]] = [("no_trace_current", None)]
    variants.extend(("trace_only", eps) for eps in TRACE_CONVERGENCE_EPSILONS)
    variants.append(("fixed_support", 0.005))
    variants.append(("gapfill_release", 0.005))
    for geometry, epsilon in variants:
        doc, label = _trace_convergence_current_geometry_doc(
            geometry=geometry,
            epsilon=epsilon,
        )
        mesh_path = _write_temp_fixture(doc, tmpdir, label)
        ctx, summary = _run_protocol_summary(mesh_path=mesh_path, protocol=protocol)
        shell_rows = _outer_shell_rows(ctx.mesh)
        theta = float(summary.get("thetaB_value") or 0.0)
        coupling = _outer_bending_tilt_gradient_components(
            ctx=ctx,
            div_term_sign=1.0,
            pullback_sign=1.0,
        )
        interface = dict(summary.get("interface_summary", {}))
        geometry_summary = dict(summary.get("outer_shell_geometry", {}))
        ratios = dict(summary.get("tex_ratio_summary", {}))
        breakdown = dict(summary.get("energy_breakdown", {}))
        relax = dict(summary.get("leaflet_relaxation_stats", {}))
        part = _outer_participation_snapshot(ctx.mesh, ctx=ctx)
        update = _one_step_shell_update_summary(ctx=ctx, theta=theta)
        row = {
            "label": label,
            "geometry": geometry,
            "epsilon": None if epsilon is None else float(epsilon),
            "thetaB_value": _as_float(summary.get("thetaB_value")),
            "model_intent": str(
                summary.get("report_metadata", {}).get("model_intent") or ""
            ),
            "reference_mode": str(
                summary.get("report_metadata", {}).get("reference_mode") or ""
            ),
            "tex_total_ratio": _as_float(summary.get("tex_total_ratio")),
            "elastic_ratio": _as_float(ratios.get("elastic_ratio")),
            "contact_ratio": _as_float(ratios.get("contact_ratio")),
            "direct_t_out": _as_float(interface.get("direct_t_out")),
            "direct_phi": _as_float(interface.get("direct_phi")),
            "interface_source": str(interface.get("primary_source") or ""),
            "trace_source": str(interface.get("target_source") or ""),
            "trace_radius": _as_float(geometry_summary.get("outer_radius")),
            "delta_r": _as_float(geometry_summary.get("delta_r")),
            "tilt_out": _as_float(breakdown.get("tilt_out")),
            "bending_tilt_out": _as_float(breakdown.get("bending_tilt_out")),
            "shell_grad_norm": _as_float(
                _shell_vector_summary(
                    ctx.mesh, coupling["combined_gradient"], shell_rows
                ).get("norm")
            ),
            "shell_update_norm": _as_float(update.get("norm")),
            "shell_base_mean": _as_float(
                coupling.get("summary", {}).get("base_term_outer_shell_mean")
            ),
            "outer_shell_free_count": int(part.get("outer_shell_free_count", 0)),
            "active_outer_area_rows": int(relax.get("active_outer_area_rows", 0)),
            "stop_reason": str(relax.get("stop_reason") or ""),
            "final_gradient_norm": _as_float(relax.get("final_gradient_norm")),
        }
        row["classification"] = _trace_convergence_row_classification(row)
        rows.append(row)
    return {"rows": rows, "summary": _trace_convergence_summary(rows)}


def _ranked_hypotheses(report: dict[str, Any]) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    scan_rows = report.get("thetaB_scan_sensitivity_matrix", {}).get("rows", [])
    trap_count = sum(
        1 for row in scan_rows if row.get("classification") == "local_thetaB_scan_trap"
    )
    if trap_count:
        hypotheses.append(
            {
                "label": "local_thetaB_scan_trap",
                "score": int(trap_count),
                "evidence": f"{trap_count} scan cells prefer a wider-grid theta over the local base +/- delta sample.",
            }
        )

    theta_summaries = report.get("fixed_theta_relaxation_matrix", {}).get(
        "theta_summaries", []
    )
    under_relaxed_count = sum(
        1 for row in theta_summaries if row.get("classification") == "under_relaxed"
    )
    if under_relaxed_count:
        hypotheses.append(
            {
                "label": "scan_under_relaxed",
                "score": int(under_relaxed_count),
                "evidence": f"{under_relaxed_count} fixed-theta traces still gain elastic response as the tilt relaxation budget increases.",
            }
        )
    cancellation_count = sum(
        1
        for row in theta_summaries
        if row.get("classification") == "outer_canceled_by_inner"
    )
    if cancellation_count:
        hypotheses.append(
            {
                "label": "coupled_relaxation_cancellation",
                "score": int(cancellation_count),
                "evidence": f"{cancellation_count} fixed-theta traces show outer response peaking before later relaxation passes.",
            }
        )

    line_rows = report.get("line_search_interaction", [])
    if line_rows:
        theta_values = np.asarray(
            [_as_float(row.get("thetaB_value")) for row in line_rows], dtype=float
        )
        if theta_values.size and (
            float(np.max(theta_values) - np.min(theta_values)) > 0.015
        ):
            hypotheses.append(
                {
                    "label": "line_search_reduced_relaxation_interference",
                    "score": 1,
                    "evidence": "Varying line_search_reduced_tilt_inner_steps changes the converged thetaB materially.",
                }
            )

    state_path = _classify_state_path_report(report)
    if bool(state_path.get("fresh_anchor_mismatch")):
        hypotheses.append(
            {
                "label": "fresh_anchor_mismatch",
                "score": 2,
                "evidence": "Fresh fixed-theta replays recover more outer elastic response than optimized-anchor replays at matched theta and budget.",
            }
        )
    if bool(state_path.get("projection_erases_outer_field")):
        hypotheses.append(
            {
                "label": "projection_erases_outer_field",
                "score": 2,
                "evidence": "Outer shell magnitude drops materially while projection remains active and shell participation stays available.",
            }
        )
    if bool(state_path.get("outer_rows_not_free")):
        hypotheses.append(
            {
                "label": "outer_rows_not_free",
                "score": 2,
                "evidence": "Outer shell rows exist but remain fixed during leaflet relaxation.",
            }
        )
    if bool(state_path.get("outer_area_suppressed")):
        hypotheses.append(
            {
                "label": "outer_area_suppressed",
                "score": 2,
                "evidence": "Outer shell rows exist but their effective outer vertex area is zero.",
            }
        )
    if bool(state_path.get("gradient_stall")):
        hypotheses.append(
            {
                "label": "gradient_stall",
                "score": 1,
                "evidence": "Leaflet relaxation stops with high residual gradient while outer response remains small.",
            }
        )

    if not hypotheses:
        hypotheses.append(
            {
                "label": "true_reduced_energy_prefers_low_theta",
                "score": 0,
                "evidence": "No cadence probe produced a stronger alternative to the low-theta optimized state.",
            }
        )
    hypotheses.sort(key=lambda row: int(row.get("score", 0)), reverse=True)
    return hypotheses


def _classify_state_path_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("state_path_comparison_matrix", {}).get("rows", [])
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("requested_thetaB_value")),
            int(row.get("relax_steps", 0)),
        )
        by_key.setdefault(key, {})[str(row.get("warm_start_policy"))] = row

    fresh_anchor_mismatch = False
    projection_erases_outer_field = False
    outer_rows_not_free = False
    outer_area_suppressed = False
    gradient_stall = False
    for policy_rows in by_key.values():
        fresh = policy_rows.get("fresh_fixture")
        anchor = policy_rows.get("anchor_optimized")
        if fresh and anchor:
            fresh_outer = _as_float(
                fresh.get("energy_breakdown", {}).get("tilt_out")
            ) + _as_float(fresh.get("energy_breakdown", {}).get("bending_tilt_out"))
            anchor_outer = _as_float(
                anchor.get("energy_breakdown", {}).get("tilt_out")
            ) + _as_float(anchor.get("energy_breakdown", {}).get("bending_tilt_out"))
            if fresh_outer > max(anchor_outer * 3.0, anchor_outer + 0.01):
                fresh_anchor_mismatch = True
        for row in policy_rows.values():
            relax = row.get("leaflet_relaxation_stats", {})
            part = row.get("outer_participation", {})
            shell = row.get("outer_shell_field", {})
            if (
                int(part.get("outer_shell_row_count", 0)) > 0
                and int(part.get("outer_shell_free_count", 0)) == 0
            ):
                outer_rows_not_free = True
            if (
                int(part.get("outer_shell_row_count", 0)) > 0
                and int(relax.get("active_outer_area_rows", 0)) == 0
            ):
                outer_area_suppressed = True
            if (
                _as_float(relax.get("initial_gradient_norm")) > 1.0e-6
                and _as_float(relax.get("final_gradient_norm"))
                > 0.9 * _as_float(relax.get("initial_gradient_norm"))
                and str(relax.get("stop_reason"))
                in {"line_search_rejected", "completed_max_iters"}
            ):
                gradient_stall = True
            norm_ref = _as_float(relax.get("tilt_projection_norm_ref_outer_far"))
            norm_loss = _as_float(relax.get("tilt_projection_norm_loss_outer_far"))
            if norm_ref > 0.0 and norm_loss > 0.5 * norm_ref:
                projection_erases_outer_field = True
            if (
                int(shell.get("count", 0)) > 0
                and _as_float(shell.get("tilt_out_norm_mean")) < 1.0e-9
                and int(relax.get("outer_shell_row_count", 0)) > 0
            ):
                projection_erases_outer_field = projection_erases_outer_field or (
                    _as_float(relax.get("projection_apply_count")) > 0
                )
    return {
        "fresh_anchor_mismatch": bool(fresh_anchor_mismatch),
        "projection_erases_outer_field": bool(projection_erases_outer_field),
        "outer_rows_not_free": bool(outer_rows_not_free),
        "outer_area_suppressed": bool(outer_area_suppressed),
        "gradient_stall": bool(gradient_stall),
    }


def _classify_report(report: dict[str, Any]) -> dict[str, Any]:
    scan_rows = report.get("thetaB_scan_sensitivity_matrix", {}).get("rows", [])
    theta_summaries = report.get("fixed_theta_relaxation_matrix", {}).get(
        "theta_summaries", []
    )
    line_rows = report.get("line_search_interaction", [])
    classifications = {
        "local_thetaB_scan_trap": any(
            row.get("classification") == "local_thetaB_scan_trap" for row in scan_rows
        ),
        "scan_under_relaxed": any(
            row.get("classification") == "under_relaxed" for row in theta_summaries
        ),
        "coupled_relaxation_cancellation": any(
            row.get("classification") == "outer_canceled_by_inner"
            for row in theta_summaries
        ),
        "line_search_reduced_relaxation_interference": False,
    }
    if line_rows:
        theta_values = np.asarray(
            [_as_float(row.get("thetaB_value")) for row in line_rows], dtype=float
        )
        if theta_values.size:
            classifications["line_search_reduced_relaxation_interference"] = (
                float(np.max(theta_values) - np.min(theta_values)) > 0.015
            )
    classifications.update(_classify_state_path_report(report))
    if not any(classifications.values()):
        classifications["true_reduced_energy_prefers_low_theta"] = True
    return classifications


def run_audit(
    *, mode: str = "run", protocol: tuple[str, ...] = DEFAULT_PROTOCOL
) -> dict[str, Any]:
    if mode == "schema":
        return {
            "meta": {"mode": "schema", "protocol": list(protocol)},
            "sections": [
                "optimized_trace_replay",
                "fixed_theta_relaxation_matrix",
                "thetaB_scan_sensitivity_matrix",
                "state_path_comparison_matrix",
                "single_step_relaxation_trace",
                "single_pass_outer_survival_trace",
                "thetaB_candidate_state_delta",
                "relaxation_solver_path",
                "full_physics_lane_matrix",
                "full_physics_trace_convergence",
                "full_physics_scaffold_collapse_probe",
                "full_physics_scaffold_support_ownership_probe",
                "trace_continuation_landscape_probe",
                "bending_tilt_out_scaffold_interface_audit",
                "bending_tilt_out_divergence_conditioning_audit",
                "scaffold_geometry_spacing_probe",
                "outer_energy_gradient_assembly",
                "runtime_gradient_bridge",
                "outer_coupling_sign_sweep",
                "base_term_reference_sweep",
                "line_search_interaction",
                "classification",
                "ranked_hypotheses",
            ],
            "defaults": {
                "fixed_theta_grid": [float(x) for x in FIXED_THETA_GRID],
                "relaxation_budgets": [int(x) for x in RELAXATION_BUDGETS],
                "scan_deltas": [float(x) for x in SCAN_DELTAS],
                "scan_inner_steps": [int(x) for x in SCAN_INNER_STEPS],
                "main_inner_steps": [int(x) for x in MAIN_INNER_STEPS],
                "trace_thetas": [float(x) for x in TRACE_THETAS],
                "trace_passes": int(TRACE_PASSES),
                "line_search_reduced_steps": [
                    int(x) for x in LINE_SEARCH_REDUCED_STEPS
                ],
                "warm_start_policies": list(WARM_START_POLICIES),
                "state_path_thetas": [float(x) for x in STATE_PATH_THETAS],
                "state_path_relaxation_budgets": [
                    int(x) for x in STATE_PATH_RELAXATION_BUDGETS
                ],
                "survival_trace_thetas": [float(x) for x in SURVIVAL_TRACE_THETAS],
                "survival_trace_passes": int(SURVIVAL_TRACE_PASSES),
                "survival_trace_policies": list(SURVIVAL_TRACE_POLICIES),
                "solver_path_thetas": [float(x) for x in SOLVER_PATH_THETAS],
                "solver_path_inner_steps": int(SOLVER_PATH_INNER_STEPS),
                "solver_path_variants": [row["label"] for row in SOLVER_PATH_VARIANTS],
                "assembly_thetas": [float(x) for x in ASSEMBLY_THETAS],
                "runtime_bridge_thetas": [float(x) for x in RUNTIME_BRIDGE_THETAS],
                "full_physics_lane_variants": [
                    str(row["label"]) for row in _full_physics_lane_specs()
                ],
                "trace_convergence_epsilons": [
                    float(x) for x in TRACE_CONVERGENCE_EPSILONS
                ],
                "trace_convergence_geometries": list(TRACE_CONVERGENCE_GEOMETRIES),
                "scaffold_collapse_geometries": list(SCAFFOLD_COLLAPSE_GEOMETRIES),
                "scaffold_collapse_variants": list(SCAFFOLD_COLLAPSE_VARIANTS),
                "scaffold_support_ownership_variants": list(
                    SCAFFOLD_SUPPORT_OWNERSHIP_VARIANTS
                ),
                "trace_continuation_landscape_variants": list(
                    TRACE_CONTINUATION_LANDSCAPE_VARIANTS
                ),
                "trace_continuation_landscape_modes": list(
                    TRACE_CONTINUATION_LANDSCAPE_MODES
                ),
                "trace_continuation_landscape_alphas": [
                    float(x) for x in TRACE_CONTINUATION_LANDSCAPE_ALPHAS
                ],
                "bending_tilt_out_interface_reference_modes": list(
                    BENDING_TILT_OUT_INTERFACE_REFERENCE_MODES
                ),
                "scaffold_geometry_spacing_variants": [
                    str(row["label"]) for row in SCAFFOLD_GEOMETRY_SPACING_VARIANTS
                ],
                "outer_coupling_sweep_thetas": [
                    float(x) for x in OUTER_COUPLING_SWEEP_THETAS
                ],
                "outer_coupling_sign_variants": [
                    str(row["label"]) for row in OUTER_COUPLING_SIGN_VARIANTS
                ],
                "base_term_reference_variants": [
                    str(row["label"]) for row in BASE_TERM_REFERENCE_VARIANTS
                ],
            },
        }

    if mode == "scaffold_geometry":
        with tempfile.TemporaryDirectory(prefix="thetaB_scaffold_geometry_") as tmp:
            tmpdir = Path(tmp)
            spacing = _scaffold_geometry_spacing_probe(tmpdir=tmpdir, protocol=protocol)
        return {
            "meta": {"mode": "scaffold_geometry", "protocol": list(protocol)},
            "scaffold_geometry_spacing_probe": spacing,
        }

    with tempfile.TemporaryDirectory(prefix="thetaB_cadence_audit_") as tmp:
        tmpdir = Path(tmp)
        optimized = _optimized_trace_replay(tmpdir=tmpdir, protocol=protocol)
        mesh_path, ctx, snapshot, anchor_summary = _build_default_anchor(
            tmpdir=tmpdir, protocol=protocol
        )
        fresh_snapshot = _build_default_fresh_snapshot(
            mesh_path=mesh_path, anchor_snapshot=snapshot
        )
        fixed_theta_matrix = _fixed_theta_relaxation_matrix(
            ctx=ctx,
            anchor_snapshot=snapshot,
            mesh_path=mesh_path,
            protocol=protocol,
        )
        scan_matrix = _thetaB_scan_sensitivity_matrix(
            ctx=ctx,
            anchor_snapshot=snapshot,
            mesh_path=mesh_path,
            protocol=protocol,
        )
        state_path_matrix = _state_path_comparison_matrix(
            ctx=ctx,
            anchor_snapshot=snapshot,
            fresh_snapshot=fresh_snapshot,
            mesh_path=mesh_path,
            protocol=protocol,
        )
        single_step = _single_step_relaxation_trace(
            ctx=ctx,
            anchor_snapshot=snapshot,
            mesh_path=mesh_path,
            protocol=protocol,
        )
        single_pass_outer = _single_pass_outer_survival_trace(
            ctx=ctx,
            anchor_snapshot=snapshot,
            fresh_snapshot=fresh_snapshot,
            mesh_path=mesh_path,
            protocol=protocol,
        )
        candidate_delta = _thetaB_candidate_state_delta(
            ctx=ctx,
            anchor_snapshot=snapshot,
            mesh_path=mesh_path,
            protocol=protocol,
        )
        solver_path_rows = _relaxation_solver_path(
            ctx=ctx,
            anchor_snapshot=snapshot,
            mesh_path=mesh_path,
            protocol=protocol,
        )
        full_physics_rows = _full_physics_lane_matrix(
            tmpdir=tmpdir,
            protocol=protocol,
        )
        trace_convergence = _full_physics_trace_convergence(
            tmpdir=tmpdir,
            protocol=protocol,
        )
        scaffold_collapse = _full_physics_scaffold_collapse_probe(
            tmpdir=tmpdir,
            protocol=protocol,
        )
        support_ownership = _full_physics_scaffold_support_ownership_probe(
            tmpdir=tmpdir,
            protocol=protocol,
        )
        trace_landscape = _trace_continuation_landscape_probe(
            tmpdir=tmpdir,
            protocol=protocol,
        )
        btl_out_interface = trace_landscape.pop(
            "bending_tilt_out_interface_audit",
            {"rows": [], "summary": _bending_tilt_out_interface_summary([])},
        )
        btl_out_conditioning = trace_landscape.pop(
            "bending_tilt_out_divergence_conditioning_audit",
            {
                "rows": [],
                "summary": _bending_tilt_out_divergence_conditioning_summary([]),
            },
        )
        assembly_rows = _outer_energy_gradient_assembly(
            ctx=ctx,
            anchor_snapshot=snapshot,
            mesh_path=mesh_path,
            protocol=protocol,
        )
        bridge_rows = _runtime_gradient_bridge(
            ctx=ctx,
            anchor_snapshot=snapshot,
            mesh_path=mesh_path,
            protocol=protocol,
        )
        coupling_rows = _outer_coupling_sign_sweep(
            ctx=ctx,
            anchor_snapshot=snapshot,
            mesh_path=mesh_path,
            protocol=protocol,
        )
        reference_rows = _base_term_reference_sweep(
            ctx=ctx,
            anchor_snapshot=snapshot,
            mesh_path=mesh_path,
            protocol=protocol,
        )
        line_search_rows = _line_search_interaction(tmpdir=tmpdir, protocol=protocol)

    report = {
        "meta": {"mode": "run", "protocol": list(protocol)},
        "optimized_trace_replay": optimized,
        "anchor_state": {
            "thetaB_value": _as_float(anchor_summary.get("thetaB_value")),
            "tex_total_ratio": _as_float(anchor_summary.get("tex_total_ratio")),
            "elastic_total_from_breakdown": _as_float(
                anchor_summary.get("elastic_total_from_breakdown")
            ),
        },
        "fixed_theta_relaxation_matrix": fixed_theta_matrix,
        "thetaB_scan_sensitivity_matrix": scan_matrix,
        "state_path_comparison_matrix": state_path_matrix,
        "single_step_relaxation_trace": single_step,
        "single_pass_outer_survival_trace": single_pass_outer,
        "thetaB_candidate_state_delta": candidate_delta,
        "relaxation_solver_path": solver_path_rows,
        "full_physics_lane_matrix": full_physics_rows,
        "full_physics_trace_convergence": trace_convergence,
        "full_physics_scaffold_collapse_probe": scaffold_collapse,
        "full_physics_scaffold_support_ownership_probe": support_ownership,
        "trace_continuation_landscape_probe": trace_landscape,
        "bending_tilt_out_scaffold_interface_audit": btl_out_interface,
        "bending_tilt_out_divergence_conditioning_audit": btl_out_conditioning,
        "outer_energy_gradient_assembly": assembly_rows,
        "runtime_gradient_bridge": bridge_rows,
        "outer_coupling_sign_sweep": coupling_rows,
        "base_term_reference_sweep": reference_rows,
        "line_search_interaction": line_search_rows,
    }
    report["classification"] = _classify_report(report)
    report["ranked_hypotheses"] = _ranked_hypotheses(report)
    return report


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# thetaB Cadence / Relaxation Audit",
        "",
        f"- Mode: `{report.get('meta', {}).get('mode', 'unknown')}`",
        f"- Protocol: `{' '.join(str(x) for x in report.get('meta', {}).get('protocol', []))}`",
        "",
        "## Optimized Replay",
        "",
        "| Variant | thetaB | tex ratio | elastic total | tilt_out | bending_tilt_out |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("optimized_trace_replay", []):
        breakdown = row.get("energy_breakdown", {})
        lines.append(
            f"| `{row.get('label')}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('tex_total_ratio'))} | {_fmt(row.get('elastic_total_from_breakdown'))} | {_fmt(breakdown.get('tilt_out'))} | {_fmt(breakdown.get('bending_tilt_out'))} |"
        )

    lines.extend(
        [
            "",
            "## Top Hypotheses",
            "",
        ]
    )
    for row in report.get("ranked_hypotheses", []):
        lines.append(f"- `{row.get('label')}`: {row.get('evidence')}")

    lines.extend(
        [
            "",
            "## Classification",
            "",
        ]
    )
    for key, value in (report.get("classification") or {}).items():
        lines.append(f"- `{key}`: `{bool(value)}`")

    fixed_summaries = report.get("fixed_theta_relaxation_matrix", {}).get(
        "theta_summaries", []
    )
    if fixed_summaries:
        lines.extend(
            [
                "",
                "## Fixed Theta Summary",
                "",
                "| thetaB | classification | high-budget elastic | high-budget tilt_out | high-budget bending_tilt_out |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in fixed_summaries:
            budget_rows = row.get("budget_rows", [])
            last = budget_rows[-1] if budget_rows else {}
            lines.append(
                f"| `{row.get('theta_label')}` | `{row.get('classification')}` | {_fmt(last.get('elastic_total_from_breakdown'))} | {_fmt(last.get('tilt_out'))} | {_fmt(last.get('bending_tilt_out'))} |"
            )

    state_rows = report.get("state_path_comparison_matrix", {}).get("rows", [])
    if state_rows:
        lines.extend(
            [
                "",
                "## State Path Summary",
                "",
                "| policy | thetaB | steps | elastic total | tilt_out | bending_tilt_out | shell tilt_out mean | stop reason | final grad |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for row in state_rows[:16]:
            shell = row.get("outer_shell_field", {})
            relax = row.get("leaflet_relaxation_stats", {})
            breakdown = row.get("energy_breakdown", {})
            lines.append(
                f"| `{row.get('warm_start_policy')}` | {_fmt(row.get('requested_thetaB_value'))} | {int(row.get('relax_steps', 0))} | {_fmt(row.get('elastic_total_from_breakdown'))} | {_fmt(breakdown.get('tilt_out'))} | {_fmt(breakdown.get('bending_tilt_out'))} | {_fmt(shell.get('tilt_out_norm_mean'))} | `{relax.get('stop_reason', 'n/a')}` | {_fmt(relax.get('final_gradient_norm'))} |"
            )

    candidate_rows = report.get("thetaB_candidate_state_delta", [])
    if candidate_rows:
        lines.extend(
            [
                "",
                "## Candidate Delta",
                "",
                "| thetaB | elastic total | shell tilt_out delta | shell radial delta | stop reason |",
                "| ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in candidate_rows:
            delta = row.get("candidate_delta", {})
            relax = row.get("leaflet_relaxation_stats", {})
            lines.append(
                f"| {_fmt(row.get('requested_thetaB_value'))} | {_fmt(row.get('elastic_total_from_breakdown'))} | {_fmt(delta.get('tilt_out_norm_mean_delta'))} | {_fmt(delta.get('tilt_out_radial_mean_delta'))} | `{relax.get('stop_reason', 'n/a')}` |"
            )

    solver_rows = report.get("relaxation_solver_path", [])
    if solver_rows:
        lines.extend(
            [
                "",
                "## Solver Path",
                "",
                "| variant | thetaB | elastic total | tilt_out | bending_tilt_out | grad out shell before | grad out shell after | update out shell | precond out shell | stop reason |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in solver_rows:
            relax = row.get("leaflet_relaxation_stats", {})
            before = relax.get("gradient_norms_before_constraints", {}).get("out", {})
            after = relax.get("gradient_norms_after_constraints", {}).get("out", {})
            updates = relax.get("accepted_update_norms_out", {})
            precond = relax.get("preconditioner_mean_inv_out", {})
            breakdown = row.get("energy_breakdown", {})
            lines.append(
                f"| `{row.get('solver_path_label')}` | {_fmt(row.get('requested_thetaB_value'))} | {_fmt(row.get('elastic_total_from_breakdown'))} | {_fmt(breakdown.get('tilt_out'))} | {_fmt(breakdown.get('bending_tilt_out'))} | {_fmt(before.get('outer_shell'), 6)} | {_fmt(after.get('outer_shell'), 6)} | {_fmt(updates.get('outer_shell'), 6)} | {_fmt(precond.get('outer_shell'))} | `{relax.get('stop_reason', 'n/a')}` |"
            )

    lane_rows = report.get("full_physics_lane_matrix", [])
    if lane_rows:
        lines.extend(
            [
                "",
                "## Full-Physics Lane Matrix",
                "",
                "| lane | intent | ref mode | thetaB | tex ratio | contact ratio | elastic ratio | shell grad | shell update | shell active rows | shell base mean | stop reason |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in lane_rows:
            ratio = row.get("tex_ratio_summary", {})
            combined = row.get("combined_shell_summary", {})
            update = row.get("first_shell_update_summary", {})
            coupling = row.get("bending_coupling_summary", {})
            part = row.get("outer_participation", {})
            relax = row.get("leaflet_relaxation_stats", {})
            lines.append(
                f"| `{row.get('label')}` | `{row.get('model_intent')}` | `{row.get('reference_mode')}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('tex_total_ratio'))} | {_fmt(ratio.get('contact_ratio'))} | {_fmt(ratio.get('elastic_ratio'))} | {_fmt(combined.get('norm'), 6)} | {_fmt(update.get('norm'), 6)} | {int(part.get('outer_shell_free_count', 0))} | {_fmt(coupling.get('base_term_outer_shell_mean'), 6)} | `{relax.get('stop_reason', 'n/a')}` |"
            )

    trace_conv = report.get("full_physics_trace_convergence", {})
    trace_rows = trace_conv.get("rows", [])
    if trace_rows:
        lines.extend(
            [
                "",
                "## Full-Physics Trace Convergence",
                "",
                "| variant | eps | geometry | thetaB | direct_t_out | direct_phi | shell grad | shell update | contact ratio | elastic ratio | classification |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in trace_rows:
            lines.append(
                f"| `{row.get('label')}` | {_fmt(row.get('epsilon'))} | `{row.get('geometry')}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('direct_t_out'))} | {_fmt(row.get('direct_phi'))} | {_fmt(row.get('shell_grad_norm'), 6)} | {_fmt(row.get('shell_update_norm'), 6)} | {_fmt(row.get('contact_ratio'))} | {_fmt(row.get('elastic_ratio'))} | `{row.get('classification')}` |"
            )
        lines.append("")
        lines.append(
            f"- Trace convergence summary: `{trace_conv.get('summary', {}).get('classification', 'n/a')}`"
        )

    scaffold_probe = report.get("full_physics_scaffold_collapse_probe", {})
    scaffold_rows = scaffold_probe.get("rows", [])
    if scaffold_rows:
        lines.extend(
            [
                "",
                "## Full-Physics Scaffold Collapse Probe",
                "",
                "| geometry | variant | thetaB | direct_t_out | trace t_out | support t_out | shell grad | shell update | gd descent | gd best dE | gd trace update | cg fallback | projector | scans | stop reason | classification |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
            ]
        )
        for row in scaffold_rows:
            fields = row.get("role_field_summary", {})
            trace_field = fields.get("trace", {})
            support_field = fields.get("support", {})
            gd_probe = row.get("gd_line_search_probe", {})
            best = gd_probe.get("best_sample", {})
            lines.append(
                f"| `{row.get('geometry')}` | `{row.get('variant')}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('direct_t_out'))} | {_fmt(trace_field.get('tilt_out_radial_mean'))} | {_fmt(support_field.get('tilt_out_radial_mean'))} | {_fmt(row.get('shell_grad_norm'), 6)} | {_fmt(row.get('shell_update_norm'), 6)} | `{bool(gd_probe.get('has_gd_descent_step'))}` | {_fmt(best.get('tilt_dependent_delta'), 6)} | {_fmt(best.get('trace_update_radial_mean'), 6)} | {int(row.get('cg_fallback_accepted_count', 0))} | `{row.get('projector_mode')}` | {int(row.get('thetaB_scan_count', 0))} | `{row.get('stop_reason', 'n/a')}` | `{row.get('classification')}` |"
            )
        lines.append("")
        lines.append(
            f"- Scaffold collapse summary: `{scaffold_probe.get('summary', {}).get('classification', 'n/a')}`"
        )

    support_probe = report.get("full_physics_scaffold_support_ownership_probe", {})
    support_rows = support_probe.get("rows", [])
    if support_rows:
        lines.extend(
            [
                "",
                "## Full-Physics Scaffold Support Ownership",
                "",
                "| geometry | variant | thetaB | direct_t_out | trace t_out | support t_out | trace phi | support phi | trace grad | support grad | trace update | support update | cont dE | stop reason | classification |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in support_rows:
            fields = row.get("role_field_summary", {})
            trace_field = fields.get("trace", {})
            support_field = fields.get("support", {})
            grads = row.get("role_gradient_summary", {})
            updates = row.get("role_update_summary", {})
            continuation = row.get("support_continuation_probe", {}).get(
                "best_sample", {}
            )
            lines.append(
                f"| `{row.get('geometry')}` | `{row.get('variant')}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('direct_t_out'))} | {_fmt(trace_field.get('tilt_out_radial_mean'))} | {_fmt(support_field.get('tilt_out_radial_mean'))} | {_fmt(trace_field.get('phi_to_inner'))} | {_fmt(support_field.get('phi_to_inner'))} | {_fmt(grads.get('trace', {}).get('norm'), 6)} | {_fmt(grads.get('support', {}).get('norm'), 6)} | {_fmt(updates.get('trace', {}).get('norm'), 6)} | {_fmt(updates.get('support', {}).get('norm'), 6)} | {_fmt(continuation.get('tilt_dependent_delta'), 6)} | `{row.get('stop_reason', 'n/a')}` | `{row.get('classification')}` |"
            )
        lines.append("")
        lines.append(
            f"- Support ownership summary: `{support_probe.get('summary', {}).get('classification', 'n/a')}`"
        )

    landscape = report.get("trace_continuation_landscape_probe", {})
    landscape_rows = landscape.get("rows", [])
    if landscape_rows:
        lines.extend(
            [
                "",
                "## Trace Continuation Landscape",
                "",
                "| variant | geometry | mode | alpha | thetaB | trace t_out | support t_out | dE tilt-dependent | dE tilt_out | dE bending_tilt_out | dominant positive |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in landscape_rows:
            theta = row.get("thetaB_value")
            for sample in row.get("samples", []):
                terms = sample.get("term_deltas", {})
                lines.append(
                    f"| `{row.get('variant')}` | `{row.get('geometry')}` | `{sample.get('mode')}` | {_fmt(sample.get('alpha'))} | {_fmt(theta)} | {_fmt(sample.get('trace_radial_before'))} | {_fmt(sample.get('support_radial_before'))} | {_fmt(sample.get('tilt_dependent_delta'), 6)} | {_fmt(terms.get('tilt_out'), 6)} | {_fmt(terms.get('bending_tilt_out'), 6)} | `{sample.get('dominant_positive_term')}` |"
                )
        lines.append("")
        lines.append(
            f"- Landscape summary: dominant suppressing term `{landscape.get('summary', {}).get('most_common_suppressing_term', 'n/a')}`"
        )

    btl_interface = report.get("bending_tilt_out_scaffold_interface_audit", {})
    btl_interface_rows = btl_interface.get("rows", [])
    if btl_interface_rows:
        lines.extend(
            [
                "",
                "## Bending Tilt Out Scaffold Interface",
                "",
                "| variant | geometry | reference | role | thetaB | direct t_out | triangles | area | total | base | divergence | cross | base mean | div mean |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in btl_interface_rows:
            for decomposition in row.get("decompositions", []):
                reference = decomposition.get("reference_mode")
                for role, role_data in decomposition.get("roles", {}).items():
                    lines.append(
                        f"| `{row.get('variant')}` | `{row.get('geometry')}` | `{reference}` | `{role}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('direct_t_out'))} | {int(role_data.get('triangle_count', 0))} | {_fmt(role_data.get('area'), 6)} | {_fmt(role_data.get('total_energy'), 6)} | {_fmt(role_data.get('base_energy'), 6)} | {_fmt(role_data.get('divergence_energy'), 6)} | {_fmt(role_data.get('cross_energy'), 6)} | {_fmt(role_data.get('base_term_mean'), 6)} | {_fmt(role_data.get('divergence_mean'), 6)} |"
                    )
        lines.append("")
        lines.append(
            f"- Interface decomposition summary: largest current-geometry cross role `{btl_interface.get('summary', {}).get('largest_current_geometry_cross_role', 'n/a')}`"
        )

    conditioning = report.get("bending_tilt_out_divergence_conditioning_audit", {})
    conditioning_rows = conditioning.get("rows", [])
    if conditioning_rows:
        lines.extend(
            [
                "",
                "## Bending Tilt Out Divergence Conditioning",
                "",
                "| variant | geometry | role | thetaB | direct t_out | triangles | area mean | min edge | aspect mean | basis max | div abs mean | div max | trace corner | support corner | disk corner |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in conditioning_rows:
            for role, role_data in row.get("conditioning", {}).get("roles", {}).items():
                corner_roles = role_data.get("corner_components_by_row_role", {})
                lines.append(
                    f"| `{row.get('variant')}` | `{row.get('geometry')}` | `{role}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('direct_t_out'))} | {int(role_data.get('triangle_count', 0))} | {_fmt(role_data.get('area', {}).get('mean'), 6)} | {_fmt(role_data.get('min_edge', {}).get('min'), 6)} | {_fmt(role_data.get('aspect', {}).get('mean'), 6)} | {_fmt(role_data.get('basis_norm', {}).get('max_abs'), 6)} | {_fmt(role_data.get('divergence', {}).get('abs_mean'), 6)} | {_fmt(role_data.get('divergence', {}).get('max_abs'), 6)} | {_fmt(corner_roles.get('trace', {}).get('mean'), 6)} | {_fmt(corner_roles.get('support', {}).get('mean'), 6)} | {_fmt(corner_roles.get('disk', {}).get('mean'), 6)} |"
                )
        lines.append("")
        lines.append(
            f"- Divergence conditioning summary: max divergence `{conditioning.get('summary', {}).get('max_divergence_role', 'n/a')}`, max basis `{conditioning.get('summary', {}).get('max_basis_norm_role', 'n/a')}`"
        )

    spacing = report.get("scaffold_geometry_spacing_probe", {})
    spacing_rows = spacing.get("rows", [])
    if spacing_rows:
        lines.extend(
            [
                "",
                "## Scaffold Geometry Spacing Probe",
                "",
                "| label | geometry | div mode | inner shape mode | mesh op | shape fallback | shells | d | thetaB | direct t_out | bending_tilt_out | trace grad radial | trace update radial | shape descent z | fallback accepted | fallback count | fallback dz | dominant down | dominant up | projection 0.15-> | gap to phi | high-tilt replay | high-geom replay | trace div abs | stop |",
                "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in spacing_rows:
            breakdown = row.get("energy_breakdown", {})
            gradient_probe = row.get("gradient_update_probe", {})
            trace_gradient = (
                gradient_probe.get("combined_gradient", {})
                .get("trace", {})
                .get("radial_mean")
            )
            trace_update = (
                gradient_probe.get("first_update", {})
                .get("trace", {})
                .get("radial_mean")
            )
            high_seed = row.get("high_trace_seed_replay", {})
            projection = row.get("high_trace_constraint_projection", {})
            shape_gradient = row.get("shape_gradient_probe", {})
            shape_trace = shape_gradient.get("roles", {}).get("trace", {})
            module_probe = row.get("shape_gradient_module_probe", {})
            down = module_probe.get("dominant_trace_downward_module", {})
            up = module_probe.get("dominant_trace_upward_module", {})
            geometry_seed = row.get("high_trace_geometry_seed_probe", {})
            fallback = row.get("shape_scaffold_rejected_step_fallback_stats", {})
            lines.append(
                f"| `{row.get('label')}` | `{row.get('geometry')}` | `{row.get('interface_divergence_mode', 'p1_triangle')}` | `{row.get('inner_scaffold_shape_stencil_mode', 'off')}` | `{row.get('scaffold_mesh_operation_mode', 'project')}` | `{row.get('shape_scaffold_rejected_step_fallback', 'off')}` | {int(row.get('outer_shells', 0))} | {_fmt(row.get('outer_shells_d'), 4)} | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('direct_t_out'))} | {_fmt(breakdown.get('bending_tilt_out'))} | {_fmt(trace_gradient, 6)} | {_fmt(trace_update, 6)} | {_fmt(shape_trace.get('descent_z_mean'), 6)} | `{fallback.get('accepted', '')}` | {int(fallback.get('accepted_count', 0) or 0)} | {_fmt(fallback.get('trace_dz_mean'), 6)} | `{down.get('module', '')}:{_fmt(down.get('trace_descent_z_mean'), 4)}` | `{up.get('module', '')}:{_fmt(up.get('trace_descent_z_mean'), 4)}` | {_fmt(projection.get('after_trace_t_out'))} | {_fmt(projection.get('after_gap_to_phi'), 6)} | {_fmt(high_seed.get('relaxed_direct_t_out'))} | {_fmt(geometry_seed.get('relaxed_direct_t_out'))} | {_fmt(row.get('trace_div_abs_mean'), 6)} | `{row.get('stop_reason', 'n/a')}` |"
            )
        lines.append("")
        lines.append(
            f"- Spacing summary: best trace divergence `{spacing.get('summary', {}).get('best_trace_divergence_label', 'n/a')}`, best direct t_out `{spacing.get('summary', {}).get('best_direct_t_out_label', 'n/a')}`, best high-tilt replay `{spacing.get('summary', {}).get('best_high_seed_label', 'n/a')}`, best high-geometry replay `{spacing.get('summary', {}).get('best_geometry_seed_label', 'n/a')}`"
        )
        stage_rows = [
            (row.get("label"), replay_row)
            for row in spacing_rows
            for replay_row in (
                row.get("high_trace_stage_replay_probe", {}).get("rows", [])
            )
        ]
        if stage_rows:
            lines.extend(
                [
                    "",
                    "### High-Geometry Stage Replay",
                    "",
                    "| label | iter | stage | thetaB | trace t_out | trace t_in | trace phi | energy | down | up | step ok | step size |",
                    "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |",
                ]
            )
            for label, replay in stage_rows:
                down = replay.get("dominant_down", {})
                up = replay.get("dominant_up", {})
                lines.append(
                    f"| `{label}` | {int(replay.get('iteration', 0))} | `{replay.get('stage')}` | {_fmt(replay.get('thetaB_value'))} | {_fmt(replay.get('trace_t_out'))} | {_fmt(replay.get('trace_t_in'))} | {_fmt(replay.get('trace_phi'))} | {_fmt(replay.get('energy'))} | `{down.get('module', '')}:{_fmt(down.get('trace_descent_z_mean'), 4)}` | `{up.get('module', '')}:{_fmt(up.get('trace_descent_z_mean'), 4)}` | `{replay.get('shape_step_success', '')}` | {_fmt(replay.get('shape_step_size_out'))} |"
                )
        branch_rows = [
            (row.get("label"), branch_row)
            for row in spacing_rows
            for branch_row in (row.get("branch_access_probe", {}).get("rows", []))
        ]
        if branch_rows:
            lines.extend(
                [
                    "",
                    "### Branch Access Probe",
                    "",
                    "| label | state | thetaB | trace t_out | trace phi | tilt descent radial | first update radial | shape descent z | shape up | shape down |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
                ]
            )
            for label, branch in branch_rows:
                tilt_descent = branch.get("tilt_descent_trace", {})
                first_update = branch.get("first_update_trace", {})
                shape_trace = branch.get("shape_trace", {})
                up = branch.get("shape_dominant_up", {})
                down = branch.get("shape_dominant_down", {})
                lines.append(
                    f"| `{label}` | `{branch.get('label')}` | {_fmt(branch.get('thetaB_value'))} | {_fmt(branch.get('trace_t_out'))} | {_fmt(branch.get('trace_phi'))} | {_fmt(tilt_descent.get('radial_mean'), 6)} | {_fmt(first_update.get('radial_mean'), 6)} | {_fmt(shape_trace.get('descent_z_mean'), 6)} | `{up.get('module', '')}:{_fmt(up.get('trace_descent_z_mean'), 4)}` | `{down.get('module', '')}:{_fmt(down.get('trace_descent_z_mean'), 4)}` |"
                )
        trace_z_trial_rows = [
            (row.get("label"), trial)
            for row in spacing_rows
            for trial in (
                row.get("trace_z_fallback_trial_decomposition_probe", {}).get(
                    "samples", []
                )
            )
        ]
        if trace_z_trial_rows:
            lines.extend(
                [
                    "",
                    "### Trace-Z Fallback Trial Decomposition",
                    "",
                    "| label | alpha | constraints | dE | dominant positive | BT in role | BT out role | trace dz kept | trace phi | support phi |",
                    "| --- | ---: | --- | ---: | --- | --- | --- | ---: | ---: | ---: |",
                ]
            )
            for label, trial in trace_z_trial_rows:
                dominant = trial.get("dominant_positive_delta", {})
                bt_roles = trial.get("bending_tilt_role_deltas", {})
                bt_in = bt_roles.get("bending_tilt_in", {})
                bt_out = bt_roles.get("bending_tilt_out", {})
                lines.append(
                    f"| `{label}` | {_fmt(trial.get('alpha'), 6)} | `{trial.get('constraint_context')}` | {_fmt(trial.get('energy_delta'), 6)} | `{dominant.get('module', '')}:{_fmt(dominant.get('delta'), 6)}` | `{bt_in.get('dominant_positive_role', '')}:{_fmt(bt_in.get('dominant_positive_delta'), 6)}` | `{bt_out.get('dominant_positive_role', '')}:{_fmt(bt_out.get('dominant_positive_delta'), 6)}` | {_fmt(trial.get('trace_dz_preserved_ratio'), 6)} | {_fmt(trial.get('trace_phi_after'), 6)} | {_fmt(trial.get('support_phi_after'), 6)} |"
                )

    assembly_rows = report.get("outer_energy_gradient_assembly", [])
    if assembly_rows:
        lines.extend(
            [
                "",
                "## Gradient Assembly",
                "",
                "| thetaB | tilt_out shell grad | bending_tilt_out shell grad | combined shell grad | cosine | kept shell tris | full shell tris | shell base term mean | shell weight mean |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in assembly_rows:
            tilt_mod = row.get("tilt_out_module", {})
            btl_mod = row.get("bending_tilt_out_module", {})
            tri_counts = btl_mod.get("triangle_counts", {})
            combined = row.get("combined_outer_shell_gradient", {})
            lines.append(
                f"| {_fmt(row.get('requested_thetaB_value'))} | {_fmt(tilt_mod.get('tilt_grad_norm_by_region', {}).get('outer_shell'), 6)} | {_fmt(btl_mod.get('tilt_grad_norm_by_region', {}).get('outer_shell'), 6)} | {_fmt(combined.get('norm'), 6)} | {_fmt(combined.get('cosine'))} | {int(tri_counts.get('kept_touching_outer_shell', 0))} | {int(tri_counts.get('full_touching_outer_shell', 0))} | {_fmt(btl_mod.get('base_term_outer_shell_mean'), 6)} | {_fmt(tilt_mod.get('active_row_weight_mean_outer_shell'))} |"
            )

    bridge_rows = report.get("runtime_gradient_bridge", [])
    if bridge_rows:
        lines.extend(
            [
                "",
                "## Runtime Gradient Bridge",
                "",
                "| thetaB | tilt shell grad | bending shell grad | combined shell grad | tilt/bending cosine | runtime shell grad before | runtime shell grad after | first shell update | direct/runtime cosine | before/after cosine | after/update cosine |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in bridge_rows:
            compare = row.get("shell_vector_comparison", {})
            direct_summary = row.get("direct_module_outer_gradient", {})
            direct = direct_summary.get("tilt_grad_norm_by_region", {})
            before = row.get("runtime_aggregated_gradient_before_constraints", {}).get(
                "tilt_grad_norm_by_region", {}
            )
            after = row.get("runtime_aggregated_gradient_after_constraints", {}).get(
                "tilt_grad_norm_by_region", {}
            )
            update = row.get("accepted_update", {}).get("tilt_grad_norm_by_region", {})
            lines.append(
                f"| {_fmt(row.get('requested_thetaB_value'))} | {_fmt(direct_summary.get('tilt_out_shell_norm'), 6)} | {_fmt(direct_summary.get('bending_tilt_out_shell_norm'), 6)} | {_fmt(direct.get('outer_shell'), 6)} | {_fmt(direct_summary.get('tilt_vs_bending_cosine'))} | {_fmt(before.get('outer_shell'), 6)} | {_fmt(after.get('outer_shell'), 6)} | {_fmt(update.get('outer_shell'), 6)} | {_fmt(compare.get('direct_vs_runtime_before_cosine'))} | {_fmt(compare.get('runtime_before_vs_after_cosine'))} | {_fmt(compare.get('runtime_after_vs_update_cosine'))} |"
            )

    reference_rows = report.get("base_term_reference_sweep", [])
    coupling_rows = report.get("outer_coupling_sign_sweep", [])
    if coupling_rows:
        lines.extend(
            [
                "",
                "## Outer Coupling Sign Sweep",
                "",
                "| variant | thetaB | tilt shell grad | bending shell grad | combined shell grad | tilt/bending cosine | base/div cosine | combined radial | combined tangential | shell base mean | shell div mean | shell descent radial |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in coupling_rows:
            tilt_shell = row.get("tilt_shell_summary", {})
            bending_shell = row.get("bending_shell_summary", {})
            combined_shell = row.get("combined_shell_summary", {})
            descent_shell = row.get("descent_shell_update_summary", {})
            coupling = row.get("bending_coupling_summary", {})
            lines.append(
                f"| `{row.get('variant_label')}` | {_fmt(row.get('requested_thetaB_value'))} | {_fmt(tilt_shell.get('norm'), 6)} | {_fmt(bending_shell.get('norm'), 6)} | {_fmt(combined_shell.get('norm'), 6)} | {_fmt(row.get('tilt_vs_bending_cosine'))} | {_fmt(row.get('base_vs_divergence_cosine'))} | {_fmt(combined_shell.get('radial_norm'), 6)} | {_fmt(combined_shell.get('tangential_norm'), 6)} | {_fmt(coupling.get('base_term_outer_shell_mean'), 6)} | {_fmt(coupling.get('div_eval_outer_shell_mean'), 6)} | {_fmt(descent_shell.get('radial_norm'), 6)} |"
            )

    if reference_rows:
        lines.extend(
            [
                "",
                "## Base-Term Reference Sweep",
                "",
                "| variant | thetaB | shell base mean | tilt shell grad | bending shell grad | combined shell grad | cosine | first shell update | tilt_out | bending_tilt_out | contact ratio | elastic ratio | total ratio |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in reference_rows:
            ratio = row.get("tex_ratio_summary", {})
            breakdown = row.get("energy_breakdown", {})
            combined = row.get("combined_outer_shell_gradient", {})
            lines.append(
                f"| `{row.get('variant_label')}` | {_fmt(row.get('requested_thetaB_value'))} | {_fmt(row.get('outer_shell_base_term_mean'), 6)} | {_fmt(row.get('tilt_out_shell_gradient'), 6)} | {_fmt(row.get('bending_tilt_out_shell_gradient'), 6)} | {_fmt(combined.get('norm'), 6)} | {_fmt(combined.get('cosine'))} | {_fmt(row.get('first_accepted_shell_update_norm'), 6)} | {_fmt(breakdown.get('tilt_out'))} | {_fmt(breakdown.get('bending_tilt_out'))} | {_fmt(ratio.get('contact_ratio'))} | {_fmt(ratio.get('elastic_ratio'))} | {_fmt(ratio.get('total_ratio'))} |"
            )

    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("run", "schema", "scaffold_geometry"), default="run"
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--protocol", nargs="*", default=list(DEFAULT_PROTOCOL))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_audit(
        mode=str(args.mode), protocol=tuple(str(x) for x in args.protocol)
    )
    yaml_text = yaml.safe_dump(report, sort_keys=False)
    md_text = render_markdown_report(report)
    if args.out is None:
        print(yaml_text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml_text, encoding="utf-8")
        print(f"wrote: {args.out}")
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(md_text, encoding="utf-8")
        print(f"wrote: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
