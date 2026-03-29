#!/usr/bin/env python3
"""Reproduce theory-parity metrics and write a YAML report."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import special

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.context import CommandContext
from commands.executor import execute_command_line
from geometry.geom_io import load_data, parse_geometry
from geometry.tilt_operators import p1_vertex_divergence
from modules.constraints.local_interface_shells import (
    build_local_interface_shell_data,
    extrapolate_trace_to_radius,
    radial_unit_vectors,
)
from modules.constraints.rim_slope_match_out import (
    _fit_plane_normal,
    _resolve_center,
    _resolve_normal,
    matching_residual_diagnostics,
    matching_ring_diagnostics,
)
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MESH = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
DEFAULT_OUT = (
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "theory_parity_report.yaml"
)
DEFAULT_EXPANSION_POLICY = (
    ROOT / "tests" / "fixtures" / "theory_parity_expansion_policy.yaml"
)
DEFAULT_EXPANSION_STATE = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "diagnostics"
    / "theory_parity_expansion_state.yaml"
)
DEFAULT_PROTOCOL = ("g10", "r", "V2", "t5e-3", "g8", "t2e-3", "g12")
DEFAULT_THEORY_RADIUS = 7.0 / 15.0
DEFAULT_TEX_BENDING_MODULUS = 1.0
DEFAULT_TEX_TILT_MODULUS = 225.0
DEFAULT_PHYSICAL_EDGE_Z_BUMP = 1.0e-3


def _report_fixture_path(mesh_path: Path) -> str:
    """Return a stable report label for the active fixture path."""
    resolved = Path(mesh_path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _parity_lane_name(mesh_path: Path, mesh) -> str:
    """Return the configured parity lane name for diagnostics and tests."""
    configured = str(mesh.global_parameters.get("theory_parity_lane") or "").strip()
    if configured:
        return configured
    return Path(mesh_path).stem


def _build_context(mesh_path: Path) -> CommandContext:
    mesh = parse_geometry(load_data(str(mesh_path)))
    lane = _parity_lane_name(mesh_path, mesh)
    if not str(mesh.global_parameters.get("theory_parity_lane") or "").strip():
        mesh.global_parameters.set("theory_parity_lane", lane)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    return CommandContext(mesh, minim, minim.stepper)


def _stabilize_rim_radius_for_parity(mesh) -> dict[str, float]:
    """Keep the tagged legacy rim shell on its current circle during parity runs."""
    mode = str(mesh.global_parameters.get("rim_slope_match_mode") or "").strip().lower()
    if mode == "physical_edge_staggered_v1":
        return {"available": 0.0, "radius": 0.0, "count": 0.0}
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    rows: list[int] = []
    changed = False
    for row, vid in enumerate(mesh.vertex_ids):
        opts = dict(getattr(mesh.vertices[int(vid)], "options", None) or {})
        if str(opts.get("rim_slope_match_group") or "") != "rim":
            continue
        rows.append(int(row))
        constraints = list(opts.get("constraints") or [])
        if "pin_to_circle" not in constraints:
            constraints.append("pin_to_circle")
            changed = True
        opts["constraints"] = constraints
        opts["pin_to_circle_group"] = "rim_hold"
        opts["pin_to_circle_normal"] = [0.0, 0.0, 1.0]
        opts["pin_to_circle_point"] = [0.0, 0.0, 0.0]
        mesh.vertices[int(vid)].options = opts
    if not rows:
        return {"available": 0.0, "radius": 0.0, "count": 0.0}

    radius = float(np.median(r[np.asarray(rows, dtype=int)]))
    for row in rows:
        vid = int(mesh.vertex_ids[int(row)])
        opts = dict(getattr(mesh.vertices[vid], "options", None) or {})
        if float(opts.get("pin_to_circle_radius") or 0.0) != float(radius):
            opts["pin_to_circle_radius"] = float(radius)
            mesh.vertices[vid].options = opts
            changed = True
    if changed:
        mesh.increment_version()
    return {"available": 1.0, "radius": float(radius), "count": float(len(rows))}


def _activate_local_outer_shell_for_parity(mesh) -> dict[str, float]:
    """Retag a nearby shell as `outer` for parity diagnostics."""
    mode = str(mesh.global_parameters.get("rim_slope_match_mode") or "").strip().lower()
    if mode == "physical_edge_staggered_v1":
        try:
            shell_data = build_local_interface_shell_data(
                mesh, positions=mesh.positions_view(), group="disk"
            )
        except AssertionError:
            shell_data = None
        if shell_data is not None:
            bump_raw = mesh.global_parameters.get("parity_physical_edge_z_bump")
            bump = DEFAULT_PHYSICAL_EDGE_Z_BUMP if bump_raw is None else float(bump_raw)
            changed = False
            for row in np.asarray(shell_data.rim_rows, dtype=int):
                vid = int(mesh.vertex_ids[int(row)])
                vertex = mesh.vertices[vid]
                if abs(float(vertex.position[2])) < 0.5 * bump:
                    vertex.position[2] = float(bump)
                    changed = True
            if changed:
                mesh.build_position_cache()
        diag = matching_ring_diagnostics(
            mesh, mesh.global_parameters, mesh.positions_view()
        )
        return {
            "available": float(bool(diag.get("available"))),
            "construction_mode": str(
                diag.get("construction_mode") or "physical_edge_local_shell"
            ),
            "rim_radius": float(diag.get("rim_radius") or 0.0),
            "outer_radius": float(diag.get("outer_radius") or 0.0),
            "delta_r": float(
                float(diag.get("outer_radius") or 0.0)
                - float(diag.get("rim_radius") or 0.0)
            ),
            "n_outer_rows": float(diag.get("outer_count") or 0.0),
        }
    local_diag: dict[str, float | str] | None = None
    try:
        shell_data = build_local_interface_shell_data(
            mesh, positions=mesh.positions_view(), group="disk"
        )
    except AssertionError:
        shell_data = None
    if shell_data is not None:
        local_diag = {
            "available": 1.0,
            "construction_mode": "parity_disk_local_shell_measurement",
            "rim_radius": float(shell_data.disk_radius),
            "outer_radius": float(shell_data.rim_radius),
            "delta_r": float(shell_data.rim_radius - shell_data.disk_radius),
            "n_outer_rows": float(shell_data.rim_rows.size),
        }
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    rim_rows: list[int] = []
    changed = False
    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") == "rim":
            rim_rows.append(int(row))
        if opts.get("rim_slope_match_group") == "outer":
            opts = dict(opts)
            opts.pop("rim_slope_match_group", None)
            mesh.vertices[int(vid)].options = opts
            changed = True
    if not rim_rows:
        return {
            "available": 0.0,
            "rim_radius": 0.0,
            "outer_radius": 0.0,
            "delta_r": 0.0,
            "n_outer_rows": 0.0,
        }

    rim_radius = float(np.max(r[np.asarray(rim_rows, dtype=int)]))
    shell_radii = np.unique(np.round(r[r > rim_radius + 1.0e-3], 3))
    if shell_radii.size == 0:
        if changed:
            mesh.increment_version()
        return {
            "available": 0.0,
            "rim_radius": float(rim_radius),
            "outer_radius": 0.0,
            "delta_r": 0.0,
            "n_outer_rows": 0.0,
        }

    outer_radius = float(shell_radii[0])
    outer_rows = np.where(np.isclose(r, outer_radius, atol=1.0e-3))[0]
    for row in outer_rows:
        vid = int(mesh.vertex_ids[int(row)])
        opts = dict(getattr(mesh.vertices[vid], "options", None) or {})
        opts["rim_slope_match_group"] = "outer"
        mesh.vertices[vid].options = opts
        changed = True
    if changed:
        mesh.increment_version()
    legacy_diag = {
        "available": 1.0,
        "construction_mode": "legacy_retagged_outer_shell",
        "rim_radius": float(rim_radius),
        "outer_radius": float(outer_radius),
        "delta_r": float(outer_radius - rim_radius),
        "n_outer_rows": float(len(outer_rows)),
    }
    if local_diag is not None:
        local_diag["legacy_solver_outer_radius"] = float(outer_radius)
        local_diag["legacy_solver_rim_radius"] = float(rim_radius)
        return local_diag
    return legacy_diag


def _tilt_stats_quantiles(mesh) -> dict[str, float]:
    mesh.build_position_cache()
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return {
            "tstat_in_p90_norm": 0.0,
            "tstat_out_p90_norm": 0.0,
        }

    tin = np.asarray(mesh.tilts_in_view(), dtype=float)
    tout = np.asarray(mesh.tilts_out_view(), dtype=float)

    # Keep parity metrics compact and robust.
    mags_in = np.linalg.norm(tin, axis=1)
    mags_out = np.linalg.norm(tout, axis=1)
    _div_in, _ = p1_vertex_divergence(
        n_vertices=len(mesh.vertex_ids),
        positions=positions,
        tilts=tin,
        tri_rows=tri_rows,
    )
    _div_out, _ = p1_vertex_divergence(
        n_vertices=len(mesh.vertex_ids),
        positions=positions,
        tilts=tout,
        tri_rows=tri_rows,
    )

    return {
        "tstat_in_p90_norm": float(np.quantile(mags_in, 0.90)),
        "tstat_out_p90_norm": float(np.quantile(mags_out, 0.90)),
    }


def _relative_rmse(values: np.ndarray, reference: np.ndarray) -> float:
    """Return RMSE normalized by the reference RMS magnitude."""
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if values.size == 0 or reference.size == 0:
        return 0.0
    rmse = float(np.sqrt(np.mean((values - reference) ** 2)))
    ref_scale = float(np.sqrt(np.mean(reference**2)))
    if ref_scale <= 1.0e-12:
        return 0.0 if rmse <= 1.0e-12 else float("inf")
    return float(rmse / ref_scale)


def _quadratic_boundary_slope(
    *,
    target_radius: float,
    boundary_height: np.ndarray,
    first_radius: float,
    first_height: np.ndarray,
    second_radius: float | None = None,
    second_height: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate the outer-side height slope at `target_radius`."""
    z0 = np.asarray(boundary_height, dtype=float)
    z1 = np.asarray(first_height, dtype=float)
    x1 = float(first_radius) - float(target_radius)
    if second_radius is None or second_height is None or abs(x1) <= 1.0e-12:
        return (z1 - z0) / x1 if abs(x1) > 1.0e-12 else np.zeros_like(z1)

    z2 = np.asarray(second_height, dtype=float)
    x2 = float(second_radius) - float(target_radius)
    if abs(x2) <= 1.0e-12 or abs(x2 - x1) <= 1.0e-12:
        return (z1 - z0) / x1

    y1 = z1 - z0
    y2 = z2 - z0
    numer = y1 * (x2**2) - y2 * (x1**2)
    denom = x1 * x2 * (x2 - x1)
    return numer / denom


def _ring_mean_heights(
    *,
    positions: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    min_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return outer-ring radii and mean heights beyond `min_radius`."""
    rel = positions - center[None, :]
    rel_plane = rel - np.einsum("ij,j->i", rel, normal)[:, None] * normal[None, :]
    radii = np.linalg.norm(rel_plane[:, :], axis=1)
    heights = np.einsum("ij,j->i", rel, normal)
    tol = max(1.0e-6, 1.0e-5 * max(1.0, float(min_radius)))
    rounded = np.round(radii, 9)
    unique = np.unique(rounded[rounded > (float(min_radius) + tol)])
    if unique.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    ring_radii: list[float] = []
    ring_heights: list[float] = []
    for radius_token in unique:
        rows = np.flatnonzero(np.isclose(rounded, radius_token, atol=1.0e-9))
        if rows.size == 0:
            continue
        ring_radii.append(float(np.mean(radii[rows])))
        ring_heights.append(float(np.mean(heights[rows])))
    return np.asarray(ring_radii, dtype=float), np.asarray(ring_heights, dtype=float)


def _interface_trace_diagnostics(
    mesh, positions: np.ndarray, theta_meas: float
) -> dict[str, float | bool]:
    """Return one-sided trace diagnostics at the physical disk edge."""
    mode = str(mesh.global_parameters.get("rim_slope_match_mode") or "").strip().lower()
    out: dict[str, float | bool] = {
        "available": False,
        "disk_theta_at_R": 0.0,
        "disk_t_in_at_R": 0.0,
        "outer_t_out_first_shell": 0.0,
        "outer_geometry_trace_at_R_plus": 0.0,
        "outer_t_out_trace_at_R_plus": 0.0,
        "phi_trace_at_R_plus": 0.0,
        "disk_minus_outer_trace": 0.0,
        "disk_minus_phi_trace": 0.0,
        "outer_geometry_vs_tilt_trace_gap": 0.0,
        "outer_first_shell_minus_outer_trace": 0.0,
    }
    if mode != "physical_edge_staggered_v1":
        return out

    try:
        shell_data = build_local_interface_shell_data(
            mesh, positions=positions, group="disk"
        )
    except AssertionError:
        return out

    center = _resolve_center(mesh.global_parameters)
    normal = _resolve_normal(mesh.global_parameters)
    if normal is None:
        normal = _fit_plane_normal(positions[shell_data.disk_rows])
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    disk_rows = np.asarray(shell_data.disk_rows, dtype=int)
    first_rows = np.asarray(shell_data.rim_rows_for_disk, dtype=int)
    second_rows = np.asarray(shell_data.outer_rows_for_disk, dtype=int)

    _, disk_r_hat = radial_unit_vectors(positions[disk_rows])
    _, first_r_hat = radial_unit_vectors(positions[first_rows])
    _, second_r_hat = radial_unit_vectors(positions[second_rows])

    tilts_in = np.asarray(mesh.tilts_in_view(), dtype=float)
    tilts_out = np.asarray(mesh.tilts_out_view(), dtype=float)
    disk_t_in = np.einsum("ij,ij->i", tilts_in[disk_rows], disk_r_hat)
    first_t_out = np.einsum("ij,ij->i", tilts_out[first_rows], first_r_hat)
    second_t_out = np.einsum("ij,ij->i", tilts_out[second_rows], second_r_hat)
    trace_t_out = extrapolate_trace_to_radius(
        target_radius=float(shell_data.disk_radius),
        first_radius=float(shell_data.rim_radius),
        first_values=first_t_out,
        second_radius=float(shell_data.outer_radius),
        second_values=second_t_out,
    )

    rel = positions - center[None, :]
    heights = np.einsum("ij,j->i", rel, normal)
    disk_height = heights[disk_rows]
    first_height = heights[first_rows]
    second_height = heights[second_rows]
    phi_trace = _quadratic_boundary_slope(
        target_radius=float(shell_data.disk_radius),
        boundary_height=disk_height,
        first_radius=float(shell_data.rim_radius),
        first_height=first_height,
        second_radius=float(shell_data.outer_radius),
        second_height=second_height,
    )

    disk_theta = np.full(disk_t_in.shape, float(theta_meas), dtype=float)
    phi_trace_mean = float(np.mean(phi_trace))
    out.update(
        {
            "available": True,
            "disk_theta_at_R": float(np.mean(disk_theta)),
            "disk_t_in_at_R": float(np.mean(disk_t_in)),
            "outer_t_out_first_shell": float(np.mean(first_t_out)),
            "outer_geometry_trace_at_R_plus": phi_trace_mean,
            "outer_t_out_trace_at_R_plus": float(np.mean(trace_t_out)),
            "phi_trace_at_R_plus": phi_trace_mean,
            "disk_minus_outer_trace": float(np.mean(disk_t_in - trace_t_out)),
            "disk_minus_phi_trace": float(np.mean(disk_theta) - 2.0 * phi_trace_mean),
            "outer_geometry_vs_tilt_trace_gap": float(
                phi_trace_mean - np.mean(trace_t_out)
            ),
            "outer_first_shell_minus_outer_trace": float(
                np.mean(first_t_out - trace_t_out)
            ),
        }
    )
    return out


def _outer_profile_parity(
    mesh, positions: np.ndarray, theta_meas: float
) -> dict[str, float | bool]:
    """Return outer-profile parity diagnostics against the TeX solution."""
    mode = str(mesh.global_parameters.get("rim_slope_match_mode") or "").strip().lower()
    out: dict[str, float | bool] = {
        "available": False,
        "phi_profile_rel_rmse": 0.0,
        "z_profile_rel_rmse": 0.0,
        "sample_count": 0.0,
    }
    if mode != "physical_edge_staggered_v1":
        return out

    try:
        shell_data = build_local_interface_shell_data(
            mesh, positions=positions, group="disk"
        )
    except AssertionError:
        return out

    center = _resolve_center(mesh.global_parameters)
    normal = _resolve_normal(mesh.global_parameters)
    if normal is None:
        normal = _fit_plane_normal(positions[shell_data.disk_rows])
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    ring_r, ring_z = _ring_mean_heights(
        positions=positions,
        center=center,
        normal=normal,
        min_radius=float(shell_data.disk_radius),
    )
    if ring_r.size:
        order = np.argsort(ring_r)
        ring_r = ring_r[order]
        ring_z = ring_z[order]
        rounded = np.round(ring_r, 9)
        _, unique_idx = np.unique(rounded, return_index=True)
        unique_idx = np.sort(unique_idx)
        ring_r = ring_r[unique_idx]
        ring_z = ring_z[unique_idx]
    if ring_r.size < 2:
        return out

    rel = positions[np.asarray(shell_data.disk_rows, dtype=int)] - center[None, :]
    zR = float(np.mean(np.einsum("ij,j->i", rel, normal)))
    z_rel = ring_z - zR
    phi_num = np.gradient(z_rel, ring_r, edge_order=1)

    phi_star = 0.5 * float(theta_meas)
    r_theory = float(shell_data.disk_radius)
    phi_ref = phi_star * r_theory / ring_r
    z_ref = phi_star * r_theory * np.log(ring_r / r_theory)

    out.update(
        {
            "available": True,
            "phi_profile_rel_rmse": _relative_rmse(phi_num, phi_ref),
            "z_profile_rel_rmse": _relative_rmse(z_rel, z_ref),
            "sample_count": float(ring_r.size),
        }
    )
    return out


def _collect_report_from_context(
    *,
    ctx: CommandContext,
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> dict[str, Any]:
    minim = ctx.minimizer
    breakdown = minim.compute_energy_breakdown()
    tilt_stats = _tilt_stats_quantiles(ctx.mesh)
    gp = ctx.mesh.global_parameters

    kappa = float(
        (gp.get("bending_modulus_in") or 0.0) + (gp.get("bending_modulus_out") or 0.0)
    )
    kappa_t = float(
        (gp.get("tilt_modulus_in") or 0.0) + (gp.get("tilt_modulus_out") or 0.0)
    )
    drive = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)
    r_theory = float(gp.get("theory_radius") or DEFAULT_THEORY_RADIUS)

    theta_meas = float(gp.get("tilt_thetaB_value") or 0.0)
    contact_meas = float(breakdown.get("tilt_thetaB_contact_in") or 0.0)
    elastic_meas = float(
        (breakdown.get("tilt_in") or 0.0)
        + (breakdown.get("tilt_out") or 0.0)
        + (breakdown.get("bending_tilt_in") or 0.0)
        + (breakdown.get("bending_tilt_out") or 0.0)
    )
    total_meas = float(minim.compute_energy())

    def _benchmark_terms(
        *, kappa_value: float, kappa_t_value: float, radius_value: float
    ) -> dict[str, float]:
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
            "thetaB_star": float(theta_star),
            "elastic_star": float(elastic_star),
            "contact_star": float(contact_star),
            "total_star": float(total_star),
            "ratios": {
                "theta_ratio": _ratio(theta_meas, theta_star),
                "elastic_ratio": _ratio(elastic_meas, elastic_star),
                "contact_ratio": _ratio(contact_meas, contact_star),
                "total_ratio": _ratio(total_meas, total_star),
            },
        }

    def _ratio(meas: float, theory: float) -> float:
        if abs(theory) < 1e-16:
            return 0.0
        return float(meas / theory)

    legacy_anchor = _benchmark_terms(
        kappa_value=float(kappa),
        kappa_t_value=float(kappa_t),
        radius_value=float(r_theory),
    )
    tex_benchmark = _benchmark_terms(
        kappa_value=float(DEFAULT_TEX_BENDING_MODULUS),
        kappa_t_value=float(DEFAULT_TEX_TILT_MODULUS),
        radius_value=float(r_theory),
    )
    split_diag = matching_residual_diagnostics(
        ctx.mesh, ctx.mesh.global_parameters, ctx.mesh.positions_view()
    )
    phi_mean = float(split_diag.get("phi_mean") or 0.0)
    t_out_mean = float(
        phi_mean + float(split_diag.get("outer_residual", {}).get("mean") or 0.0)
    )
    theta_disk_mean = float(split_diag.get("theta_disk_mean") or 0.0)
    inner_residual_mean = float(split_diag.get("inner_residual", {}).get("mean") or 0.0)
    inner_available = bool(split_diag.get("inner_residual_available", False))
    if inner_available:
        t_in_mean = float(theta_disk_mean - phi_mean + inner_residual_mean)
    else:
        t_in_mean = 0.0
    phi_over_half_theta = 0.0
    if abs(theta_meas) > 1.0e-16:
        phi_over_half_theta = float(phi_mean / (0.5 * theta_meas))
    interface_traces = _interface_trace_diagnostics(
        ctx.mesh, ctx.mesh.positions_view(), theta_meas
    )
    outer_profile = _outer_profile_parity(
        ctx.mesh, ctx.mesh.positions_view(), theta_meas
    )

    report = {
        "meta": {
            "fixture": _report_fixture_path(mesh_path),
            "lane": _parity_lane_name(mesh_path, ctx.mesh),
            "protocol": list(protocol),
            "format": "yaml",
        },
        "metrics": {
            "final_energy": total_meas,
            "thetaB_value": theta_meas,
            "breakdown": {
                "bending_tilt_in": float(breakdown.get("bending_tilt_in") or 0.0),
                "bending_tilt_out": float(breakdown.get("bending_tilt_out") or 0.0),
                "tilt_in": float(breakdown.get("tilt_in") or 0.0),
                "tilt_out": float(breakdown.get("tilt_out") or 0.0),
                "tilt_thetaB_contact_in": float(
                    breakdown.get("tilt_thetaB_contact_in") or 0.0
                ),
            },
            "tilt_stats": tilt_stats,
            "reduced_terms": {
                "elastic_measured": elastic_meas,
                "contact_measured": contact_meas,
                "total_measured": total_meas,
            },
            "diagnostics": {
                "outer_split": {
                    "available": bool(split_diag.get("available", False)),
                    "phi_mean": phi_mean,
                    "t_in_mean": t_in_mean,
                    "t_out_mean": t_out_mean,
                    "theta_disk_mean": theta_disk_mean,
                    "phi_over_half_theta": phi_over_half_theta,
                },
                "outer_shell_geometry": dict(
                    getattr(ctx.mesh, "_parity_outer_shell_geometry", None)
                    or matching_ring_diagnostics(
                        ctx.mesh, ctx.mesh.global_parameters, ctx.mesh.positions_view()
                    )
                ),
                "interface_traces_at_R": interface_traces,
                "outer_profile_parity": outer_profile,
            },
            "theory": legacy_anchor,
            "legacy_anchor": legacy_anchor,
            "tex_benchmark": tex_benchmark,
        },
    }
    return report


def _collect_report(
    mesh_path: Path, protocol: tuple[str, ...], fixed_polish_steps: int = 0
) -> dict[str, Any]:
    ctx = _build_context(mesh_path)
    _stabilize_rim_radius_for_parity(ctx.mesh)
    ctx.mesh._parity_outer_shell_geometry = _activate_local_outer_shell_for_parity(
        ctx.mesh
    )
    for cmd in protocol:
        execute_command_line(ctx, cmd)
        _stabilize_rim_radius_for_parity(ctx.mesh)
        ctx.mesh._parity_outer_shell_geometry = _activate_local_outer_shell_for_parity(
            ctx.mesh
        )
    for _ in range(int(fixed_polish_steps)):
        execute_command_line(ctx, "g1")
        _stabilize_rim_radius_for_parity(ctx.mesh)
        ctx.mesh._parity_outer_shell_geometry = _activate_local_outer_shell_for_parity(
            ctx.mesh
        )
    report = _collect_report_from_context(
        ctx=ctx, mesh_path=mesh_path, protocol=protocol
    )
    report["meta"]["fixed_polish_steps"] = int(fixed_polish_steps)
    return report


def _load_yaml(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return copy.deepcopy(default) if default is not None else {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _max_ratio_drift(
    cur_ratios: dict[str, float], prev_ratios: dict[str, float]
) -> float:
    keys = sorted(set(cur_ratios.keys()) & set(prev_ratios.keys()))
    if not keys:
        return 0.0
    return float(max(abs(float(cur_ratios[k]) - float(prev_ratios[k])) for k in keys))


def _is_finite_metrics(report: dict[str, Any]) -> bool:
    vals = [
        float(report["metrics"]["final_energy"]),
        float(report["metrics"]["reduced_terms"]["elastic_measured"]),
        float(report["metrics"]["reduced_terms"]["contact_measured"]),
        float(report["metrics"]["reduced_terms"]["total_measured"]),
    ]
    vals.extend(float(v) for v in report["metrics"]["theory"]["ratios"].values())
    return all(np.isfinite(v) for v in vals)


def _default_state() -> dict[str, Any]:
    return {
        "current_stage": 0,
        "stage3_pass_streak": 0,
        "stage4_fail_streak": 0,
        "stage4_locked": False,
        "last_stage3_ratios": {},
        "last_stage3_energy": None,
        "stage3_anchor_ratios": {},
        "stage3_anchor_energy": None,
        "history": [],
    }


def update_expansion_state(
    *,
    state: dict[str, Any],
    report: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Update expansion state from one run and return (new_state, decisions)."""
    new_state = copy.deepcopy(state) if state else _default_state()
    thresholds = policy["thresholds"]
    stage = int(new_state.get("current_stage", 0))
    ratios = {
        str(k): float(v) for k, v in report["metrics"]["theory"]["ratios"].items()
    }
    energy = float(report["metrics"]["final_energy"])
    decisions: dict[str, Any] = {
        "stage_before": stage,
        "promoted_to_stage4": False,
        "rolled_back_to_stage3": False,
        "stage4_validation_passed": None,
    }

    finite_ok = _is_finite_metrics(report)
    if stage == 3:
        prev_ratios = new_state.get("last_stage3_ratios") or {}
        prev_energy = new_state.get("last_stage3_energy")
        if prev_ratios and prev_energy is not None:
            ratio_drift = _max_ratio_drift(ratios, prev_ratios)
            energy_drift = abs(energy - float(prev_energy))
            pass_ok = (
                finite_ok
                and ratio_drift <= float(thresholds["ratio_drift_max"])
                and energy_drift <= float(thresholds["energy_drift_max"])
            )
        else:
            pass_ok = finite_ok
        if pass_ok:
            new_state["stage3_pass_streak"] = (
                int(new_state.get("stage3_pass_streak", 0)) + 1
            )
        else:
            new_state["stage3_pass_streak"] = 0
        new_state["last_stage3_ratios"] = ratios
        new_state["last_stage3_energy"] = energy
        if not bool(new_state.get("stage4_locked", False)) and int(
            new_state["stage3_pass_streak"]
        ) >= int(thresholds["consecutive_passes_to_promote"]):
            new_state["current_stage"] = 4
            new_state["stage4_fail_streak"] = 0
            new_state["stage3_anchor_ratios"] = ratios
            new_state["stage3_anchor_energy"] = energy
            decisions["promoted_to_stage4"] = True

    elif stage == 4:
        stage4_cfg = thresholds["stage4"]
        anchor_ratios = new_state.get("stage3_anchor_ratios") or {}
        ratio_delta = _max_ratio_drift(ratios, anchor_ratios) if anchor_ratios else 0.0
        pass_ok = (
            finite_ok
            and ratio_delta <= float(stage4_cfg["ratio_delta_vs_stage3_max"])
            and np.isfinite(energy)
        )
        decisions["stage4_validation_passed"] = bool(pass_ok)
        if pass_ok:
            new_state["stage4_fail_streak"] = 0
        else:
            new_state["stage4_fail_streak"] = (
                int(new_state.get("stage4_fail_streak", 0)) + 1
            )
            if int(new_state["stage4_fail_streak"]) >= int(
                stage4_cfg["max_consecutive_failures_before_rollback"]
            ):
                new_state["current_stage"] = 3
                new_state["stage4_locked"] = True
                new_state["stage3_pass_streak"] = 0
                new_state["stage4_fail_streak"] = 0
                decisions["rolled_back_to_stage3"] = True

    decisions["stage_after"] = int(new_state.get("current_stage", stage))
    new_state.setdefault("history", []).append(
        {
            "stage_before": int(stage),
            "stage_after": int(new_state.get("current_stage", stage)),
            "energy": float(energy),
            "ratios": ratios,
            "decisions": decisions,
        }
    )
    return new_state, decisions


def _stage_suffix(policy: dict[str, Any], stage: int) -> list[str]:
    stages = policy["stages"]
    key = f"stage_{stage}"
    return [str(x) for x in stages.get(key, [])]


def _attach_stage_metadata(
    report: dict[str, Any], *, stage: int, suffix: list[str], ctx: CommandContext
) -> None:
    report["meta"]["expansion"] = {
        "stage": int(stage),
        "stage_suffix": list(suffix),
    }
    if stage == 4:
        report["meta"]["expansion"]["post_refine_mesh"] = {
            "vertex_count": int(len(ctx.mesh.vertices)),
            "triangle_count": int(len(ctx.mesh.facets)),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--protocol-mode",
        choices=("fixed", "expanded"),
        default="fixed",
        help="Use fixed protocol or convergence-gated expansion ladder.",
    )
    parser.add_argument(
        "--protocol",
        nargs="+",
        default=list(DEFAULT_PROTOCOL),
        help="Command sequence to reproduce parity metrics.",
    )
    parser.add_argument(
        "--expansion-policy",
        type=Path,
        default=DEFAULT_EXPANSION_POLICY,
        help="YAML policy for expanded protocol stages and gates.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_EXPANSION_STATE,
        help="Persistent YAML state file for expansion mode.",
    )
    parser.add_argument(
        "--fixed-polish-steps",
        type=int,
        default=0,
        help="Additional trailing g1 steps for fixed mode only (default: 0).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mesh_path = Path(args.mesh)
    out_path = Path(args.out)
    protocol = tuple(str(x) for x in args.protocol)
    fixed_polish_steps = int(args.fixed_polish_steps)
    if fixed_polish_steps < 0:
        raise ValueError("--fixed-polish-steps must be >= 0")

    if str(args.protocol_mode) == "fixed":
        report = _collect_report(
            mesh_path=mesh_path,
            protocol=protocol,
            fixed_polish_steps=fixed_polish_steps,
        )
    else:
        policy = _load_yaml(Path(args.expansion_policy))
        state_path = Path(args.state_file)
        state = _load_yaml(state_path, default=_default_state())
        stage = int(state.get("current_stage", 0))
        suffix = _stage_suffix(policy, stage)
        full_protocol = protocol + tuple(suffix)
        ctx = _build_context(mesh_path)
        for cmd in full_protocol:
            execute_command_line(ctx, cmd)
        report = _collect_report_from_context(
            ctx=ctx,
            mesh_path=mesh_path,
            protocol=full_protocol,
        )
        _attach_stage_metadata(report, stage=stage, suffix=suffix, ctx=ctx)
        new_state, decisions = update_expansion_state(
            state=state,
            report=report,
            policy=policy,
        )
        report["meta"]["expansion"]["decisions"] = decisions
        _save_yaml(state_path, new_state)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
