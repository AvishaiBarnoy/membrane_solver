#!/usr/bin/env python3
"""Identify the source of curved 1-disk outer-profile mismatches.

The audit is diagnostic-only.  It traces when the outer leaflet pair becomes
anti-symmetric and probes symmetric, anti-symmetric, and shape-only
perturbations through the current runtime energy path.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np
from scipy.special import k1

from modules.energy import bending_tilt_in, bending_tilt_out
from tools.diagnostics.curved_1disk_rim_inner_tilt_profile_audit import (
    _field_trace,
    _profile_source_probe,
)
from tools.diagnostics.curved_1disk_shape_propagation_blocker import (
    _build_minimizer,
)
from tools.diagnostics.curved_1disk_shared_rim_phi_target_audit import THEORY_THETA_B
from tools.diagnostics.curved_1disk_theory_benchmark import (
    OUTER_K1_WINDOW,
    OUTER_LOG_WINDOW,
    SHAPE_STEPS,
    _relative_rmse,
    _run_canonical_schedule,
    _shell_profile,
    _window_rows,
)
from tools.diagnostics.curved_1disk_trumpet_descent_audit import (
    _module_deltas,
)
from tools.diagnostics.curved_disk_theory import (
    compute_curved_disk_theory,
    tex_reference_params,
)
from tools.diagnostics.free_disk_profile_fits import _fit_k1, _fit_log
from tools.diagnostics.utils import (
    abs_by_region,
    capture_state,
    energy_total,
    radial_projection,
    radius_labels,
    restore_state,
    row_region,
)

ALLOWED_CLASSIFICATIONS = {
    "leaflet_relaxation_drives_antisymmetric_state",
    "bending_tilt_sign_or_ownership_drives_cancellation",
    "support_transition_mask_seeds_leaflet_cancellation",
    "shape_tilt_coupling_missing_after_valid_shape_update",
    "outer_tilt_k1_ok_but_log_shape_suppressed",
    "far_boundary_or_window_artifact",
    "inconclusive",
}
SIGN_CONVENTION_CLASSIFICATIONS = {
    "diagnostic_leaflet_sign_convention_mismatch",
    "runtime_relaxation_drives_antisymmetric_state",
    "inconclusive",
}


def _window_masks(mesh) -> dict[str, np.ndarray]:
    params = tex_reference_params()
    radius = float(params.radius)
    labels = radius_labels(mesh)
    max_radius = float(np.max(labels))
    free_outer = np.array(
        [row_region(mesh, row) == "outer_free" for row in range(len(labels))],
        dtype=bool,
    )
    free_radii = sorted(
        float(v)
        for v in set(labels[free_outer])
        if float(v) > radius + 1.0e-6 and float(v) < max_radius - 1.0e-6
    )
    first_free = set(free_radii[:2])
    far_cut = (
        free_radii[max(0, int(0.75 * (len(free_radii) - 1)))]
        if free_radii
        else max_radius
    )
    return {
        "outer_support": np.array(
            [row_region(mesh, row) == "outer_support" for row in range(len(labels))],
            dtype=bool,
        ),
        "first_free": free_outer & np.isin(labels, list(first_free)),
        "k1_window": free_outer
        & (labels >= OUTER_K1_WINDOW[0] * radius)
        & (labels <= OUTER_K1_WINDOW[1] * radius),
        "log_window": free_outer
        & (labels >= OUTER_LOG_WINDOW[0] * radius)
        & (labels <= OUTER_LOG_WINDOW[1] * radius),
        "far_boundary": free_outer & (labels >= far_cut),
    }


def _shell_trace(mesh, *, label: str) -> dict[str, object]:
    base = _field_trace(mesh, label=label)
    positions = mesh.positions_view()
    shell_rows = _shell_profile(mesh)
    by_radius = {round(float(row["radius"]), 8): row for row in shell_rows}
    masks = _window_masks(mesh)
    for row in base["shells"]:
        radius_key = round(float(row["radius"]), 8)
        profile_row = by_radius.get(radius_key, {})
        theta_in = float(row["theta_in_median"])
        theta_out = float(row["theta_out_median"])
        row["z_median"] = float(profile_row.get("z", 0.0))
        row["curvature_median"] = float(profile_row.get("J", 0.0))
        row["leaflet_gap_median"] = float(abs(theta_in - theta_out))
        row["symmetric_sum_abs"] = float(abs(theta_in + theta_out))
        row["antisymmetric_gap_abs"] = float(abs(theta_in - theta_out))
        shell_mask = np.isclose(radius_labels(mesh), radius_key, atol=5.0e-9)
        row["windows"] = sorted(
            name for name, mask in masks.items() if np.any(mask & shell_mask)
        )
        row["z_span"] = float(
            np.max(positions[shell_mask, 2]) - np.min(positions[shell_mask, 2])
        )
    return base


def _module_tilt_gradient_probe(minim) -> dict[str, object]:
    mesh = minim.mesh
    positions = mesh.positions_view()
    common = {
        "mesh": mesh,
        "global_params": mesh.global_parameters,
        "param_resolver": minim.param_resolver,
        "positions": positions,
        "index_map": mesh.vertex_index_to_row,
        "ctx": None,
        "tilts_in": mesh.tilts_in_view(),
        "tilts_out": mesh.tilts_out_view(),
    }
    out: dict[str, object] = {}
    for name, module, tilt_key in (
        ("bending_tilt_in", bending_tilt_in, "tilt_in"),
        ("bending_tilt_out", bending_tilt_out, "tilt_out"),
    ):
        grad = np.zeros_like(positions)
        tin_grad = np.zeros_like(mesh.tilts_in_view())
        tout_grad = np.zeros_like(mesh.tilts_out_view())
        energy = module.compute_energy_and_gradient_array(
            grad_arr=grad,
            tilt_in_grad_arr=tin_grad,
            tilt_out_grad_arr=tout_grad,
            **common,
        )
        tilt_grad = tin_grad if tilt_key == "tilt_in" else tout_grad
        _, theta_in, theta_out, _ = _radial_components(mesh)
        theta = theta_in if tilt_key == "tilt_in" else theta_out
        radial_grad = radial_projection(mesh, tilt_grad)
        out[name] = {
            "energy": float(energy),
            "tilt_grad_norm": float(np.linalg.norm(tilt_grad)),
            "tilt_grad_abs_by_region": abs_by_region(
                mesh, np.linalg.norm(tilt_grad, axis=1)
            ),
            "radial_grad_dot_theta_by_window": _window_dot_summary(
                mesh, radial_grad, theta
            ),
        }
    return out


def _radial_components(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = radii > 1.0e-12
    r_hat[good, :2] = positions[good, :2] / radii[good, None]
    theta_in = np.einsum("ij,ij->i", mesh.tilts_in_view(), r_hat)
    theta_out = np.einsum("ij,ij->i", mesh.tilts_out_view(), r_hat)
    return radii, theta_in, theta_out, 0.5 * (theta_in + theta_out)


def theta_outer_common_physical(
    theta_in: np.ndarray, theta_out: np.ndarray
) -> np.ndarray:
    """Return the physical outer common mode from stored leaflet components.

    The diagnostic radial components for ``theta_in`` and ``theta_out`` are
    stored in opposite local leaflet frames on the outer membrane, so the
    physical same-sign leaflet mode is their difference, not their raw sum.
    """
    return 0.5 * (
        np.asarray(theta_in, dtype=float) - np.asarray(theta_out, dtype=float)
    )


def _window_dot_summary(mesh, lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    masks = _window_masks(mesh)
    out = {}
    for name, mask in masks.items():
        if np.any(mask):
            out[name] = float(np.dot(lhs[mask], rhs[mask]))
        else:
            out[name] = 0.0
    return out


def _apply_tilt_mode(mesh, mode: str, epsilon: float) -> None:
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = radii > 1.0e-12
    r_hat[good, :2] = positions[good, :2] / radii[good, None]
    mask = _window_masks(mesh)["k1_window"]
    delta = np.zeros_like(positions)
    delta[mask] = float(epsilon) * r_hat[mask]
    if mode == "symmetric_leaflet":
        mesh.tilts_in_view()[:] += delta
        mesh.tilts_out_view()[:] += delta
    elif mode == "antisymmetric_leaflet":
        mesh.tilts_in_view()[:] += delta
        mesh.tilts_out_view()[:] -= delta
    else:
        raise ValueError(f"Unknown tilt perturbation mode: {mode}")
    mesh.touch_tilts_in()
    mesh.touch_tilts_out()


def _apply_shape_log_mode(mesh, epsilon: float) -> None:
    positions = mesh.positions_view().copy(order="F")
    radii = np.linalg.norm(positions[:, :2], axis=1)
    mask = _window_masks(mesh)["log_window"]
    if np.any(mask):
        r_min = float(np.min(radii[mask]))
        values = np.log(np.maximum(radii, r_min) / max(r_min, 1.0e-12))
        scale = float(np.max(np.abs(values[mask])))
        if scale > 0.0:
            positions[mask, 2] += float(epsilon) * values[mask] / scale
    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = positions[row]
    mesh.increment_version()
    mesh.build_position_cache()


def _perturbation_probes(minim, *, epsilon: float = 1.0e-6) -> list[dict[str, object]]:
    mesh = minim.mesh
    baseline = minim.compute_energy_breakdown()
    state = capture_state(mesh)
    rows = []
    for name in ("symmetric_leaflet", "antisymmetric_leaflet", "shape_log"):
        if name == "shape_log":
            _apply_shape_log_mode(mesh, epsilon)
        else:
            _apply_tilt_mode(mesh, name, epsilon)
        perturbed = minim.compute_energy_breakdown()
        deltas = _module_deltas(baseline, perturbed)
        total_delta = energy_total(perturbed) - energy_total(baseline)
        module_sum = energy_total(deltas)
        rows.append(
            {
                "name": name,
                "epsilon": float(epsilon),
                "total_delta": float(total_delta),
                "module_delta_sum": float(module_sum),
                "module_residual": float(total_delta - module_sum),
                "module_deltas": deltas,
                "top_module_deltas": [
                    {"module": str(module), "delta": float(delta)}
                    for module, delta in sorted(
                        deltas.items(), key=lambda item: abs(item[1]), reverse=True
                    )[:6]
                ],
            }
        )
        restore_state(mesh, *state)
    return rows


def _fit_k1_channel(
    shell_rows: list[dict[str, float]],
    *,
    channel: str,
    radius: float,
    lambda_theory: float,
    last_free_shell_radius: float,
) -> dict[str, object]:
    rows = _window_rows(
        shell_rows,
        radius=radius,
        window=OUTER_K1_WINDOW,
        last_free_shell_radius=last_free_shell_radius,
    )
    r = np.asarray([float(row["radius"]) for row in rows], dtype=float)
    if channel == "theta_in":
        y = np.asarray([float(row["theta_in"]) for row in rows], dtype=float)
    elif channel == "theta_out":
        y = np.asarray([float(row["theta_out"]) for row in rows], dtype=float)
    elif channel == "shared_abs":
        y = np.asarray([abs(float(row["theta_shared"])) for row in rows], dtype=float)
    elif channel == "theta_outer_common_physical":
        theta_in = np.asarray([float(row["theta_in"]) for row in rows], dtype=float)
        theta_out = np.asarray([float(row["theta_out"]) for row in rows], dtype=float)
        y = theta_outer_common_physical(theta_in, theta_out)
    else:
        y = np.asarray([float(row["theta_shared"]) for row in rows], dtype=float)
    amp, lam_fit, _ = _fit_k1(r, y, float(radius))
    yhat = amp * k1(lam_fit * r) / k1(lam_fit * float(radius))
    return {
        "channel": channel,
        "count": int(len(r)),
        "amplitude_fit": float(amp),
        "lambda_fit": float(lam_fit),
        "lambda_ratio": float(lam_fit / lambda_theory),
        "rel_rmse": _relative_rmse(y, yhat),
    }


def _fit_k1_values(
    rows: list[dict[str, float]],
    *,
    name: str,
    values: np.ndarray,
    radius: float,
    lambda_theory: float,
) -> dict[str, object]:
    r = np.asarray([float(row["radius"]) for row in rows], dtype=float)
    amp, lam_fit, _ = _fit_k1(r, values, float(radius))
    yhat = amp * k1(lam_fit * r) / k1(lam_fit * float(radius))
    return {
        "name": name,
        "count": int(len(r)),
        "rim_amplitude": float(amp),
        "amplitude_sign": "positive" if float(amp) >= 0.0 else "negative",
        "lambda_fit": float(lam_fit),
        "lambda_ratio": float(lam_fit / lambda_theory),
        "rel_rmse": _relative_rmse(values, yhat),
    }


def _leaflet_sign_convention_probe(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    lambda_theory: float,
    last_free_shell_radius: float,
    log_slope_ratio: float,
) -> dict[str, object]:
    rows = _window_rows(
        shell_rows,
        radius=radius,
        window=OUTER_K1_WINDOW,
        last_free_shell_radius=last_free_shell_radius,
    )
    theta_in = np.asarray([float(row["theta_in"]) for row in rows], dtype=float)
    theta_out = np.asarray([float(row["theta_out"]) for row in rows], dtype=float)
    common_raw = 0.5 * (theta_in + theta_out)
    antisym_raw = 0.5 * (theta_in - theta_out)
    common_flip = 0.5 * (theta_in - theta_out)
    antisym_flip = 0.5 * (theta_in + theta_out)
    fits = [
        _fit_k1_values(
            rows,
            name="theta_common_raw",
            values=common_raw,
            radius=radius,
            lambda_theory=lambda_theory,
        ),
        _fit_k1_values(
            rows,
            name="theta_antisym_raw",
            values=antisym_raw,
            radius=radius,
            lambda_theory=lambda_theory,
        ),
        _fit_k1_values(
            rows,
            name="theta_common_flip",
            values=common_flip,
            radius=radius,
            lambda_theory=lambda_theory,
        ),
        _fit_k1_values(
            rows,
            name="theta_antisym_flip",
            values=antisym_flip,
            radius=radius,
            lambda_theory=lambda_theory,
        ),
    ]
    by_name = {str(row["name"]): row for row in fits}
    raw_common_good = _is_good_k1_fit(by_name["theta_common_raw"])
    flip_common_good = _is_good_k1_fit(by_name["theta_common_flip"])
    raw_antisym_good = _is_good_k1_fit(by_name["theta_antisym_raw"])
    if raw_common_good:
        location = "raw_common_mode"
        classification = "inconclusive"
        recommendation = (
            "Raw common mode already fits K1; inspect remaining height/curvature "
            "coupling before changing leaflet signs."
        )
    elif flip_common_good:
        location = "flipped_common_mode"
        classification = "diagnostic_leaflet_sign_convention_mismatch"
        recommendation = (
            "Flipped common mode fits K1 well; inspect and fix diagnostic leaflet "
            "sign convention before changing runtime physics."
        )
    elif raw_antisym_good:
        location = "raw_antisymmetric_physical_mode"
        classification = "runtime_relaxation_drives_antisymmetric_state"
        recommendation = (
            "Only the antisymmetric physical mode fits K1; inspect runtime leaflet "
            "coupling, bending-tilt energy signs, or tilt relaxation signs."
        )
    else:
        location = "no_good_k1_mode"
        classification = "inconclusive"
        recommendation = (
            "No common or antisymmetric candidate cleanly fits K1; continue with "
            "shape/curvature and window diagnostics before changing runtime physics."
        )
    return {
        "fits": fits,
        "log_height_slope_ratio": float(log_slope_ratio),
        "good_k1_profile_location": location,
        "classification": classification,
        "allowed_classifications": sorted(SIGN_CONVENTION_CLASSIFICATIONS),
        "recommendation": recommendation,
    }


def _is_good_k1_fit(row: dict[str, object]) -> bool:
    return (
        int(row["count"]) > 0
        and abs(float(row["lambda_ratio"]) - 1.0) <= 0.40
        and float(row["rel_rmse"]) <= 0.10
        and abs(float(row["rim_amplitude"])) > 1.0e-8
    )


def _is_good_k1_channel(row: dict[str, object]) -> bool:
    return (
        int(row["count"]) > 0
        and abs(float(row["lambda_ratio"]) - 1.0) <= 0.40
        and float(row["rel_rmse"]) <= 0.10
        and abs(float(row["amplitude_fit"])) > 1.0e-8
    )


def _profile_fit_controls(mesh, *, theta_b: float) -> dict[str, object]:
    params = tex_reference_params()
    theory = compute_curved_disk_theory(params)
    shell_rows = _shell_profile(mesh)
    radius = float(params.radius)
    radii = [float(row["radius"]) for row in shell_rows]
    max_radius = float(max(radii))
    free_outer_radii = sorted(
        rr for rr in radii if rr > radius + 1.0e-6 and rr < max_radius - 1.0e-6
    )
    last_free = float(free_outer_radii[-1]) if free_outer_radii else max_radius
    log_rows = _window_rows(
        shell_rows,
        radius=radius,
        window=OUTER_LOG_WINDOW,
        last_free_shell_radius=last_free,
    )
    clean_log_rows = [row for row in log_rows if abs(float(row["J"])) <= 0.05]
    log_clean = _fit_log_channel(
        clean_log_rows,
        radius=radius,
        slope_theory=0.5 * float(theta_b) * radius,
    )
    log_all = _fit_log_channel(
        log_rows,
        radius=radius,
        slope_theory=0.5 * float(theta_b) * radius,
    )
    physical_common_k1 = _fit_k1_channel(
        shell_rows,
        channel="theta_outer_common_physical",
        radius=radius,
        lambda_theory=float(theory.lambda_value),
        last_free_shell_radius=last_free,
    )
    phi_star = 0.5 * float(theta_b)
    expected_log_slope = phi_star * radius
    return {
        "k1_by_channel": [
            _fit_k1_channel(
                shell_rows,
                channel=channel,
                radius=radius,
                lambda_theory=float(theory.lambda_value),
                last_free_shell_radius=last_free,
            )
            for channel in (
                "theta_in",
                "theta_out",
                "shared_signed",
                "shared_abs",
                "theta_outer_common_physical",
            )
        ],
        "primary_physical_common_k1": physical_common_k1,
        "theory_comparison": {
            "physical_common_lambda_fit": float(physical_common_k1["lambda_fit"]),
            "physical_common_rel_rmse": float(physical_common_k1["rel_rmse"]),
            "physical_common_rim_amplitude": float(physical_common_k1["amplitude_fit"]),
            "expected_lambda": float(theory.lambda_value),
            "theta_B": float(theta_b),
            "theta_B_half": float(0.5 * float(theta_b)),
            "rim_physical_theta_amplitude": float(physical_common_k1["amplitude_fit"]),
            "rim_physical_theta_amplitude_over_half_theta_B": float(
                physical_common_k1["amplitude_fit"]
            )
            / max(abs(0.5 * float(theta_b)), 1.0e-12),
            "measured_log_height_slope": float(log_all["slope_fit"]),
            "expected_log_height_phi_star": float(phi_star),
            "expected_log_height_slope": float(expected_log_slope),
            "log_height_slope_ratio": float(log_all["slope_ratio"]),
        },
        "log_all": log_all,
        "log_curvature_filtered": log_clean,
        "curvature_filtered_shell_count": int(len(clean_log_rows)),
        "leaflet_sign_convention_probe": _leaflet_sign_convention_probe(
            shell_rows,
            radius=radius,
            lambda_theory=float(theory.lambda_value),
            last_free_shell_radius=last_free,
            log_slope_ratio=float(log_all["slope_ratio"]),
        ),
    }


def _fit_log_channel(
    rows: list[dict[str, float]],
    *,
    radius: float,
    slope_theory: float,
) -> dict[str, float | int]:
    if len(rows) < 2:
        return {
            "count": int(len(rows)),
            "z0_fit": 0.0,
            "slope_fit": 0.0,
            "slope_ratio": 0.0,
            "rel_rmse": 0.0,
        }
    r = np.asarray([float(row["radius"]) for row in rows], dtype=float)
    z = np.asarray([float(row["z"]) for row in rows], dtype=float)
    z0, slope_fit, _ = _fit_log(r, z, float(radius))
    zhat = z0 + slope_fit * np.log(r / float(radius))
    return {
        "count": int(len(rows)),
        "z0_fit": float(z0),
        "slope_fit": float(slope_fit),
        "slope_ratio": float(slope_fit / max(float(slope_theory), 1.0e-12)),
        "rel_rmse": _relative_rmse(z, zhat),
    }


def _first_collapse_stage(traces: list[dict[str, object]]) -> dict[str, object]:
    for trace in traces:
        candidates = []
        for row in trace["shells"]:
            windows = set(row.get("windows", []))
            if not ({"k1_window", "first_free", "outer_support"} & windows):
                continue
            anti = float(row["antisymmetric_gap_abs"])
            sym = float(row["symmetric_sum_abs"])
            if anti > 1.0e-7 and sym / max(anti, 1.0e-12) < 0.25:
                candidates.append(
                    {
                        "stage": trace["label"],
                        "radius": float(row["radius"]),
                        "windows": sorted(windows),
                        "theta_in": float(row["theta_in_median"]),
                        "theta_out": float(row["theta_out_median"]),
                        "symmetric_sum_abs": sym,
                        "antisymmetric_gap_abs": anti,
                    }
                )
        if candidates:
            return candidates[0]
    return {"stage": "none", "radius": 0.0, "windows": []}


def _case_report(*, label: str, theta_b: float, selected: bool) -> dict[str, object]:
    if selected:
        schedule = _run_canonical_schedule()
        mesh = schedule["mesh"]
        return {
            "label": label,
            "theta_B": float(schedule["theta_B_selected"]),
            "selected_theta": True,
            "shell_traces": [_shell_trace(mesh, label="selected_theta_final")],
            "profile_probe": _profile_source_probe(
                mesh, theta_b=float(schedule["theta_B_selected"])
            ),
            "profile_fit_controls": _profile_fit_controls(
                mesh, theta_b=float(schedule["theta_B_selected"])
            ),
        }

    minim = _build_minimizer(float(theta_b), max_iter=10)
    mesh = minim.mesh
    traces = [_shell_trace(mesh, label="configured")]
    minim.enforce_constraints_after_mesh_ops(mesh)
    traces.append(_shell_trace(mesh, label="after_geometric_enforcement"))
    minim._relax_leaflet_tilts(
        positions=mesh.positions_view(),
        mode=mesh.global_parameters.get("tilt_solve_mode", "fixed"),
    )
    traces.append(_shell_trace(mesh, label="after_tilt_relaxation"))
    gradient_probe = _module_tilt_gradient_probe(minim)
    perturbations = _perturbation_probes(minim)
    minim.minimize(n_steps=SHAPE_STEPS)
    traces.append(_shell_trace(mesh, label="after_shape_minimize"))
    mesh.project_tilts_to_tangent()
    mesh.increment_version()
    traces.append(_shell_trace(mesh, label="after_tangent_projection"))
    return {
        "label": label,
        "theta_B": float(theta_b),
        "selected_theta": False,
        "shell_traces": traces,
        "first_collapse_stage": _first_collapse_stage(traces),
        "module_tilt_gradient_probe": gradient_probe,
        "perturbation_probes": perturbations,
        "profile_probe": _profile_source_probe(mesh, theta_b=float(theta_b)),
        "profile_fit_controls": _profile_fit_controls(mesh, theta_b=float(theta_b)),
    }


def _classify(report: dict[str, object]) -> str:
    fixed = report["cases"][0]
    first = fixed["first_collapse_stage"]
    stage = str(first.get("stage") or "")
    profile = fixed["profile_probe"]
    fit_controls = fixed["profile_fit_controls"]
    physical_common = fit_controls["primary_physical_common_k1"]
    gap_ratio = float(profile["window_leaflet_gap_ratio"])
    log_ratio = abs(float(fit_controls["log_all"]["slope_ratio"]))
    log_filtered_ratio = abs(
        float(fit_controls["log_curvature_filtered"]["slope_ratio"])
    )
    if _is_good_k1_channel(physical_common) and log_ratio < 0.25:
        return "outer_tilt_k1_ok_but_log_shape_suppressed"
    if stage == "after_tilt_relaxation" and gap_ratio > 10.0:
        return "leaflet_relaxation_drives_antisymmetric_state"
    if stage == "after_geometric_enforcement":
        return "support_transition_mask_seeds_leaflet_cancellation"
    if gap_ratio > 10.0:
        return "bending_tilt_sign_or_ownership_drives_cancellation"
    if log_ratio < 0.25 and log_filtered_ratio < 0.25:
        return "shape_tilt_coupling_missing_after_valid_shape_update"
    if log_ratio < 0.25 <= log_filtered_ratio:
        return "far_boundary_or_window_artifact"
    return "inconclusive"


def run_curved_1disk_outer_profile_source_audit(
    *,
    theta_b: float = THEORY_THETA_B,
    include_selected: bool = True,
) -> dict[str, object]:
    """Return the outer-profile source audit report."""
    cases = [
        _case_report(label="fixed_theory_theta", theta_b=float(theta_b), selected=False)
    ]
    if include_selected:
        cases.append(
            _case_report(label="selected_theta", theta_b=float(theta_b), selected=True)
        )
    report: dict[str, object] = {
        "theta_B_fixed": float(theta_b),
        "cases": cases,
        "diagnosis": {},
    }
    classification = _classify(report)
    report["diagnosis"] = {
        "classification": classification,
        "sign_convention_classification": report["cases"][0]["profile_fit_controls"][
            "leaflet_sign_convention_probe"
        ]["classification"],
        "allowed_classifications": sorted(ALLOWED_CLASSIFICATIONS),
        "allowed_sign_convention_classifications": sorted(
            SIGN_CONVENTION_CLASSIFICATIONS
        ),
        "summary": (
            "This diagnostic ranks the source of the outer profile mismatch. Any "
            "runtime or theory-facing change requires a narrow Feature Contract."
        ),
        "recommended_next_stream": (
            "Use the classified source to write a Feature Contract before changing "
            "bending-tilt signs, ownership, tilt relaxation, support masks, or "
            "shape/tilt coupling. Do not rescale energy, calibrate coefficients, "
            "add hidden weights, or tune to the theory curve."
        ),
        "no_energy_rescaling": True,
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=THEORY_THETA_B)
    parser.add_argument(
        "--skip-selected",
        action="store_true",
        help="Only run the fixed-theta case.",
    )
    args = parser.parse_args(argv)
    report = run_curved_1disk_outer_profile_source_audit(
        theta_b=float(args.theta),
        include_selected=not bool(args.skip_selected),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
