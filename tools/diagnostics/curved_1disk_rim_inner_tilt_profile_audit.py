#!/usr/bin/env python3
"""Audit rim inner-leaflet tilt and outer-profile failures on the curved 1-disk lane.

This diagnostic is read-only with respect to runtime physics.  It traces the
leaflet radial tilt fields through the existing fixed-theta and selected-theta
paths, then reports whether the remaining miss is most consistent with a rim
tilt constraint/projection issue, inner-leaflet ownership, or outer profile
measurement/coupling.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from modules.energy import bending_tilt_in, bending_tilt_out
from modules.energy.bt_selection import (
    _collect_group_rows,
    _collect_preset_rows,
)
from tools.diagnostics.curved_1disk_shape_propagation_blocker import (
    _build_minimizer,
)
from tools.diagnostics.curved_1disk_shared_rim_phi_target_audit import THEORY_THETA_B
from tools.diagnostics.curved_1disk_theory_benchmark import (
    OUTER_K1_WINDOW,
    OUTER_LOG_WINDOW,
    SHAPE_STEPS,
    _fit_outer_k1,
    _fit_outer_log_height,
    _outer_curvature_summary,
    _run_canonical_schedule,
    _shell_profile,
)
from tools.diagnostics.curved_1disk_trumpet_descent_audit import (
    _breakdown_total,
    _module_deltas,
)
from tools.diagnostics.curved_disk_theory import (
    compute_curved_disk_theory,
    tex_reference_params,
)

RIM_CLASSIFICATIONS = {
    "inner_leaflet_not_driven",
    "inner_leaflet_masked_at_rim",
    "tilt_projection_zeroes_inner_leaflet",
    "inner_leaflet_energy_prefers_zero",
    "measurement_uses_wrong_rows",
    "inconclusive",
}
PROFILE_CLASSIFICATIONS = {
    "leaflet_mismatch_dominates",
    "shape_slope_not_coupled_to_tilt",
    "transition_band_still_flattens_profile",
    "far_boundary_curvature_pollutes_fit",
    "measurement_window_artifact",
    "inconclusive",
}


def _row_region(mesh, row: int) -> str:
    index_map = mesh.vertex_index_to_row
    # Use cached selection helpers for consistent and efficient region identification.
    if row in _collect_preset_rows(
        mesh, presets=("disk",), cache_tag="diag_audit_disk", index_map=index_map
    ):
        return "disk"
    if row in _collect_group_rows(mesh, group="rim", index_map=index_map):
        return "shared_rim"
    if row in _collect_group_rows(mesh, group="outer", index_map=index_map):
        return "outer_support"
    return "outer_free"


def _radial_thetas(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = radii > 1.0e-12
    r_hat[good, :2] = positions[good, :2] / radii[good, None]
    theta_in = np.einsum("ij,ij->i", mesh.tilts_in_view(), r_hat)
    theta_out = np.einsum("ij,ij->i", mesh.tilts_out_view(), r_hat)
    return radii, theta_in, theta_out, 0.5 * (theta_in + theta_out)


def _field_trace(mesh, *, label: str) -> dict[str, object]:
    radii, theta_in, theta_out, theta_shared = _radial_thetas(mesh)
    rows = []
    for radius in sorted({round(float(r), 8) for r in radii if r > 1.0e-12}):
        mask = np.isclose(np.round(radii, 8), radius, atol=5.0e-9)
        regions = sorted({_row_region(mesh, int(row)) for row in np.flatnonzero(mask)})
        rows.append(
            {
                "radius": float(np.median(radii[mask])),
                "regions": regions,
                "row_count": int(np.count_nonzero(mask)),
                "theta_in_median": float(np.median(theta_in[mask])),
                "theta_out_median": float(np.median(theta_out[mask])),
                "theta_shared_median": float(np.median(theta_shared[mask])),
                "theta_in_abs_max": float(np.max(np.abs(theta_in[mask]))),
                "theta_out_abs_max": float(np.max(np.abs(theta_out[mask]))),
            }
        )
    rim_mask = np.array(
        [_row_region(mesh, row) == "shared_rim" for row in range(len(radii))]
    )
    support_mask = np.array(
        [_row_region(mesh, row) == "outer_support" for row in range(len(radii))]
    )
    return {
        "label": str(label),
        "shells": rows,
        "rim": _region_theta_summary(theta_in, theta_out, theta_shared, rim_mask),
        "outer_support": _region_theta_summary(
            theta_in, theta_out, theta_shared, support_mask
        ),
    }


def _region_theta_summary(
    theta_in: np.ndarray,
    theta_out: np.ndarray,
    theta_shared: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    if not np.any(mask):
        return {
            "row_count": 0,
            "theta_in_median": 0.0,
            "theta_out_median": 0.0,
            "theta_shared_median": 0.0,
            "leaflet_gap_median": 0.0,
        }
    return {
        "row_count": int(np.count_nonzero(mask)),
        "theta_in_median": float(np.median(theta_in[mask])),
        "theta_out_median": float(np.median(theta_out[mask])),
        "theta_shared_median": float(np.median(theta_shared[mask])),
        "leaflet_gap_median": float(
            np.median(np.abs(theta_in[mask] - theta_out[mask]))
        ),
    }


def _tilt_enforce_trace(theta_b: float) -> tuple[object, list[dict[str, object]]]:
    minim = _build_minimizer(float(theta_b), max_iter=10)
    mesh = minim.mesh
    traces = [_field_trace(mesh, label="configured")]
    minim.enforce_constraints_after_mesh_ops(mesh)
    traces.append(_field_trace(mesh, label="after_geometric_enforcement"))
    minim._relax_leaflet_tilts(
        positions=mesh.positions_view(),
        mode=mesh.global_parameters.get("tilt_solve_mode", "fixed"),
    )
    traces.append(_field_trace(mesh, label="after_tilt_relaxation"))
    before_in = mesh.tilts_in_view().copy(order="F")
    before_out = mesh.tilts_out_view().copy(order="F")
    if hasattr(minim.constraint_manager, "enforce_tilt_constraints"):
        minim.constraint_manager.enforce_tilt_constraints(
            mesh, global_params=mesh.global_parameters
        )
    traces.append(_field_trace(mesh, label="after_rim_tilt_enforcement"))
    traces[-1]["tilt_enforcement_delta"] = {
        "tilt_in_l2": float(np.linalg.norm(mesh.tilts_in_view() - before_in)),
        "tilt_out_l2": float(np.linalg.norm(mesh.tilts_out_view() - before_out)),
    }
    return minim, traces


def _module_gradient_probe(minim) -> dict[str, object]:
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
        out[name] = {
            "energy": float(energy),
            "shape_grad_norm": float(np.linalg.norm(grad)),
            "shape_grad_z_abs_by_region": _abs_by_region(mesh, grad[:, 2]),
            "tilt_grad_norm": float(np.linalg.norm(tilt_grad)),
            "tilt_grad_abs_by_region": _abs_by_region(
                mesh, np.linalg.norm(tilt_grad, axis=1)
            ),
        }
    return out


def _abs_by_region(mesh, values: np.ndarray) -> dict[str, float]:
    out = {"disk": 0.0, "shared_rim": 0.0, "outer_support": 0.0, "outer_free": 0.0}
    for row, value in enumerate(np.asarray(values, dtype=float)):
        out[_row_region(mesh, row)] += abs(float(value))
    return out


def _energy_reconciliation(minim, *, epsilon: float = 1.0e-6) -> dict[str, object]:
    mesh = minim.mesh
    positions0 = mesh.positions_view().copy(order="F")
    baseline = minim.compute_energy_breakdown()
    radii = np.linalg.norm(positions0[:, :2], axis=1)
    free = np.array(
        [_row_region(mesh, row) == "outer_free" for row in range(len(radii))]
    )
    mode = np.zeros(len(radii), dtype=float)
    if np.any(free):
        r_min = float(np.min(radii[free]))
        vals = np.log(np.maximum(radii, r_min) / max(r_min, 1.0e-12))
        scale = float(np.max(np.abs(vals[free])))
        if scale > 0.0:
            mode[free] = vals[free] / scale
    perturbed = positions0.copy(order="F")
    perturbed[:, 2] += float(epsilon) * mode
    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = perturbed[row]
    mesh.increment_version()
    mesh.build_position_cache()
    after = minim.compute_energy_breakdown()
    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = positions0[row]
    mesh.increment_version()
    mesh.build_position_cache()
    deltas = _module_deltas(baseline, after)
    total_delta = _breakdown_total(after) - _breakdown_total(baseline)
    module_sum = _breakdown_total(deltas)
    return {
        "epsilon": float(epsilon),
        "total_delta": float(total_delta),
        "module_delta_sum": float(module_sum),
        "module_residual": float(total_delta - module_sum),
        "module_deltas": deltas,
    }


def _profile_source_probe(mesh, *, theta_b: float) -> dict[str, object]:
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
    k1_rows = [
        row
        for row in shell_rows
        if OUTER_K1_WINDOW[0] * radius <= float(row["radius"]) <= last_free - 1.0e-6
        and float(row["radius"]) <= OUTER_K1_WINDOW[1] * radius
    ]
    profile = {
        "outer_k1_shared": _fit_outer_k1(
            shell_rows,
            radius=radius,
            lambda_theory=float(theory.lambda_value),
            last_free_shell_radius=last_free,
            window=OUTER_K1_WINDOW,
        ),
        "outer_height_log": _fit_outer_log_height(
            shell_rows,
            radius=radius,
            slope_theory=0.5 * float(theta_b) * radius,
            last_free_shell_radius=last_free,
            window=OUTER_LOG_WINDOW,
        ),
        "outer_curvature": _outer_curvature_summary(
            shell_rows, radius=radius, last_free_shell_radius=last_free
        ),
        "fit_window_shells": [
            {
                "radius": float(row["radius"]),
                "theta_in": float(row["theta_in"]),
                "theta_out": float(row["theta_out"]),
                "theta_shared": float(row["theta_shared"]),
                "z": float(row["z"]),
                "J": float(row["J"]),
                "leaflet_gap": float(abs(row["theta_in"] - row["theta_out"])),
            }
            for row in k1_rows
        ],
    }
    gaps = [float(row["leaflet_gap"]) for row in profile["fit_window_shells"]]
    shared = [abs(float(row["theta_shared"])) for row in profile["fit_window_shells"]]
    profile["window_leaflet_gap_ratio"] = (
        float(np.median(gaps) / max(np.median(shared), 1.0e-12)) if gaps else 0.0
    )
    return profile


def _classify_rim(report: dict[str, object]) -> str:
    fixed = report["cases"][0]
    traces = fixed["tilt_trace"]
    after_relax = next(row for row in traces if row["label"] == "after_tilt_relaxation")
    after_enforce = next(
        row for row in traces if row["label"] == "after_rim_tilt_enforcement"
    )
    rim_relax = after_relax["rim"]
    rim_enforce = after_enforce["rim"]
    support_enforce = after_enforce["outer_support"]
    in_grad = fixed["module_gradient_probe"]["bending_tilt_in"]
    rim_tilt_grad = float(in_grad["tilt_grad_abs_by_region"]["shared_rim"])
    enforce_delta = float(after_enforce["tilt_enforcement_delta"]["tilt_in_l2"])
    theta_in = abs(float(rim_enforce["theta_in_median"]))
    theta_out = abs(float(rim_enforce["theta_out_median"]))
    support_theta_in = abs(float(support_enforce["theta_in_median"]))
    support_theta_out = abs(float(support_enforce["theta_out_median"]))
    if theta_in > 1.0e-3 and support_theta_in < 1.0e-6 and support_theta_out > 1.0e-3:
        return "measurement_uses_wrong_rows"
    if theta_in < 1.0e-6 and theta_out > 1.0e-3 and rim_tilt_grad <= 1.0e-10:
        return "inner_leaflet_not_driven"
    if theta_in < 1.0e-6 and float(rim_relax["theta_in_median"]) > 1.0e-4:
        return "tilt_projection_zeroes_inner_leaflet"
    if theta_in < 1.0e-6 and enforce_delta > 1.0e-8:
        return "tilt_projection_zeroes_inner_leaflet"
    if theta_in < 1.0e-6 and rim_tilt_grad > 1.0e-10:
        return "inner_leaflet_energy_prefers_zero"
    return "inconclusive"


def _classify_profile(report: dict[str, object]) -> str:
    fixed = report["cases"][0]
    profile = fixed["outer_profile_probe"]
    gap_ratio = float(profile["window_leaflet_gap_ratio"])
    log_ratio = abs(float(profile["outer_height_log"]["slope_ratio"]))
    curvature = float(profile["outer_curvature"]["mean_abs_J"])
    if gap_ratio > 0.5:
        return "leaflet_mismatch_dominates"
    if curvature > 0.05:
        return "far_boundary_curvature_pollutes_fit"
    if log_ratio < 0.25:
        return "shape_slope_not_coupled_to_tilt"
    return "inconclusive"


def _case_report(*, label: str, theta_b: float, selected: bool) -> dict[str, object]:
    if selected:
        schedule = _run_canonical_schedule()
        mesh = schedule["mesh"]
        return {
            "label": label,
            "theta_B": float(schedule["theta_B_selected"]),
            "selected_theta": True,
            "near_rim": dict(schedule["near_rim"]),
            "tilt_trace": [_field_trace(mesh, label="selected_theta_final")],
            "module_gradient_probe": {},
            "energy_reconciliation": {},
            "outer_profile_probe": _profile_source_probe(
                mesh, theta_b=float(schedule["theta_B_selected"])
            ),
        }

    minim, traces = _tilt_enforce_trace(float(theta_b))
    gradient_probe = _module_gradient_probe(minim)
    reconciliation = _energy_reconciliation(minim)
    minim.minimize(n_steps=SHAPE_STEPS)
    traces.append(_field_trace(minim.mesh, label="after_shape_minimize"))
    return {
        "label": label,
        "theta_B": float(theta_b),
        "selected_theta": False,
        "tilt_trace": traces,
        "module_gradient_probe": gradient_probe,
        "energy_reconciliation": reconciliation,
        "outer_profile_probe": _profile_source_probe(
            minim.mesh, theta_b=float(theta_b)
        ),
    }


def run_curved_1disk_rim_inner_tilt_profile_audit(
    *,
    theta_b: float = THEORY_THETA_B,
    include_selected: bool = True,
) -> dict[str, object]:
    """Return a diagnostic report for the remaining rim/profile miss."""
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
    rim_class = _classify_rim(report)
    profile_class = _classify_profile(report)
    report["diagnosis"] = {
        "rim_inner_tilt_classification": rim_class,
        "outer_profile_classification": profile_class,
        "allowed_rim_classifications": sorted(RIM_CLASSIFICATIONS),
        "allowed_profile_classifications": sorted(PROFILE_CLASSIFICATIONS),
        "recommended_next_stream": (
            "Write a narrow Feature Contract for the classified blocker before "
            "changing rim tilt constraints, leaflet ownership, shape/tilt coupling, "
            "or profile/far-boundary handling. Do not rescale energy or tune coefficients."
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
    report = run_curved_1disk_rim_inner_tilt_profile_audit(
        theta_b=float(args.theta),
        include_selected=not bool(args.skip_selected),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
