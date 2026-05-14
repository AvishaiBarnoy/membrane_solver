#!/usr/bin/env python3
"""Audit why accepted curved 1-disk shape updates weakly follow the log mode.

This diagnostic is read-only with respect to runtime physics.  It decomposes the
fixed-theta projected shape gradient into interpretable ``z`` directions and
probes equal-norm trial steps through the current runtime energy path.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from runtime.projections.curved_disk import project_curved_free_disk_shape_dofs
from tools.diagnostics.curved_1disk_shape_propagation_blocker import (
    _build_minimizer,
    _shell_stats,
)
from tools.diagnostics.curved_1disk_shared_rim_phi_target_audit import THEORY_THETA_B
from tools.diagnostics.curved_1disk_trumpet_descent_audit import (
    _apply_z_mode,
    _free_outer_mask,
    _module_deltas,
    _outer_modes,
)
from tools.diagnostics.utils import (
    capture_state,
    energy_total,
    radius_labels,
    restore_state,
    row_region,
)

DEFAULT_EPSILON = 1.0e-5
DEFAULT_HORIZONS = (1, 5, 20)
ALLOWED_CLASSIFICATIONS = {
    "support_shell_gradient_dominates",
    "high_frequency_gradient_dominates",
    "coordinate_metric_misweights_outer_shells",
    "line_search_rejects_profile_direction",
    "post_step_tilt_projection_erases_profile_gain",
    "inconclusive",
}


def _prepare_minimizer(theta_b: float):
    minim = _build_minimizer(float(theta_b), max_iter=10)
    mesh = minim.mesh
    minim.enforce_constraints_after_mesh_ops(mesh)
    minim._relax_leaflet_tilts(
        positions=mesh.positions_view(),
        mode=mesh.global_parameters.get("tilt_solve_mode", "fixed"),
    )
    return minim


def _projected_shape_gradient(minim) -> tuple[float, np.ndarray]:
    energy, grad = minim.compute_energy_and_gradient_array()
    minim.project_constraints_array(grad)
    project_curved_free_disk_shape_dofs(minim.mesh, minim.global_params, grad)
    return float(energy), np.asarray(grad[:, 2], dtype=float).copy()


def _unit_l2(values: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    if mask is not None:
        out[~np.asarray(mask, dtype=bool)] = 0.0
    norm = float(np.linalg.norm(out))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.zeros_like(out)
    return out / norm


def _shell_median_smooth(mesh, values: np.ndarray) -> np.ndarray:
    labels = radius_labels(mesh)
    smooth = np.zeros_like(values, dtype=float)
    for radius in sorted(set(float(v) for v in labels)):
        mask = np.isclose(labels, radius, atol=5.0e-9)
        smooth[mask] = float(np.median(values[mask]))
    return smooth


def _near_support_mask(mesh) -> np.ndarray:
    labels = radius_labels(mesh)
    free_mask = _free_outer_mask(mesh)
    support = np.array(
        [row_region(mesh, row) == "outer_support" for row in range(len(labels))]
    )
    free_radii = sorted({float(v) for v in labels[free_mask]})
    near_radii = set(free_radii[:4])
    near_free = np.array([float(v) in near_radii for v in labels], dtype=bool)
    return support | (free_mask & near_free)


def _far_field_mask(mesh) -> np.ndarray:
    labels = radius_labels(mesh)
    free_mask = _free_outer_mask(mesh)
    free_radii = sorted({float(v) for v in labels[free_mask]})
    if not free_radii:
        return np.zeros_like(free_mask)
    cutoff = free_radii[max(0, int(0.75 * (len(free_radii) - 1)))]
    return free_mask & (labels >= cutoff)


def _row_area_weights(mesh) -> np.ndarray:
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    weights = np.zeros(len(mesh.vertex_ids), dtype=float)
    if tri_rows is None or len(tri_rows) == 0:
        return np.ones(len(mesh.vertex_ids), dtype=float)
    tri_pos = positions[tri_rows]
    area = 0.5 * np.linalg.norm(
        np.cross(
            tri_pos[:, 1, :] - tri_pos[:, 0, :], tri_pos[:, 2, :] - tri_pos[:, 0, :]
        ),
        axis=1,
    )
    np.add.at(weights, tri_rows.ravel(), np.repeat(area / 3.0, 3))
    weights = np.where(weights > 1.0e-14, weights, 1.0)
    return weights


def _direction_catalog(mesh, grad_z: np.ndarray) -> dict[str, np.ndarray]:
    log_mode = _outer_modes(mesh)["outer_log_trumpet"]
    descent = -np.asarray(grad_z, dtype=float)
    log_unit = _unit_l2(log_mode)
    residual = descent - float(np.dot(descent, log_unit)) * log_unit
    smooth = _shell_median_smooth(mesh, descent)
    high_frequency = descent - smooth
    area_weights = _row_area_weights(mesh)
    shell_labels = radius_labels(mesh)
    shell_counts = np.ones_like(descent)
    for radius in sorted(set(float(v) for v in shell_labels)):
        mask = np.isclose(shell_labels, radius, atol=5.0e-9)
        shell_counts[mask] = float(np.count_nonzero(mask))

    return {
        "outer_log_trumpet": _unit_l2(log_mode),
        "projected_gradient_descent": _unit_l2(descent),
        "log_residual_gradient": _unit_l2(residual),
        "near_support_gradient": _unit_l2(descent, mask=_near_support_mask(mesh)),
        "far_field_gradient": _unit_l2(descent, mask=_far_field_mask(mesh)),
        "high_frequency_gradient": _unit_l2(
            high_frequency, mask=_free_outer_mask(mesh)
        ),
        "area_weighted_gradient_probe": _unit_l2(descent / area_weights),
        "shell_normalized_gradient_probe": _unit_l2(descent / np.sqrt(shell_counts)),
        "support_suppressed_gradient_probe": _unit_l2(
            descent, mask=~_near_support_mask(mesh)
        ),
    }


def _profile_summary(mesh) -> dict[str, float]:
    labels = radius_labels(mesh)
    free_mask = _free_outer_mask(mesh)
    if not np.any(free_mask):
        return {"outer_log_slope": 0.0, "outer_z_span": 0.0, "outer_shell_count": 0}
    radii: list[float] = []
    zvals: list[float] = []
    z = mesh.positions_view()[:, 2]
    for radius in sorted(set(float(v) for v in labels[free_mask])):
        mask = free_mask & np.isclose(labels, radius, atol=5.0e-9)
        radii.append(float(np.median(labels[mask])))
        zvals.append(float(np.median(z[mask])))
    r = np.asarray(radii, dtype=float)
    vals = np.asarray(zvals, dtype=float)
    if r.size < 2:
        slope = 0.0
    else:
        x = np.log(r / max(float(r[0]), 1.0e-12))
        slope = float(np.polyfit(x, vals, deg=1)[0])
    return {
        "outer_log_slope": slope,
        "outer_z_span": float(np.max(vals) - np.min(vals)) if vals.size else 0.0,
        "outer_shell_count": int(vals.size),
    }


def _probe_direction(
    minim,
    *,
    name: str,
    direction: np.ndarray,
    baseline: dict[str, float],
    grad_z: np.ndarray,
    epsilon: float,
    relax_tilts: bool,
) -> dict[str, object]:
    mesh = minim.mesh
    state = capture_state(mesh)
    _apply_z_mode(mesh, direction, float(epsilon))
    minim._enforce_constraints()
    if relax_tilts:
        minim._relax_leaflet_tilts(
            positions=mesh.positions_view(),
            mode=mesh.global_parameters.get("tilt_solve_mode", "fixed"),
        )
    profile_after = _profile_summary(mesh)
    perturbed = minim.compute_energy_breakdown()
    restore_state(mesh, *state)

    module_deltas = _module_deltas(baseline, perturbed)
    total_delta = energy_total(perturbed) - energy_total(baseline)
    module_delta_sum = energy_total(module_deltas)
    directional_derivative = float(np.dot(grad_z, direction))
    armijo_rhs = 1.0e-4 * float(epsilon) * directional_derivative
    return {
        "name": str(name),
        "epsilon": float(epsilon),
        "relax_tilts": bool(relax_tilts),
        "direction_norm": float(np.linalg.norm(direction)),
        "directional_derivative": directional_derivative,
        "total_delta": float(total_delta),
        "module_delta_sum": float(module_delta_sum),
        "module_residual": float(total_delta - module_delta_sum),
        "armijo_rhs": float(armijo_rhs),
        "accepted_by_decrease": bool(total_delta <= 0.0),
        "accepted_by_armijo": bool(total_delta <= armijo_rhs),
        "profile_after": profile_after,
        "top_module_deltas": [
            {"module": str(module), "delta": float(delta)}
            for module, delta in sorted(
                module_deltas.items(), key=lambda item: abs(item[1]), reverse=True
            )[:6]
        ],
    }


def _direction_summaries(
    mesh,
    directions: dict[str, np.ndarray],
    grad_z: np.ndarray,
) -> list[dict[str, object]]:
    log = directions["outer_log_trumpet"]
    grad_dir = directions["projected_gradient_descent"]
    rows: list[dict[str, object]] = []
    for name, direction in directions.items():
        rows.append(
            {
                "name": str(name),
                "norm": float(np.linalg.norm(direction)),
                "nonzero_rows": int(np.count_nonzero(np.abs(direction) > 0.0)),
                "cosine_with_log": float(np.dot(direction, log)),
                "cosine_with_projected_gradient": float(np.dot(direction, grad_dir)),
                "gradient_dot_direction": float(np.dot(grad_z, direction)),
                "abs_by_shell": _shell_stats(mesh, np.abs(direction)),
            }
        )
    return rows


def _accepted_update_replay(
    *,
    theta_b: float,
    directions: dict[str, np.ndarray],
    horizons: Sequence[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        minim = _prepare_minimizer(float(theta_b))
        mesh = minim.mesh
        before = mesh.positions_view().copy(order="F")
        before_profile = _profile_summary(mesh)
        before_energy = float(minim.compute_energy())
        result = minim.minimize(n_steps=int(horizon))
        after_no_projection = mesh.positions_view().copy(order="F")
        energy_before_tangent_projection = float(minim.compute_energy())
        mesh.project_tilts_to_tangent()
        mesh.increment_version()
        energy_after_tangent_projection = float(minim.compute_energy())
        dz = after_no_projection[:, 2] - before[:, 2]
        dxy = np.linalg.norm(after_no_projection[:, :2] - before[:, :2], axis=1)
        dz_unit = _unit_l2(dz)
        rows.append(
            {
                "n_steps": int(horizon),
                "step_success": bool(result["step_success"]),
                "energy_delta": float(float(result["energy"]) - before_energy),
                "xy_delta_abs_sum": float(np.sum(np.abs(dxy))),
                "z_delta_abs_sum": float(np.sum(np.abs(dz))),
                "profile_before": before_profile,
                "profile_after": _profile_summary(mesh),
                "tangent_projection_energy_delta": float(
                    energy_after_tangent_projection - energy_before_tangent_projection
                ),
                "mode_alignment": {
                    name: {
                        "cosine": float(np.dot(dz_unit, direction)),
                        "dot": float(np.dot(dz, direction)),
                    }
                    for name, direction in directions.items()
                },
                "z_delta_by_shell": _shell_stats(mesh, dz),
            }
        )
    return rows


def _classify(report: dict[str, object]) -> str:
    probes = {
        row["name"]: row
        for row in report["directional_probes"]
        if not bool(row["relax_tilts"])
    }
    summaries = {row["name"]: row for row in report["direction_summaries"]}
    log_probe = probes.get("outer_log_trumpet")
    if log_probe is not None and not bool(log_probe["accepted_by_decrease"]):
        return "line_search_rejects_profile_direction"

    replay = report["accepted_update_replay"]
    if replay:
        tangent = max(
            abs(float(row["tangent_projection_energy_delta"])) for row in replay
        )
        if tangent > 1.0e-5:
            return "post_step_tilt_projection_erases_profile_gain"
        first_align = replay[0]["mode_alignment"]
        support_cos = abs(float(first_align["near_support_gradient"]["cosine"]))
        high_cos = abs(float(first_align["high_frequency_gradient"]["cosine"]))
        log_cos = abs(float(first_align["outer_log_trumpet"]["cosine"]))
        if support_cos > max(0.5, 3.0 * log_cos):
            return "support_shell_gradient_dominates"
        if high_cos > max(0.5, 3.0 * log_cos):
            return "high_frequency_gradient_dominates"

    base_grad = summaries.get("projected_gradient_descent", {})
    area_grad = summaries.get("area_weighted_gradient_probe", {})
    shell_grad = summaries.get("shell_normalized_gradient_probe", {})
    base_log = abs(float(base_grad.get("cosine_with_log", 0.0)))
    metric_log = max(
        abs(float(area_grad.get("cosine_with_log", 0.0))),
        abs(float(shell_grad.get("cosine_with_log", 0.0))),
    )
    if metric_log > max(0.25, 3.0 * base_log):
        return "coordinate_metric_misweights_outer_shells"
    return "inconclusive"


def run_curved_1disk_shape_direction_audit(
    *,
    theta_b: float = THEORY_THETA_B,
    epsilon: float = DEFAULT_EPSILON,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[str, object]:
    """Return the fixed-theta shape direction source audit report."""
    minim = _prepare_minimizer(float(theta_b))
    mesh = minim.mesh
    baseline = minim.compute_energy_breakdown()
    gradient_energy, grad_z = _projected_shape_gradient(minim)
    directions = _direction_catalog(mesh, grad_z)
    probes = []
    for name, direction in directions.items():
        probes.append(
            _probe_direction(
                minim,
                name=name,
                direction=direction,
                baseline=baseline,
                grad_z=grad_z,
                epsilon=float(epsilon),
                relax_tilts=False,
            )
        )
        probes.append(
            _probe_direction(
                minim,
                name=name,
                direction=direction,
                baseline=baseline,
                grad_z=grad_z,
                epsilon=float(epsilon),
                relax_tilts=True,
            )
        )

    report: dict[str, object] = {
        "theta_B": float(theta_b),
        "epsilon": float(epsilon),
        "baseline_energy": {
            "total": energy_total(baseline),
            "modules": baseline,
            "gradient_energy": float(gradient_energy),
        },
        "direction_summaries": _direction_summaries(mesh, directions, grad_z),
        "directional_probes": probes,
        "accepted_update_replay": _accepted_update_replay(
            theta_b=float(theta_b),
            directions=directions,
            horizons=horizons,
        ),
        "diagnosis": {},
    }
    classification = _classify(report)
    report["diagnosis"] = {
        "classification": classification,
        "allowed_classifications": sorted(ALLOWED_CLASSIFICATIONS),
        "summary": (
            "The audit compares equal-norm shape directions only. Any solver "
            "preconditioning, projection, line-search, or energy change requires "
            "a new Feature Contract before implementation."
        ),
        "no_energy_rescaling": True,
        "recommended_next_stream": (
            "Write a narrow Feature Contract for the classified blocker; do not "
            "rescale energy, fit coefficients, or add hidden weights."
        ),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=THEORY_THETA_B)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument(
        "--horizon",
        type=int,
        action="append",
        dest="horizons",
        help="Accepted-update replay horizon. May be passed multiple times.",
    )
    args = parser.parse_args(argv)
    horizons = (
        tuple(int(v) for v in args.horizons) if args.horizons else DEFAULT_HORIZONS
    )
    report = run_curved_1disk_shape_direction_audit(
        theta_b=float(args.theta),
        epsilon=float(args.epsilon),
        horizons=horizons,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
