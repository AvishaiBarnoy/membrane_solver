#!/usr/bin/env python3
"""Audit whether a small outer trumpet mode is a descent direction.

This diagnostic uses theory theta only as a fixed external drive.  It applies
small, explicit shape perturbations to free outer-shell ``z`` coordinates and
measures the current runtime response without changing physics, constraints, or
solver settings.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from runtime.projections.curved_disk import project_curved_free_disk_shape_dofs
from tools.diagnostics.curved_1disk_shape_propagation_blocker import (
    _build_minimizer,
    _restore_state,
    _shell_stats,
)
from tools.diagnostics.curved_1disk_shared_rim_phi_target_audit import THEORY_THETA_B

DEFAULT_EPSILONS = (1.0e-5, 3.0e-5, 1.0e-4)
ALLOWED_CLASSIFICATIONS = {
    "trumpet_descent_available",
    "trumpet_rejected_by_runtime_energy",
    "trumpet_erased_by_tilt_relaxation",
    "trumpet_reset_by_constraint_enforcement",
    "projection_removes_trumpet_gradient",
    "inconclusive",
}


def _capture_state(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        mesh.positions_view().copy(order="F"),
        mesh.tilts_view().copy(order="F"),
        mesh.tilts_in_view().copy(order="F"),
        mesh.tilts_out_view().copy(order="F"),
    )


def _breakdown_total(breakdown: dict[str, float]) -> float:
    return float(sum(float(value) for value in breakdown.values()))


def _module_deltas(
    baseline: dict[str, float], perturbed: dict[str, float]
) -> dict[str, float]:
    keys = sorted(set(baseline) | set(perturbed))
    return {
        key: float(perturbed.get(key, 0.0) - baseline.get(key, 0.0)) for key in keys
    }


def _row_region(mesh, row: int) -> str:
    opts = getattr(mesh.vertices[int(mesh.vertex_ids[int(row)])], "options", None) or {}
    if str(opts.get("preset") or "") == "disk":
        return "disk"
    group = str(opts.get("rim_slope_match_group") or "")
    if group == "rim":
        return "shared_rim"
    if group == "outer":
        return "outer_support"
    return "outer_free"


def _free_outer_mask(mesh) -> np.ndarray:
    mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    for row in range(len(mesh.vertex_ids)):
        mask[row] = _row_region(mesh, row) == "outer_free"
    if mesh.fixed_mask.shape == mask.shape:
        mask &= ~mesh.fixed_mask
    return mask


def _normalize_mode(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    if not np.any(mask):
        return out
    scale = float(np.max(np.abs(values[mask])))
    if scale <= 0.0 or not np.isfinite(scale):
        return out
    out[mask] = values[mask] / scale
    return out


def _outer_modes(mesh) -> dict[str, np.ndarray]:
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    mask = _free_outer_mask(mesh)
    modes: dict[str, np.ndarray] = {}
    if not np.any(mask):
        return {
            "outer_log_trumpet": np.zeros(len(mesh.vertex_ids), dtype=float),
            "outer_log_trumpet_flipped": np.zeros(len(mesh.vertex_ids), dtype=float),
            "outer_local_shell_bump": np.zeros(len(mesh.vertex_ids), dtype=float),
        }

    r_min = float(np.min(radii[mask]))
    log_values = np.log(np.maximum(radii, r_min) / max(r_min, 1.0e-12))
    log_mode = _normalize_mode(log_values, mask)
    modes["outer_log_trumpet"] = log_mode
    modes["outer_log_trumpet_flipped"] = -log_mode

    free_radii = np.asarray(sorted({round(float(r), 8) for r in radii[mask]}))
    shell = float(free_radii[len(free_radii) // 2])
    shell_mask = mask & np.isclose(np.round(radii, 8), shell, atol=5.0e-9)
    bump = np.zeros(len(mesh.vertex_ids), dtype=float)
    bump[shell_mask] = 1.0
    modes["outer_local_shell_bump"] = bump
    return modes


def _apply_z_mode(mesh, mode: np.ndarray, amplitude: float) -> None:
    positions = mesh.positions_view().copy(order="F")
    positions[:, 2] += float(amplitude) * np.asarray(mode, dtype=float)
    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = positions[row]
    mesh.increment_version()
    mesh.build_position_cache()


def _region_delta_by_shell(mesh, mode: np.ndarray) -> list[dict[str, object]]:
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    rows: list[dict[str, object]] = []
    for radius in sorted({round(float(r), 8) for r in radii}):
        shell_mask = np.isclose(np.round(radii, 8), radius, atol=5.0e-9)
        vals = np.asarray(mode[shell_mask], dtype=float)
        regions = sorted(
            {_row_region(mesh, int(row)) for row in np.flatnonzero(shell_mask)}
        )
        rows.append(
            {
                "radius": float(np.median(radii[shell_mask])),
                "row_count": int(vals.size),
                "regions": regions,
                "mode_abs_sum": float(np.sum(np.abs(vals))),
                "mode_max_abs": float(np.max(np.abs(vals))) if vals.size else 0.0,
            }
        )
    return rows


def _gradient_alignment(minim, mode: np.ndarray) -> dict[str, object]:
    energy, grad = minim.compute_energy_and_gradient_array()
    raw = np.array(grad, copy=True)
    minim.project_constraints_array(grad)
    project_curved_free_disk_shape_dofs(minim.mesh, minim.global_params, grad)
    projected = np.array(grad, copy=True)
    mode_vec = np.zeros_like(projected)

    mode_vec[:, 2] = np.asarray(mode, dtype=float)
    raw_dot = float(np.sum(raw * mode_vec))
    projected_dot = float(np.sum(projected * mode_vec))
    return {
        "energy": float(energy),
        "raw_dot_mode": raw_dot,
        "projected_dot_mode": projected_dot,
        "raw_norm": float(np.linalg.norm(raw)),
        "projected_norm": float(np.linalg.norm(projected)),
        "raw_z_by_shell": _shell_stats(minim.mesh, raw[:, 2]),
        "projected_z_by_shell": _shell_stats(minim.mesh, projected[:, 2]),
    }


def _enforcement_probe(minim, mode: np.ndarray, epsilon: float) -> dict[str, float]:
    mesh = minim.mesh
    state = _capture_state(mesh)
    _apply_z_mode(mesh, mode, float(epsilon))
    before = mesh.positions_view().copy(order="F")
    minim._enforce_constraints()
    after = mesh.positions_view().copy(order="F")
    _restore_state(mesh, *state)
    dz_before = before[:, 2] - state[0][:, 2]
    dz_after = after[:, 2] - state[0][:, 2]
    dxy_after = np.linalg.norm(after[:, :2] - state[0][:, :2], axis=1)
    return {
        "epsilon": float(epsilon),
        "z_before_abs_sum": float(np.sum(np.abs(dz_before))),
        "z_after_abs_sum": float(np.sum(np.abs(dz_after))),
        "z_reset_abs_sum": float(np.sum(np.abs(dz_after - dz_before))),
        "xy_after_abs_sum": float(np.sum(np.abs(dxy_after))),
    }


def _evaluate_case(
    minim,
    *,
    mode: np.ndarray,
    epsilon: float,
    baseline: dict[str, float],
    relax_tilts: bool,
) -> dict[str, object]:
    mesh = minim.mesh
    state = _capture_state(mesh)
    _apply_z_mode(mesh, mode, float(epsilon))
    if relax_tilts:
        minim._relax_leaflet_tilts(
            positions=mesh.positions_view(),
            mode=mesh.global_parameters.get("tilt_solve_mode", "fixed"),
        )
    perturbed = minim.compute_energy_breakdown()
    _restore_state(mesh, *state)

    module_deltas = _module_deltas(baseline, perturbed)
    total_delta = _breakdown_total(perturbed) - _breakdown_total(baseline)
    module_delta_sum = _breakdown_total(module_deltas)
    ranked = sorted(module_deltas.items(), key=lambda item: abs(item[1]), reverse=True)
    return {
        "epsilon": float(epsilon),
        "relax_tilts": bool(relax_tilts),
        "total_delta": float(total_delta),
        "module_delta_sum": float(module_delta_sum),
        "module_residual": float(total_delta - module_delta_sum),
        "module_deltas": module_deltas,
        "top_module_deltas": [
            {"module": str(name), "delta": float(delta)} for name, delta in ranked[:6]
        ],
    }


def _mode_report(
    minim,
    *,
    name: str,
    mode: np.ndarray,
    epsilons: Sequence[float],
    baseline: dict[str, float],
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for epsilon in epsilons:
        for sign in (1.0, -1.0):
            signed = float(sign * epsilon)
            cases.append(
                _evaluate_case(
                    minim,
                    mode=mode,
                    epsilon=signed,
                    baseline=baseline,
                    relax_tilts=False,
                )
            )
            cases.append(
                _evaluate_case(
                    minim,
                    mode=mode,
                    epsilon=signed,
                    baseline=baseline,
                    relax_tilts=True,
                )
            )
    return {
        "name": str(name),
        "mode_norm": float(np.linalg.norm(mode)),
        "mode_max_abs": float(np.max(np.abs(mode))) if mode.size else 0.0,
        "nonzero_rows": int(np.count_nonzero(np.abs(mode) > 0.0)),
        "shell_support": _region_delta_by_shell(minim.mesh, mode),
        "gradient_alignment": _gradient_alignment(minim, mode),
        "enforcement_probe": _enforcement_probe(
            minim, mode, float(max(abs(eps) for eps in epsilons))
        ),
        "finite_difference_cases": cases,
    }


def _accepted_update_alignment(
    *,
    theta_b: float,
    modes: dict[str, np.ndarray],
    n_steps_values: Sequence[int] = (1, 5),
) -> list[dict[str, object]]:
    """Return accepted minimizer shape-update alignment with probe modes."""
    rows: list[dict[str, object]] = []
    for n_steps in n_steps_values:
        minim = _build_minimizer(float(theta_b), max_iter=10)
        mesh = minim.mesh
        minim.enforce_constraints_after_mesh_ops(mesh)
        minim._relax_leaflet_tilts(
            positions=mesh.positions_view(),
            mode=mesh.global_parameters.get("tilt_solve_mode", "fixed"),
        )
        before = mesh.positions_view().copy(order="F")
        before_energy = float(minim.compute_energy())
        result = minim.minimize(n_steps=int(n_steps))
        after = mesh.positions_view().copy(order="F")
        dz = after[:, 2] - before[:, 2]
        dxy = np.linalg.norm(after[:, :2] - before[:, :2], axis=1)
        mode_rows: dict[str, dict[str, float]] = {}
        for name, mode in modes.items():
            mode = np.asarray(mode, dtype=float)
            denom = float(np.linalg.norm(dz) * np.linalg.norm(mode))
            cosine = float(np.dot(dz, mode) / denom) if denom > 0.0 else 0.0
            mode_rows[str(name)] = {
                "dot": float(np.dot(dz, mode)),
                "cosine": cosine,
            }
        rows.append(
            {
                "n_steps": int(n_steps),
                "step_success": bool(result["step_success"]),
                "energy_delta": float(float(result["energy"]) - before_energy),
                "xy_delta_abs_sum": float(np.sum(np.abs(dxy))),
                "z_delta_abs_sum": float(np.sum(np.abs(dz))),
                "mode_alignment": mode_rows,
                "z_delta_by_shell": _shell_stats(mesh, dz),
            }
        )
    return rows


def _case_delta(
    mode_report: dict[str, object], *, epsilon: float, relax_tilts: bool
) -> float:
    for case in mode_report["finite_difference_cases"]:
        if (
            case["epsilon"] == float(epsilon)
            and bool(case["relax_tilts"]) == relax_tilts
        ):
            return float(case["total_delta"])
    return float("nan")


def _classify(report: dict[str, object]) -> str:
    modes = {mode["name"]: mode for mode in report["modes"]}
    trumpet = modes.get("outer_log_trumpet")
    flipped = modes.get("outer_log_trumpet_flipped")
    if trumpet is None or flipped is None:
        return "inconclusive"
    eps = float(report["epsilons"][0])
    enforce = trumpet["enforcement_probe"]
    if float(enforce["z_reset_abs_sum"]) > 0.25 * max(
        float(enforce["z_before_abs_sum"]), 1.0e-18
    ):
        return "trumpet_reset_by_constraint_enforcement"
    align = trumpet["gradient_alignment"]
    raw_dot = abs(float(align["raw_dot_mode"]))
    projected_dot = abs(float(align["projected_dot_mode"]))
    if raw_dot > 1.0e-12 and projected_dot < 0.1 * raw_dot:
        return "projection_removes_trumpet_gradient"

    frozen_pos = _case_delta(trumpet, epsilon=eps, relax_tilts=False)
    frozen_neg = _case_delta(flipped, epsilon=eps, relax_tilts=False)
    relaxed_pos = _case_delta(trumpet, epsilon=eps, relax_tilts=True)
    relaxed_neg = _case_delta(flipped, epsilon=eps, relax_tilts=True)
    frozen_descent = min(frozen_pos, frozen_neg)
    relaxed_descent = min(relaxed_pos, relaxed_neg)
    if frozen_descent < 0.0 and relaxed_descent >= 0.0:
        return "trumpet_erased_by_tilt_relaxation"
    if relaxed_descent < 0.0:
        return "trumpet_descent_available"
    if frozen_descent >= 0.0 and relaxed_descent >= 0.0:
        return "trumpet_rejected_by_runtime_energy"
    return "inconclusive"


def _rank_module_responses(report: dict[str, object]) -> list[dict[str, object]]:
    modes = {mode["name"]: mode for mode in report["modes"]}
    trumpet = modes.get("outer_log_trumpet")
    if trumpet is None:
        return []
    eps = float(report["epsilons"][0])
    accum: dict[str, float] = {}
    for case in trumpet["finite_difference_cases"]:
        if case["epsilon"] != eps or bool(case["relax_tilts"]):
            continue
        for name, delta in case["module_deltas"].items():
            accum[str(name)] = accum.get(str(name), 0.0) + float(delta)
    ranked = sorted(accum.items(), key=lambda item: abs(item[1]), reverse=True)
    return [
        {
            "module": str(name),
            "signed_delta": float(delta),
            "interpretation": "resists_mode" if float(delta) > 0.0 else "favors_mode",
        }
        for name, delta in ranked
        if abs(float(delta)) > 1.0e-15
    ][:8]


def run_curved_1disk_trumpet_descent_audit(
    *,
    theta_b: float = THEORY_THETA_B,
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
) -> dict[str, object]:
    """Return a diagnostic-only trumpet descent audit report."""
    minim = _build_minimizer(float(theta_b), max_iter=10)
    mesh = minim.mesh
    minim.enforce_constraints_after_mesh_ops(mesh)
    minim._relax_leaflet_tilts(
        positions=mesh.positions_view(),
        mode=mesh.global_parameters.get("tilt_solve_mode", "fixed"),
    )
    baseline = minim.compute_energy_breakdown()
    modes = _outer_modes(mesh)
    report: dict[str, object] = {
        "theta_B": float(theta_b),
        "epsilons": [float(eps) for eps in epsilons],
        "baseline_energy": {
            "total": _breakdown_total(baseline),
            "modules": baseline,
        },
        "mode_construction": {
            "outer_free_row_count": int(np.count_nonzero(_free_outer_mask(mesh))),
            "z_only": True,
            "amplitude_is_probe_not_parameter": True,
            "no_energy_rescaling": True,
        },
        "modes": [
            _mode_report(
                minim,
                name=name,
                mode=mode,
                epsilons=epsilons,
                baseline=baseline,
            )
            for name, mode in modes.items()
        ],
        "accepted_update_alignment": _accepted_update_alignment(
            theta_b=float(theta_b),
            modes=modes,
        ),
        "diagnosis": {},
    }
    classification = _classify(report)
    report["diagnosis"] = {
        "classification": classification,
        "allowed_classifications": sorted(ALLOWED_CLASSIFICATIONS),
        "module_response_rank": _rank_module_responses(report),
        "recommended_next_stream": (
            "Write a narrow Feature Contract for the classified blocker before "
            "changing runtime physics; do not rescale energy or fit coefficients."
        ),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=THEORY_THETA_B)
    parser.add_argument(
        "--epsilon",
        type=float,
        action="append",
        dest="epsilons",
        help="Small finite-difference amplitude. May be passed multiple times.",
    )
    args = parser.parse_args(argv)
    epsilons = (
        tuple(float(eps) for eps in args.epsilons)
        if args.epsilons
        else DEFAULT_EPSILONS
    )
    report = run_curved_1disk_trumpet_descent_audit(
        theta_b=float(args.theta), epsilons=epsilons
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
