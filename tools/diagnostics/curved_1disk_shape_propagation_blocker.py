#!/usr/bin/env python3
"""Diagnose fixed-theta curved 1-disk shape propagation blockers.

This report uses theory theta only as a fixed external drive.  It classifies
whether the current lane has a shape force, whether projection removes it,
whether constraint enforcement mutates unrelated DOFs during line search, and
whether the line-search budget reaches an accepted physical descent step.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.projections.curved_disk import project_curved_free_disk_shape_dofs
from runtime.steppers.gradient_descent import GradientDescent
from tools.diagnostics.curved_1disk_shared_rim_phi_target_audit import THEORY_THETA_B
from tools.diagnostics.curved_1disk_theory_benchmark import (
    _refine_once,
    load_free_disk_curved_bilayer_mesh,
)
from tools.diagnostics.free_disk_profile_protocol import (
    configure_free_disk_curved_bilayer_stage2,
)

DEFAULT_ALPHAS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)


def _build_minimizer(theta_b: float, *, max_iter: int) -> Minimizer:
    """Return a configured fixed-theta curved free-disk minimizer."""
    mesh = _refine_once(load_free_disk_curved_bilayer_mesh())
    configure_free_disk_curved_bilayer_stage2(mesh, theta_b=float(theta_b), z_bump=None)
    gp = mesh.global_parameters
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 10)
    gp.set("tilt_tol", 1.0e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)
    gp.set("tilt_thetaB_optimize", False)
    gp.set("tilt_thetaB_value", float(theta_b))
    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 1.0e-3)
    return Minimizer(
        mesh,
        gp,
        GradientDescent(max_iter=int(max_iter)),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1.0e-6,
    )


def _shell_stats(mesh, values: np.ndarray) -> list[dict[str, float | int]]:
    """Aggregate a per-row scalar by radial shell."""
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    keys = np.round(radii, decimals=8)
    rows: list[dict[str, float | int]] = []
    for key in sorted(set(float(k) for k in keys)):
        mask = np.isclose(keys, key, atol=5.0e-9)
        vals = np.asarray(values[mask], dtype=float)
        rows.append(
            {
                "radius": float(np.median(radii[mask])),
                "row_count": int(vals.size),
                "abs_sum": float(np.sum(np.abs(vals))),
                "max_abs": float(np.max(np.abs(vals))) if vals.size else 0.0,
                "median": float(np.median(vals)) if vals.size else 0.0,
            }
        )
    return rows


def _restore_state(mesh, positions, tilts, tilts_in, tilts_out) -> None:
    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = positions[row]
    mesh.set_tilts_from_array(tilts)
    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)
    mesh.increment_version()
    mesh.build_position_cache()


def _line_search_probe(minim: Minimizer, alphas: Sequence[float]) -> dict[str, object]:
    """Return trial-energy deltas along the projected shape descent direction."""
    mesh = minim.mesh
    minim.enforce_constraints_after_mesh_ops(mesh)
    minim._relax_leaflet_tilts(
        positions=mesh.positions_view(),
        mode=mesh.global_parameters.get("tilt_solve_mode", "fixed"),
    )
    energy, grad = minim.compute_energy_and_gradient_array()
    raw_grad = np.array(grad, copy=True)
    minim.project_constraints_array(grad)
    project_curved_free_disk_shape_dofs(minim.mesh, minim.global_params, grad)
    projected_grad = np.array(grad, copy=True)

    positions0 = mesh.positions_view().copy(order="F")
    tilts0 = mesh.tilts_view().copy(order="F")
    tilts_in0 = mesh.tilts_in_view().copy(order="F")
    tilts_out0 = mesh.tilts_out_view().copy(order="F")
    baseline = float(minim._line_search_energy_fn()())

    minim._enforce_constraints()
    alpha0_energy = float(minim._line_search_energy_fn()())
    alpha0_pos_delta = float(np.linalg.norm(mesh.positions_view() - positions0))
    alpha0_tilt_out_delta = float(
        np.max(np.linalg.norm(mesh.tilts_out_view() - tilts_out0, axis=1))
    )
    _restore_state(mesh, positions0, tilts0, tilts_in0, tilts_out0)

    direction = -projected_grad
    movable = ~mesh.fixed_mask
    trials: list[dict[str, float | bool]] = []
    for alpha in alphas:
        trial_positions = positions0.copy(order="F")
        trial_positions[movable] = (
            trial_positions[movable] + float(alpha) * direction[movable]
        )
        for row, vid in enumerate(mesh.vertex_ids):
            mesh.vertices[int(vid)].position[:] = trial_positions[row]
        mesh.increment_version()
        mesh.build_position_cache()
        no_enforce = float(minim.compute_energy())
        minim._enforce_constraints()
        enforced = float(minim._line_search_energy_fn()())
        trials.append(
            {
                "alpha": float(alpha),
                "energy_delta_no_enforce": float(no_enforce - baseline),
                "energy_delta_after_enforce": float(enforced - baseline),
                "accepted_by_decrease": bool(enforced <= baseline),
            }
        )
        _restore_state(mesh, positions0, tilts0, tilts_in0, tilts_out0)

    raw_z = raw_grad[:, 2]
    proj_z = projected_grad[:, 2]
    return {
        "baseline_energy": baseline,
        "gradient_energy": float(energy),
        "raw_gradient_norm": float(np.linalg.norm(raw_grad)),
        "projected_gradient_norm": float(np.linalg.norm(projected_grad)),
        "projection_norm_loss": float(np.linalg.norm(raw_grad - projected_grad)),
        "raw_z_by_shell": _shell_stats(mesh, raw_z),
        "projected_z_by_shell": _shell_stats(mesh, proj_z),
        "alpha0_enforcement": {
            "energy_delta": float(alpha0_energy - baseline),
            "position_delta_norm": alpha0_pos_delta,
            "tilt_out_delta_max": alpha0_tilt_out_delta,
        },
        "trial_alphas": trials,
    }


def _one_step_probe(theta_b: float, *, max_iter: int) -> dict[str, object]:
    """Run one minimizer step and return accepted shell-wise z updates."""
    minim = _build_minimizer(theta_b, max_iter=max_iter)
    mesh = minim.mesh
    minim.enforce_constraints_after_mesh_ops(mesh)
    before = mesh.positions_view().copy(order="F")
    before_energy = float(minim.compute_energy())
    result = minim.minimize(n_steps=1)
    after = mesh.positions_view().copy(order="F")
    dz = after[:, 2] - before[:, 2]
    dxy = np.linalg.norm(after[:, :2] - before[:, :2], axis=1)
    return {
        "max_iter": int(max_iter),
        "step_success": bool(result["step_success"]),
        "energy_delta": float(float(result["energy"]) - before_energy),
        "position_delta_norm": float(np.linalg.norm(after - before)),
        "xy_delta_abs_sum": float(np.sum(np.abs(dxy))),
        "z_delta_abs_sum": float(np.sum(np.abs(dz))),
        "z_delta_by_shell": _shell_stats(mesh, dz),
    }


def _classify(line_probe: dict[str, object], default_step: dict[str, object]) -> str:
    """Return a compact blocker classification."""
    alpha0 = line_probe["alpha0_enforcement"]
    if float(alpha0["energy_delta"]) > 1.0e-8:
        return "constraint_enforcement_mutates_tilt_line_search_baseline"
    trials = list(line_probe["trial_alphas"])
    if not any(bool(row["accepted_by_decrease"]) for row in trials):
        return "no_descent_alpha_found"
    if not bool(default_step["step_success"]):
        return "line_search_backtracking_budget_too_shallow"
    return "shape_update_accepted"


def run_curved_1disk_shape_propagation_blocker(
    *,
    theta_b: float = THEORY_THETA_B,
) -> dict[str, object]:
    """Return the fixed-theta shape propagation blocker report."""
    line_minim = _build_minimizer(theta_b, max_iter=10)
    line_probe = _line_search_probe(line_minim, DEFAULT_ALPHAS)
    default_step = _one_step_probe(theta_b, max_iter=10)
    extended_step = _one_step_probe(theta_b, max_iter=20)
    return {
        "theta_B": float(theta_b),
        "classification": _classify(line_probe, default_step),
        "line_search_probe": line_probe,
        "one_step_default_backtracking": default_step,
        "one_step_extended_backtracking": extended_step,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", type=float, default=THEORY_THETA_B)
    args = parser.parse_args(argv)
    report = run_curved_1disk_shape_propagation_blocker(theta_b=float(args.theta))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
