"""Diagnostic and consistency audit helpers for the minimization loop."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Dict

import numpy as np

from runtime.minimizer_helpers import capture_diagnostic_state, restore_diagnostic_state

if TYPE_CHECKING:
    from runtime.minimizer import Minimizer

logger = logging.getLogger("membrane_solver")


def log_energy_phase(iteration: int, phase: str, energy: float) -> None:
    """Log the energy at a specific phase of an iteration."""
    logger.debug("Iteration %d: %s energy=%.6f", iteration, phase, energy)


def log_step_direction_stats(iteration: int, grad_arr: np.ndarray) -> None:
    """Log statistics about the step direction derived from the gradient."""
    norm = float(np.linalg.norm(grad_arr))
    if norm <= 0.0:
        logger.debug("Iteration %d: grad_norm=0", iteration)
        return
    step_dir = -grad_arr / norm
    logger.debug(
        "Iteration %d: grad_norm=%.3e step_dir_norm=%.3e",
        iteration,
        norm,
        float(np.linalg.norm(step_dir)),
    )


def run_debug_diagnostic(minimizer: Minimizer, fn):
    """Run a diagnostic without mutating the live simulation state.

    Diagnostics are evaluated on a deep-copied mesh/parameters to guarantee
    observational behavior even when energy modules populate caches.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return fn()

    # Only handle bound methods on this Minimizer instance. For other
    # callables, fall back to the snapshot/restore guard.
    fn_self = getattr(fn, "__self__", None)
    fn_name = getattr(fn, "__name__", None)
    if fn_self is not minimizer or not isinstance(fn_name, str):
        snapshot = capture_diagnostic_state(
            minimizer.mesh,
            minimizer.global_params,
            uses_leaflet_tilts=minimizer._uses_leaflet_tilts(),
        )
        try:
            return fn()
        finally:
            restore_diagnostic_state(minimizer.mesh, minimizer.global_params, snapshot)

    mesh_copy = minimizer.mesh.copy()
    gp_copy = copy.deepcopy(minimizer.global_params)
    mesh_copy.global_parameters = gp_copy

    from runtime.minimizer import Minimizer as MinimizerClass

    diag = MinimizerClass(
        mesh_copy,
        gp_copy,
        minimizer.stepper,
        minimizer.energy_manager,
        minimizer.constraint_manager,
        energy_modules=list(getattr(mesh_copy, "energy_modules", [])),
        constraint_modules=list(getattr(mesh_copy, "constraint_modules", [])),
        step_size=float(minimizer.step_size),
        tol=float(minimizer.tol),
        quiet=True,
    )

    if fn_name == "compute_energy":
        return diag.compute_energy()
    if fn_name == "compute_energy_breakdown":
        return diag.compute_energy_breakdown()
    if fn_name == "compute_energy_and_gradient_array":
        return diag.compute_energy_and_gradient_array()

    # Fallback: if this is another Minimizer method that takes no args,
    # prefer calling it on the copied Minimizer instance.
    target = getattr(diag, fn_name, None)
    if callable(target):
        try:
            return target()
        except TypeError:
            pass

    # Last resort: snapshot/restore guard on the live state.
    snapshot = capture_diagnostic_state(
        minimizer.mesh,
        minimizer.global_params,
        uses_leaflet_tilts=minimizer._uses_leaflet_tilts(),
    )
    try:
        return fn()
    finally:
        restore_diagnostic_state(minimizer.mesh, minimizer.global_params, snapshot)


def diagnostic_energy(minimizer: Minimizer) -> float:
    """Energy computation for diagnostics (debug-safe)."""
    return float(run_debug_diagnostic(minimizer, minimizer.compute_energy))


def diagnostic_energy_breakdown(minimizer: Minimizer) -> Dict[str, float]:
    """Energy breakdown for diagnostics (debug-safe)."""
    return run_debug_diagnostic(minimizer, minimizer.compute_energy_breakdown)


def log_energy_consistency(minimizer: Minimizer, label: str) -> None:
    """Audit energy scalar/array agreement and log any mismatch."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        energy_scalar = float(run_debug_diagnostic(minimizer, minimizer.compute_energy))
        energy_array, _ = run_debug_diagnostic(
            minimizer, minimizer.compute_energy_and_gradient_array
        )
        energy_array = float(energy_array)
    except Exception as exc:
        logger.debug("Energy consistency check (%s) skipped: %s", label, exc)
        return

    logger.debug(
        "Energy consistency (%s): scalar=%.6f array=%.6f",
        label,
        energy_scalar,
        energy_array,
    )

    diff = abs(energy_scalar - energy_array)
    tol = 1e-8 * max(1.0, abs(energy_scalar), abs(energy_array))
    if diff <= tol:
        return

    try:
        breakdown = run_debug_diagnostic(minimizer, minimizer.compute_energy_breakdown)
    except Exception as exc:
        logger.debug("Energy breakdown failed during consistency check: %s", exc)
        return

    top_terms = sorted(breakdown.items(), key=lambda item: abs(item[1]), reverse=True)[
        :5
    ]
    summary = ", ".join(f"{name}={value:.6f}" for name, value in top_terms)
    logger.warning(
        "Energy consistency mismatch (%s): |Δ|=%.6e (scalar=%.6f array=%.6f). "
        "Top terms: %s",
        label,
        diff,
        energy_scalar,
        energy_array,
        summary,
    )


def log_accepted_step_stats(
    minimizer: Minimizer,
    *,
    iteration: int,
    E_before: float,
    E_accepted: float,
    step_size: float,
) -> None:
    """Log statistics for an accepted minimization step."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    E_after = minimizer.compute_energy()
    diff_acc = abs(E_after - E_accepted)
    tol_acc = 1e-8 * max(1.0, abs(E_after), abs(E_accepted))
    max_rel_violation_dbg = 0.0
    vol_msgs: list[str] = []
    if minimizer.mesh.bodies:
        for body in minimizer.mesh.bodies.values():
            target = body.target_volume
            if target is None:
                target = body.options.get("target_volume")
            if target is None:
                continue
            current = body.compute_volume(minimizer.mesh)
            denom = max(abs(target), 1.0)
            rel = (current - target) / denom
            max_rel_violation_dbg = max(max_rel_violation_dbg, abs(rel))
            vol_msgs.append(
                "body %d: V=%.6f, V0=%.6f, relΔV=%.3e"
                % (body.index, current, target, rel)
            )
    logger.debug(
        "Accepted step %d: E_before=%.6f, E_after=%.6f, "
        "E_accepted=%.6f, |E_after-E_accepted|=%.3e (tol %.3e), "
        "step_size=%.3e, max_relΔV=%.3e",
        iteration,
        E_before,
        E_after,
        E_accepted,
        diff_acc,
        tol_acc,
        step_size,
        max_rel_violation_dbg,
    )
    if vol_msgs:
        logger.debug("Volume diagnostics: %s", "; ".join(vol_msgs))
    if diff_acc > tol_acc:
        try:
            breakdown = minimizer.compute_energy_breakdown()
            top_terms = sorted(
                breakdown.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:5]
            summary = ", ".join(f"{name}={value:.6f}" for name, value in top_terms)
            logger.warning(
                "Accepted energy mismatch at step %d: "
                "E_accepted=%.6f E_after=%.6f |Δ|=%.6e. Top terms: %s",
                iteration,
                E_accepted,
                E_after,
                diff_acc,
                summary,
            )
        except Exception as exc:
            logger.debug(
                "Accepted energy breakdown failed at step %d: %s",
                iteration,
                exc,
            )


def log_lagrange_tangency_check(
    minimizer: Minimizer, grad: Dict[int, np.ndarray]
) -> None:
    """Optional DEBUG-level diagnostic: in Lagrange mode the projected
    gradient should be (numerically) tangent to each fixed-volume
    manifold, i.e. ⟨∇E, ∇V_body⟩ ≈ 0.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    mode = minimizer.global_params.get("volume_constraint_mode", "lagrange")
    if mode == "lagrange" and minimizer.mesh.bodies:
        max_abs_dot = 0.0
        for body in minimizer.mesh.bodies.values():
            V_target = body.target_volume
            if V_target is None:
                V_target = body.options.get("target_volume")

            if V_target is None:
                continue

            _, vol_grad = body.compute_volume_and_gradient(minimizer.mesh)

            dot = 0.0
            for vidx, gVi in vol_grad.items():
                gE = grad.get(vidx)
                if gE is not None:
                    dot += float(np.dot(gE, gVi))
            max_abs_dot = max(max_abs_dot, abs(dot))
        logger.debug(
            "Lagrange tangency check: max |<∇E, ∇V>| = %.3e",
            max_abs_dot,
        )


def log_debug_energy_context(minimizer: Minimizer, iteration: int) -> None:
    """Log detailed energy context for debugging."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    breakdown = diagnostic_energy_breakdown(minimizer)
    thetaB_contact = float(breakdown.get("tilt_thetaB_contact_in", 0.0))
    thetaB_val = float(minimizer.global_params.get("tilt_thetaB_value") or 0.0)
    line_search_reduced = bool(
        minimizer.global_params.get("line_search_reduced_energy", False)
    )
    line_search_steps = int(
        minimizer.global_params.get("line_search_reduced_tilt_inner_steps", 0) or 0
    )
    guard_factor = float(
        minimizer.global_params.get("tilt_relax_energy_guard_factor", 0.0) or 0.0
    )
    tilt_enforced = minimizer._last_mesh_op_tilt_constraints_enforced
    logger.debug(
        "Debug energy context i=%d: tilt_thetaB_contact_in=%.6g "
        "tilt_thetaB_value=%.6g line_search_reduced=%s "
        "line_search_steps=%d tilt_relax_guard=%.6g "
        "tilt_constraints_enforced=%s",
        iteration,
        thetaB_contact,
        thetaB_val,
        line_search_reduced,
        line_search_steps,
        guard_factor,
        tilt_enforced,
    )


def check_gauss_bonnet(minimizer: Minimizer) -> None:
    """Emit Gauss-Bonnet diagnostics if enabled."""
    if not bool(minimizer.global_params.get("gauss_bonnet_monitor", False)):
        return

    from runtime.diagnostics.gauss_bonnet import GaussBonnetMonitor

    if minimizer._gauss_bonnet_monitor is None:
        eps_angle = float(minimizer.global_params.get("gauss_bonnet_eps_angle", 1e-4))
        c1 = float(minimizer.global_params.get("gauss_bonnet_c1", 1.0))
        c2 = float(minimizer.global_params.get("gauss_bonnet_c2", 1.0))
        minimizer._gauss_bonnet_monitor = GaussBonnetMonitor.from_mesh(
            minimizer.mesh, eps_angle=eps_angle, c1=c1, c2=c2
        )

    report = minimizer._gauss_bonnet_monitor.evaluate(minimizer.mesh)
    if not report["ok"]:
        logger.warning(
            "Gauss-Bonnet drift exceeded tolerance: |ΔG|=%.3e (tol %.3e).",
            report["drift_G"],
            report["tol_G"],
        )
