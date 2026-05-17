import logging

import numpy as np

logger = logging.getLogger("membrane_solver")


def _optimize_thetaB_scalar(minimizer, *, tilt_mode: str, iteration: int) -> None:
    """Optionally optimize the scalar thetaB by sampling reduced energies.

    This treats thetaB as a global scalar degree of freedom and updates it
    by comparing the total energy after a partial tilt relaxation for a few
    candidate thetaB values.
    """
    mode_match = (
        str(minimizer.global_params.get("rim_slope_match_mode") or "").strip().lower()
    )
    trace_radius = minimizer.global_params.get("parity_trace_layer_radius")
    outer_shells = int(minimizer.global_params.get("parity_outer_shells", 0) or 0)
    if (
        mode_match == "physical_edge_staggered_v1"
        and trace_radius is not None
        and outer_shells > 0
    ):
        _record_thetaB_scan(
            minimizer,
            {
                "iteration": int(iteration),
                "status": "skipped_physical_edge_scaffold_trace_lane",
                "base_thetaB": float(
                    minimizer.global_params.get("tilt_thetaB_value") or 0.0
                ),
                "selected_thetaB": float(
                    minimizer.global_params.get("tilt_thetaB_value") or 0.0
                ),
                "trace_radius": float(trace_radius),
                "candidate_energies": [],
            },
        )
        return
    if not bool(minimizer.global_params.get("tilt_thetaB_optimize", False)):
        return

    every = int(minimizer.global_params.get("tilt_thetaB_optimize_every", 10) or 10)
    if every <= 0:
        every = 1
    if int(iteration) % every != 0:
        return

    delta = float(
        minimizer.global_params.get("tilt_thetaB_optimize_delta", 0.02) or 0.0
    )
    if delta <= 0.0:
        return

    # Warm-start from the current relaxed state.
    base_thetaB = float(minimizer.global_params.get("tilt_thetaB_value") or 0.0)
    base_tin = minimizer.mesh.tilts_in_view().copy(order="F")
    base_tout = minimizer.mesh.tilts_out_view().copy(order="F")
    scan_record: dict[str, object] = {
        "iteration": int(iteration),
        "status": "evaluated",
        "base_thetaB": float(base_thetaB),
        "selected_thetaB": float(base_thetaB),
        "trace_radius": None if trace_radius is None else float(trace_radius),
        "candidate_energies": [],
    }

    # Use a smaller inner relaxation budget for the thetaB scan to keep the
    # optimization cheap, but still responsive.
    orig_inner_steps = minimizer.global_params.get("tilt_inner_steps", None)
    scan_steps = int(
        minimizer.global_params.get("tilt_thetaB_optimize_inner_steps", 20) or 20
    )
    if scan_steps < 1:
        scan_steps = 1
    minimizer.global_params.set("tilt_inner_steps", scan_steps)

    # Guard threshold prevents thetaB scan candidates from diverging.
    scan_guard_factor = float(
        minimizer.global_params.get("tilt_relax_energy_guard_factor", 0.0) or 0.0
    )

    def eval_candidate(thetaB_val: float) -> tuple[float, np.ndarray, np.ndarray]:
        minimizer.global_params.set("tilt_thetaB_value", float(thetaB_val))
        minimizer._set_leaflet_tilts_from_arrays_fast(base_tin, base_tout)
        # Relax tilts only; shape is handled by the main loop.
        minimizer._relax_leaflet_tilts(
            positions=minimizer.mesh.positions_view(), mode=tilt_mode
        )
        e = float(minimizer.compute_energy())
        # If the candidate's energy is much worse than the baseline,
        # discard it to prevent tilt divergence from corrupting
        # the selection.
        if scan_guard_factor > 0.0:
            scan_threshold = max(
                float(
                    minimizer.global_params.get("tilt_relax_energy_guard_min", 1e-4)
                    or 1e-4
                ),
                abs(e0) * scan_guard_factor,
            )
            if e > scan_threshold:
                scan_record["candidate_energies"].append(
                    {
                        "thetaB": float(thetaB_val),
                        "energy": float(e),
                        "discarded": True,
                    }
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "thetaB scan: thetaB=%.6g candidate E=%.6g "
                        "exceeds guard threshold %.6g; discarding.",
                        thetaB_val,
                        e,
                        scan_threshold,
                    )
                # Return sentinel: restore base tilts and flag as bad.
                minimizer._set_leaflet_tilts_from_arrays_fast(base_tin, base_tout)
                return (float("inf"), base_tin, base_tout)
        scan_record["candidate_energies"].append(
            {"thetaB": float(thetaB_val), "energy": float(e), "discarded": False}
        )
        return (
            e,
            minimizer.mesh.tilts_in_view().copy(order="F"),
            minimizer.mesh.tilts_out_view().copy(order="F"),
        )

    try:
        e0 = float(minimizer.compute_energy())
        scan_record["candidate_energies"].append(
            {"thetaB": float(base_thetaB), "energy": float(e0), "discarded": False}
        )
        e_minus, tin_minus, tout_minus = eval_candidate(base_thetaB - delta)
        e_plus, tin_plus, tout_plus = eval_candidate(base_thetaB + delta)
    finally:
        # Restore the user's configured inner step budget.
        if orig_inner_steps is None:
            minimizer.global_params.unset("tilt_inner_steps")
        else:
            minimizer.global_params.set("tilt_inner_steps", orig_inner_steps)

    # Pick the best among the sampled points (cheap coordinate descent).
    best = min(
        [
            (e0, base_thetaB, base_tin, base_tout),
            (e_minus, base_thetaB - delta, tin_minus, tout_minus),
            (e_plus, base_thetaB + delta, tin_plus, tout_plus),
        ],
        key=lambda x: x[0],
    )

    best_e, best_thetaB, best_tin, best_tout = best
    # Guard: if the best candidate is worse than the starting state,
    # roll back to the original thetaB and tilts to prevent the
    # optimizer from pushing the system into an unstable configuration
    # (e.g. after mesh refinement when the tilt eigenvalue spectrum
    # has changed).
    if best_e > e0:
        minimizer.global_params.set("tilt_thetaB_value", float(base_thetaB))
        minimizer._set_leaflet_tilts_from_arrays_fast(base_tin, base_tout)
        scan_record["status"] = "rollback"
        scan_record["selected_thetaB"] = float(base_thetaB)
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(
                "thetaB optimize: i=%d rollback (best E %.6g > base E %.6g); "
                "keeping thetaB=%.6g.",
                int(iteration),
                best_e,
                e0,
                base_thetaB,
            )
    else:
        minimizer.global_params.set("tilt_thetaB_value", float(best_thetaB))
        minimizer._set_leaflet_tilts_from_arrays_fast(best_tin, best_tout)
        scan_record["selected_thetaB"] = float(best_thetaB)
    _record_thetaB_scan(minimizer, scan_record)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "thetaB optimize: i=%d thetaB %.6g -> %.6g (E %.6g -> %.6g)",
            int(iteration),
            base_thetaB,
            float(minimizer.global_params.get("tilt_thetaB_value") or 0.0),
            e0,
            best_e,
        )


def _record_thetaB_scan(minimizer, record: dict[str, object]) -> None:
    traces = getattr(minimizer.mesh, "_thetaB_scan_trace", None)
    if traces is None:
        traces = []
        setattr(minimizer.mesh, "_thetaB_scan_trace", traces)
    traces.append(record)
