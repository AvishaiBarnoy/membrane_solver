from tools.diagnostics.curved_1disk_shape_direction_audit import (
    run_curved_1disk_shape_direction_audit,
)
from tools.diagnostics.curved_1disk_shared_rim_phi_target_audit import THEORY_THETA_B
from tools.diagnostics.curved_1disk_theory_benchmark import _run_curved_theta_candidate


def test_curved_1disk_support_transition_band_no_longer_dominates_shape_update() -> (
    None
):
    report = run_curved_1disk_shape_direction_audit(horizons=(1,))

    assert report["diagnosis"]["classification"] != "support_shell_gradient_dominates"
    step = report["accepted_update_replay"][0]
    support_cos = abs(float(step["mode_alignment"]["near_support_gradient"]["cosine"]))
    assert support_cos < 0.40


def test_curved_1disk_fixed_theta_short_relaxation_moves_log_slope_physical() -> None:
    report = run_curved_1disk_shape_direction_audit(horizons=(1,))

    step = report["accepted_update_replay"][0]
    assert float(step["profile_after"]["outer_log_slope"]) > 0.0
    assert float(step["xy_delta_abs_sum"]) < 1.0e-10
    assert float(step["z_delta_abs_sum"]) > 0.0


def test_curved_1disk_log_direction_remains_valid_descent_after_support_fix() -> None:
    report = run_curved_1disk_shape_direction_audit(horizons=(1,))

    log_probe = next(
        row
        for row in report["directional_probes"]
        if row["name"] == "outer_log_trumpet" and not row["relax_tilts"]
    )
    assert bool(log_probe["accepted_by_armijo"]) is True
    assert float(log_probe["total_delta"]) < 0.0


def test_curved_1disk_transition_regularization_does_not_prefer_high_branch() -> None:
    theory = _run_curved_theta_candidate(float(THEORY_THETA_B))
    high = _run_curved_theta_candidate(0.22)

    theory_energy = float(theory["near_rim"]["total_energy"])
    high_energy = float(high["near_rim"]["total_energy"])
    assert theory_energy <= high_energy
