import math

import pytest

from tools.diagnostics.curved_1disk_theory_benchmark import (
    run_curved_1disk_theory_benchmark,
)


@pytest.mark.benchmark
@pytest.mark.slow
def test_curved_1disk_theory_benchmark_reports_current_tensionless_miss() -> None:
    """Benchmark diagnostic: current main does not yet lock the curved TeX target."""
    report = run_curved_1disk_theory_benchmark()

    assert report["canonical_schedule"] == {
        "theta_scans": 4,
        "theta_offsets": [-0.02, 0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14],
        "shape_steps": 60,
        "refine_steps": 1,
    }

    theory = report["theory"]
    assert theory["lambda_theory"] == pytest.approx(15.0, abs=1.0e-12)
    assert theory["theta_B_opt"] == pytest.approx(0.1845693593, abs=1.0e-10)
    assert theory["phi_star"] == pytest.approx(0.0922846796, abs=1.0e-10)
    assert theory["F_tot"] == pytest.approx(-1.1597607985, abs=1.0e-10)

    assert report["benchmark_lock_passed"] is False
    assert {
        "theta_in_half_split",
        "inner_i1_fit",
        "outer_k1_fit",
        "outer_height_log_fit",
        "inner_elastic",
        "outer_elastic",
        "outer_log_window_sensitivity",
    } <= set(report["benchmark_lock_failures"])

    theta_b_num = float(report["theta_B_selected"])
    assert theta_b_num > 0.0
    assert abs(theta_b_num - float(theory["theta_B_opt"])) > 0.05 * float(
        theory["theta_B_opt"]
    )

    near_rim = report["near_rim"]
    for key in (
        "phi_over_theta_B",
        "theta_in_over_half_theta_B",
        "theta_out_over_half_theta_B",
    ):
        assert math.isfinite(float(near_rim[key]))

    fits = report["fits"]
    for fit_name in ("inner_i1", "outer_k1", "outer_height_log"):
        assert fits[fit_name]
        assert math.isfinite(float(fits[fit_name]["rel_rmse"]))

    curvature = report["outer_curvature"]
    assert math.isfinite(float(curvature["mean_abs_J"]))
    assert math.isfinite(float(curvature["p95_abs_J"]))

    energies = report["energies"]
    for key in (
        "total_numeric",
        "inner_elastic_numeric",
        "outer_elastic_numeric",
        "contact_numeric",
    ):
        assert math.isfinite(float(energies[key]))

    outer_sensitivity = report["outer_window_sensitivity"]
    assert math.isfinite(float(outer_sensitivity["lambda_fit_spread"]))
    assert math.isfinite(float(outer_sensitivity["log_slope_spread"]))
    assert report["last_free_shell_radius"] > 10.0
