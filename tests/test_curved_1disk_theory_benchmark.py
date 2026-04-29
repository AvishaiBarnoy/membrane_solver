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
    assert set(report["benchmark_lock_failures"]) == {
        "theta_B_opt",
        "theta_in_half_split",
        "inner_i1_fit",
        "outer_k1_fit",
        "outer_height_log_fit",
        "outer_curvature",
        "total_energy",
        "inner_elastic",
        "outer_elastic",
        "contact_energy",
        "outer_log_window_sensitivity",
    }

    theta_b_num = float(report["theta_B_selected"])
    assert theta_b_num < 0.75 * float(theory["theta_B_opt"])

    near_rim = report["near_rim"]
    assert float(near_rim["phi_over_theta_B"]) == pytest.approx(0.5, rel=0.10)
    assert float(near_rim["theta_in_over_half_theta_B"]) < 0.85
    assert float(near_rim["theta_out_over_half_theta_B"]) == pytest.approx(
        1.0, rel=0.15
    )

    fits = report["fits"]
    inner_fit = fits["inner_i1"]
    assert float(inner_fit["lambda_fit"]) > 4.0 * float(theory["lambda_theory"])
    assert float(inner_fit["rel_rmse"]) > 0.05
    assert inner_fit["window"] == [0.25, 0.75]

    outer_fit = fits["outer_k1"]
    assert float(outer_fit["lambda_fit"]) == pytest.approx(10.0, rel=0.01)
    assert float(outer_fit["rel_rmse"]) > 0.20
    assert outer_fit["window"] == [2.0, 10.0]
    assert float(outer_fit["leaflet_mismatch_median"]) > 0.40

    height_fit = fits["outer_height_log"]
    assert abs(float(height_fit["slope_fit"])) > 1.0e-10
    assert float(height_fit["rel_rmse"]) > 0.20
    assert height_fit["window"] == [3.0, 10.0]

    curvature = report["outer_curvature"]
    assert 0.01 < float(curvature["mean_abs_J"]) < 0.05
    assert float(curvature["p95_abs_J"]) > 0.15

    energies = report["energies"]
    assert float(energies["total_numeric"]) > float(theory["F_tot"])
    assert 0.001 < float(energies["inner_elastic_numeric"]) < 0.01
    assert float(energies["outer_elastic_numeric"]) > 1.0
    assert abs(float(energies["contact_numeric"])) < abs(float(theory["F_cont"]))

    outer_sensitivity = report["outer_window_sensitivity"]
    assert float(outer_sensitivity["lambda_fit_spread"]) <= 0.10
    assert float(outer_sensitivity["log_slope_spread"]) > 0.10
    assert report["last_free_shell_radius"] > 10.0
