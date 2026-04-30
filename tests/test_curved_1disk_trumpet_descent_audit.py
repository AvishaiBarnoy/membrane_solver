import math

import pytest

from tools.diagnostics.curved_1disk_trumpet_descent_audit import (
    ALLOWED_CLASSIFICATIONS,
    run_curved_1disk_trumpet_descent_audit,
)


def test_curved_1disk_trumpet_descent_audit_reports_required_schema() -> None:
    report = run_curved_1disk_trumpet_descent_audit(epsilons=(1.0e-5,))

    assert report["mode_construction"]["z_only"] is True
    assert report["mode_construction"]["amplitude_is_probe_not_parameter"] is True
    assert report["mode_construction"]["no_energy_rescaling"] is True
    assert int(report["mode_construction"]["outer_free_row_count"]) > 0
    assert report["diagnosis"]["classification"] in ALLOWED_CLASSIFICATIONS
    assert "rescale" in report["diagnosis"]["recommended_next_stream"]
    assert "do not rescale" in report["diagnosis"]["recommended_next_stream"]

    modes = {mode["name"]: mode for mode in report["modes"]}
    assert set(modes) == {
        "outer_log_trumpet",
        "outer_log_trumpet_flipped",
        "outer_local_shell_bump",
    }
    assert modes["outer_log_trumpet"]["nonzero_rows"] > 0
    assert modes["outer_log_trumpet"]["mode_max_abs"] == pytest.approx(1.0)
    assert modes["outer_log_trumpet_flipped"]["mode_norm"] == pytest.approx(
        modes["outer_log_trumpet"]["mode_norm"]
    )
    updates = report["accepted_update_alignment"]
    assert [int(row["n_steps"]) for row in updates] == [1, 5]
    for row in updates:
        assert bool(row["step_success"]) is True
        assert float(row["xy_delta_abs_sum"]) < 1.0e-10
        assert float(row["z_delta_abs_sum"]) > 0.0
        assert "outer_log_trumpet" in row["mode_alignment"]


def test_curved_1disk_trumpet_descent_audit_reconciles_module_deltas() -> None:
    report = run_curved_1disk_trumpet_descent_audit(epsilons=(1.0e-5,))

    for mode in report["modes"]:
        alignment = mode["gradient_alignment"]
        assert math.isfinite(float(alignment["raw_dot_mode"]))
        assert math.isfinite(float(alignment["projected_dot_mode"]))
        enforcement = mode["enforcement_probe"]
        assert float(enforcement["z_before_abs_sum"]) >= 0.0
        assert float(enforcement["xy_after_abs_sum"]) < 1.0e-10

        for case in mode["finite_difference_cases"]:
            assert math.isfinite(float(case["total_delta"]))
            assert math.isfinite(float(case["module_delta_sum"]))
            assert abs(float(case["module_residual"])) < 1.0e-10
            assert case["top_module_deltas"]
