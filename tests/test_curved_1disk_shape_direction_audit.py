import math

import pytest

from tools.diagnostics.curved_1disk_shape_direction_audit import (
    ALLOWED_CLASSIFICATIONS,
    run_curved_1disk_shape_direction_audit,
)


def test_curved_1disk_shape_direction_audit_reports_required_schema() -> None:
    report = run_curved_1disk_shape_direction_audit(horizons=(1,))

    assert report["diagnosis"]["classification"] in ALLOWED_CLASSIFICATIONS
    assert report["diagnosis"]["no_energy_rescaling"] is True
    assert "Feature Contract" in report["diagnosis"]["summary"]
    assert "do not rescale" in report["diagnosis"]["recommended_next_stream"]

    directions = {row["name"]: row for row in report["direction_summaries"]}
    assert {
        "outer_log_trumpet",
        "projected_gradient_descent",
        "log_residual_gradient",
        "near_support_gradient",
        "far_field_gradient",
        "high_frequency_gradient",
        "area_weighted_gradient_probe",
        "shell_normalized_gradient_probe",
        "support_suppressed_gradient_probe",
    } <= set(directions)
    assert directions["outer_log_trumpet"]["norm"] == pytest.approx(1.0)
    assert directions["outer_log_trumpet"]["nonzero_rows"] > 0

    log_probe = next(
        row
        for row in report["directional_probes"]
        if row["name"] == "outer_log_trumpet" and not row["relax_tilts"]
    )
    assert bool(log_probe["accepted_by_decrease"]) is True
    assert float(log_probe["total_delta"]) < 0.0


def test_curved_1disk_shape_direction_audit_reconciles_and_replays_updates() -> None:
    report = run_curved_1disk_shape_direction_audit(horizons=(1,))

    for probe in report["directional_probes"]:
        assert math.isfinite(float(probe["total_delta"]))
        assert math.isfinite(float(probe["module_delta_sum"]))
        assert abs(float(probe["module_residual"])) < 1.0e-10
        assert probe["top_module_deltas"]
        assert float(probe["direction_norm"]) == pytest.approx(0.0) or float(
            probe["direction_norm"]
        ) == pytest.approx(1.0)

    replay = report["accepted_update_replay"]
    assert len(replay) == 1
    row = replay[0]
    assert int(row["n_steps"]) == 1
    assert bool(row["step_success"]) is True
    assert float(row["xy_delta_abs_sum"]) < 1.0e-10
    assert float(row["z_delta_abs_sum"]) > 0.0
    assert "outer_log_trumpet" in row["mode_alignment"]
    assert row["z_delta_by_shell"]
