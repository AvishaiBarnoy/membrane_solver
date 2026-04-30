import math

from tools.diagnostics.curved_1disk_rim_inner_tilt_profile_audit import (
    PROFILE_CLASSIFICATIONS,
    RIM_CLASSIFICATIONS,
    run_curved_1disk_rim_inner_tilt_profile_audit,
)


def test_curved_1disk_rim_inner_tilt_profile_audit_reports_schema() -> None:
    report = run_curved_1disk_rim_inner_tilt_profile_audit(include_selected=False)

    diagnosis = report["diagnosis"]
    assert diagnosis["rim_inner_tilt_classification"] in RIM_CLASSIFICATIONS
    assert diagnosis["outer_profile_classification"] in PROFILE_CLASSIFICATIONS
    assert diagnosis["no_energy_rescaling"] is True
    assert "Do not rescale energy" in diagnosis["recommended_next_stream"]
    assert "tune coefficients" in diagnosis["recommended_next_stream"]

    assert len(report["cases"]) == 1
    case = report["cases"][0]
    assert case["label"] == "fixed_theory_theta"
    assert case["selected_theta"] is False
    assert case["tilt_trace"]
    assert case["module_gradient_probe"]
    assert case["outer_profile_probe"]


def test_curved_1disk_rim_inner_tilt_trace_is_finite() -> None:
    report = run_curved_1disk_rim_inner_tilt_profile_audit(include_selected=False)
    traces = report["cases"][0]["tilt_trace"]
    labels = {row["label"] for row in traces}

    assert {
        "configured",
        "after_geometric_enforcement",
        "after_tilt_relaxation",
        "after_rim_tilt_enforcement",
    } <= labels
    for trace in traces:
        assert trace["shells"]
        for region in ("rim", "outer_support"):
            summary = trace[region]
            assert int(summary["row_count"]) >= 0
            assert math.isfinite(float(summary["theta_in_median"]))
            assert math.isfinite(float(summary["theta_out_median"]))
            assert math.isfinite(float(summary["theta_shared_median"]))


def test_curved_1disk_rim_inner_tilt_module_and_energy_reconcile() -> None:
    report = run_curved_1disk_rim_inner_tilt_profile_audit(include_selected=False)
    case = report["cases"][0]

    for module_name in ("bending_tilt_in", "bending_tilt_out"):
        row = case["module_gradient_probe"][module_name]
        assert math.isfinite(float(row["energy"]))
        assert math.isfinite(float(row["shape_grad_norm"]))
        assert math.isfinite(float(row["tilt_grad_norm"]))
        assert set(row["shape_grad_z_abs_by_region"]) == {
            "disk",
            "shared_rim",
            "outer_support",
            "outer_free",
        }
        assert set(row["tilt_grad_abs_by_region"]) == {
            "disk",
            "shared_rim",
            "outer_support",
            "outer_free",
        }

    recon = case["energy_reconciliation"]
    assert math.isfinite(float(recon["total_delta"]))
    assert math.isfinite(float(recon["module_delta_sum"]))
    assert abs(float(recon["module_residual"])) < 1.0e-10
    assert recon["module_deltas"]


def test_curved_1disk_rim_inner_tilt_profile_channels_are_reported() -> None:
    report = run_curved_1disk_rim_inner_tilt_profile_audit(include_selected=False)
    profile = report["cases"][0]["outer_profile_probe"]

    assert profile["outer_k1_shared"]["count"] > 0
    assert profile["outer_height_log"]["count"] > 0
    assert profile["outer_curvature"]["count"] > 0
    assert profile["fit_window_shells"]
    for row in profile["fit_window_shells"]:
        assert {
            "theta_in",
            "theta_out",
            "theta_shared",
            "z",
            "J",
            "leaflet_gap",
        } <= set(row)
        assert math.isfinite(float(row["theta_in"]))
        assert math.isfinite(float(row["theta_out"]))
        assert math.isfinite(float(row["theta_shared"]))
        assert math.isfinite(float(row["z"]))
        assert math.isfinite(float(row["J"]))
