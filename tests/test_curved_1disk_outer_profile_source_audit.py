import math

from tools.diagnostics.curved_1disk_outer_profile_source_audit import (
    ALLOWED_CLASSIFICATIONS,
    SIGN_CONVENTION_CLASSIFICATIONS,
    run_curved_1disk_outer_profile_source_audit,
)


def test_curved_1disk_outer_profile_source_audit_reports_schema() -> None:
    report = run_curved_1disk_outer_profile_source_audit(include_selected=False)

    diagnosis = report["diagnosis"]
    assert diagnosis["classification"] in ALLOWED_CLASSIFICATIONS
    assert (
        diagnosis["sign_convention_classification"] in SIGN_CONVENTION_CLASSIFICATIONS
    )
    assert diagnosis["no_energy_rescaling"] is True
    assert "Feature Contract" in diagnosis["summary"]
    text = diagnosis["recommended_next_stream"]
    assert "Do not rescale energy" in text
    assert "hidden weights" in text
    assert "tune to the theory curve" in text

    assert len(report["cases"]) == 1
    case = report["cases"][0]
    assert case["label"] == "fixed_theory_theta"
    assert case["shell_traces"]
    assert case["first_collapse_stage"]
    assert case["module_tilt_gradient_probe"]
    assert case["perturbation_probes"]
    assert case["profile_fit_controls"]


def test_curved_1disk_outer_profile_source_shell_traces_have_channels() -> None:
    report = run_curved_1disk_outer_profile_source_audit(include_selected=False)
    traces = report["cases"][0]["shell_traces"]
    labels = {trace["label"] for trace in traces}

    assert {
        "configured",
        "after_geometric_enforcement",
        "after_tilt_relaxation",
        "after_shape_minimize",
        "after_tangent_projection",
    } <= labels
    for trace in traces:
        assert trace["shells"]
        for row in trace["shells"]:
            assert {
                "theta_in_median",
                "theta_out_median",
                "theta_shared_median",
                "z_median",
                "curvature_median",
                "leaflet_gap_median",
                "symmetric_sum_abs",
                "antisymmetric_gap_abs",
                "windows",
            } <= set(row)
            assert math.isfinite(float(row["theta_in_median"]))
            assert math.isfinite(float(row["theta_out_median"]))
            assert math.isfinite(float(row["theta_shared_median"]))
            assert math.isfinite(float(row["z_median"]))
            assert math.isfinite(float(row["curvature_median"]))


def test_curved_1disk_outer_profile_source_perturbations_reconcile() -> None:
    report = run_curved_1disk_outer_profile_source_audit(include_selected=False)
    probes = report["cases"][0]["perturbation_probes"]
    names = {row["name"] for row in probes}

    assert {"symmetric_leaflet", "antisymmetric_leaflet", "shape_log"} == names
    for row in probes:
        assert math.isfinite(float(row["total_delta"]))
        assert math.isfinite(float(row["module_delta_sum"]))
        assert abs(float(row["module_residual"])) < 1.0e-10
        assert row["top_module_deltas"]


def test_curved_1disk_outer_profile_source_profile_fit_controls() -> None:
    report = run_curved_1disk_outer_profile_source_audit(include_selected=False)
    controls = report["cases"][0]["profile_fit_controls"]
    channels = {row["channel"] for row in controls["k1_by_channel"]}

    assert {
        "theta_in",
        "theta_out",
        "shared_signed",
        "shared_abs",
        "theta_outer_common_physical",
    } == channels
    for row in controls["k1_by_channel"]:
        assert int(row["count"]) > 0
        assert math.isfinite(float(row["lambda_fit"]))
        assert math.isfinite(float(row["rel_rmse"]))
    assert int(controls["log_all"]["count"]) > 0
    assert math.isfinite(float(controls["log_all"]["slope_ratio"]))
    assert int(controls["curvature_filtered_shell_count"]) >= 0
    primary = controls["primary_physical_common_k1"]
    assert primary["channel"] == "theta_outer_common_physical"
    assert int(primary["count"]) > 0
    assert math.isfinite(float(primary["lambda_fit"]))
    assert math.isfinite(float(primary["rel_rmse"]))
    comparison = controls["theory_comparison"]
    for key in (
        "physical_common_lambda_fit",
        "physical_common_rel_rmse",
        "physical_common_rim_amplitude",
        "expected_lambda",
        "rim_physical_theta_amplitude",
        "theta_B_half",
        "rim_physical_theta_amplitude_over_half_theta_B",
        "measured_log_height_slope",
        "expected_log_height_phi_star",
        "expected_log_height_slope",
        "log_height_slope_ratio",
    ):
        assert math.isfinite(float(comparison[key]))


def test_curved_1disk_outer_profile_source_sign_convention_probe() -> None:
    report = run_curved_1disk_outer_profile_source_audit(include_selected=False)
    probe = report["cases"][0]["profile_fit_controls"]["leaflet_sign_convention_probe"]
    fits = {row["name"]: row for row in probe["fits"]}

    assert probe["classification"] in {
        "diagnostic_leaflet_sign_convention_mismatch",
        "runtime_relaxation_drives_antisymmetric_state",
        "inconclusive",
    }
    assert {
        "theta_common_raw",
        "theta_antisym_raw",
        "theta_common_flip",
        "theta_antisym_flip",
    } == set(fits)
    for row in fits.values():
        assert int(row["count"]) > 0
        assert math.isfinite(float(row["lambda_fit"]))
        assert math.isfinite(float(row["rel_rmse"]))
        assert math.isfinite(float(row["rim_amplitude"]))
        assert row["amplitude_sign"] in {"positive", "negative"}
    assert math.isfinite(float(probe["log_height_slope_ratio"]))
    assert "recommendation" in probe

    if probe["good_k1_profile_location"] == "flipped_common_mode":
        assert probe["classification"] == "diagnostic_leaflet_sign_convention_mismatch"
        assert "diagnostic leaflet sign convention" in probe["recommendation"]
    if probe["good_k1_profile_location"] == "raw_antisymmetric_physical_mode":
        assert (
            probe["classification"] == "runtime_relaxation_drives_antisymmetric_state"
        )
        assert "runtime leaflet coupling" in probe["recommendation"]


def test_curved_1disk_outer_profile_source_classifies_k1_ok_log_suppressed() -> None:
    report = run_curved_1disk_outer_profile_source_audit(include_selected=False)
    diagnosis = report["diagnosis"]
    controls = report["cases"][0]["profile_fit_controls"]
    primary = controls["primary_physical_common_k1"]
    comparison = controls["theory_comparison"]

    assert diagnosis["classification"] == "outer_tilt_k1_ok_but_log_shape_suppressed"
    assert float(primary["rel_rmse"]) < 0.10
    assert abs(float(primary["lambda_ratio"]) - 1.0) <= 0.40
    assert abs(float(comparison["log_height_slope_ratio"])) < 0.25
    assert diagnosis["sign_convention_classification"] == (
        "diagnostic_leaflet_sign_convention_mismatch"
    )
