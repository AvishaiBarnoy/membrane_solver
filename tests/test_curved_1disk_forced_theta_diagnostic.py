import pytest

from tools.diagnostics.curved_1disk_forced_theta_diagnostic import (
    run_curved_1disk_forced_theta_diagnostic,
)


@pytest.mark.benchmark
@pytest.mark.slow
def test_curved_1disk_forced_theta_diagnostic_runs_and_reports_required_sections() -> (
    None
):
    """Benchmark diagnostic: forced-theta curved report emits the required sections."""
    report = run_curved_1disk_forced_theta_diagnostic()

    assert report["forced_theta_values"] == [0.06, 0.12, 0.1845693593]
    assert report["theory"]["R"] == pytest.approx(7.0 / 15.0)
    assert report["theory"]["lambda_theory"] == pytest.approx(15.0)

    assert len(report["cases"]) == 3
    for case, theta in zip(report["cases"], report["forced_theta_values"]):
        assert case["theta_B"] == pytest.approx(theta, abs=1.0e-12)
        assert case["phi_star_forced"] == pytest.approx(0.5 * theta, abs=1.0e-12)
        assert case["outer_k1"]["shell_count"] > 0
        assert case["outer_height_log"]["shell_count"] > 0
        assert case["outer_slope"]["shell_count"] > 0
        assert case["outer_curvature"]["shell_count"] > 0
        assert case["outer_leaflet_mismatch"]["shell_count"] > 0
        assert "outer_elastic_numeric" in case["outer_elastic"]
        assert "outer_elastic_over_theta_B_sq" in case["outer_elastic"]
        assert case["axisymmetry"]["judgment"] in {
            "axisymmetric enough",
            "noticeable azimuthal scatter",
        }

    assert report["outer_elastic_scaling"]["classification"] in {
        "approximately quadratic",
        "subquadratic",
        "superquadratic",
    }
    diagnosis = report["diagnosis"]
    assert diagnosis["dominant_failure"] in {
        "outer shape response",
        "outer tilt response",
        "coupling",
    }
    assert diagnosis["branch_preference_interpretation"] in {
        "the realized coupled energy genuinely favors a different branch",
        "the geometry solver/path is too stiff or constrained",
    }
    assert isinstance(diagnosis["recommended_next_stream"], str)
    assert diagnosis["recommended_next_stream"]
