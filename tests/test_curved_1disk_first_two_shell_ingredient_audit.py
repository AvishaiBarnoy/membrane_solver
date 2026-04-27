import pytest

from tools.diagnostics.curved_1disk_first_two_shell_ingredient_audit import (
    run_curved_1disk_first_two_shell_ingredient_audit,
)


@pytest.mark.benchmark
@pytest.mark.slow
def test_curved_1disk_first_two_shell_ingredient_audit_reports_required_sections() -> (
    None
):
    """Diagnostic audit emits the required first-two-shell sections."""
    report = run_curved_1disk_first_two_shell_ingredient_audit()

    assert report["case"]["theta_B"] == pytest.approx(0.1845693593, abs=1.0e-12)
    target_shells = report["shell_selection"]["target_shell_radii"]
    assert len(target_shells) == 2
    assert target_shells == sorted(target_shells)

    assert len(report["shellwise_comparison"]) == 2
    for row in report["shellwise_comparison"]:
        assert row["shell_radius"] in target_shells
        assert "in" in row
        assert "out" in row

    for key in (
        "rowwise_ingredient_audit",
        "trianglewise_ingredient_audit",
        "stencil_membership_audit",
        "normalization_audit",
        "first_departure",
        "diagnosis",
    ):
        assert key in report

    for shell in target_shells:
        shell_key = str(float(shell))
        assert shell_key in report["rowwise_ingredient_audit"]
        assert shell_key in report["trianglewise_ingredient_audit"]
        assert shell_key in report["stencil_membership_audit"]
        assert shell_key in report["normalization_audit"]

    assert report["first_departure"]["departure_level"] in {
        "tilt field departure",
        "divergence/shape-term departure",
        "support/stencil departure",
        "normalization/area-weight departure",
        "combined local expression departure",
    }
    assert report["diagnosis"]["call"] == report["first_departure"]["departure_level"]
    assert isinstance(report["diagnosis"]["recommended_next_stream"], str)
    assert report["diagnosis"]["recommended_next_stream"]
