import pytest

from tools.diagnostics.curved_1disk_shell2_tiltout_source_audit import (
    run_curved_1disk_shell2_tiltout_source_audit,
)


@pytest.mark.benchmark
@pytest.mark.slow
def test_curved_1disk_shell2_tiltout_source_audit_reports_upstream_call() -> None:
    """The shell-2 source audit should report the first upstream departure call."""
    report = run_curved_1disk_shell2_tiltout_source_audit()

    assert report["case"]["theta_B"] == pytest.approx(0.1845693593, abs=1.0e-12)
    assert report["shell_selection"]["shell1_radius"] == pytest.approx(
        0.519902, abs=1.0e-6
    )
    assert report["shell_selection"]["shell2_radius"] == pytest.approx(
        0.524333, abs=1.0e-6
    )
    assert report["first_upstream_departure"]["call"] in {
        "continuation-rule mismatch",
        "stencil-membership mismatch",
        "neighbor-selection mismatch",
        "leaflet-label / continuation mismatch",
        "another specific upstream field-construction defect",
    }
    assert "shell1_role" in report["source_path_audit"]
    assert "shell2_role" in report["source_path_audit"]
