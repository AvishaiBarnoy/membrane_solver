import pytest

from tools.diagnostics.curved_1disk_shell2_tiltout_audit import (
    run_curved_1disk_shell2_tiltout_audit,
)


@pytest.mark.benchmark
@pytest.mark.slow
def test_curved_1disk_shell2_tiltout_audit_reports_shell2_departure() -> None:
    """The shell-2 tilt_out audit should report the continuation ladder and departure stage."""
    report = run_curved_1disk_shell2_tiltout_audit()

    assert report["case"]["theta_B"] == pytest.approx(0.1845693593, abs=1.0e-12)
    assert report["shell_selection"]["shell1_radius"] > 0.51
    assert (
        report["shell_selection"]["shell2_radius"]
        > report["shell_selection"]["shell1_radius"]
    )
    assert report["shell_selection"]["shell1_row_count"] > 0
    assert report["shell_selection"]["shell2_row_count"] > 0
    assert report["first_material_departure"]["call"] in {
        "theta_out_radial",
        "theta_out_tangential",
    }
    assert set(report["toggle_comparison"].keys()) == {
        "tilt_out_exclude_shared_rim_outer_rows_true",
        "tilt_out_exclude_shared_rim_outer_rows_false",
    }
