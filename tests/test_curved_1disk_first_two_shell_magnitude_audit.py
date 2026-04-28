import pytest

from tools.diagnostics.curved_1disk_first_two_shell_magnitude_audit import (
    run_curved_1disk_first_two_shell_magnitude_audit,
)


@pytest.mark.benchmark
@pytest.mark.slow
def test_curved_1disk_first_two_shell_magnitude_audit_reports_target_shells() -> None:
    """The magnitude audit should isolate exactly the two target outer shells."""
    report = run_curved_1disk_first_two_shell_magnitude_audit()

    assert report["case"]["theta_B"] == pytest.approx(0.1845693593, abs=1.0e-12)
    assert report["shell_selection"]["target_shell_radii"] == pytest.approx(
        [0.519902, 0.524333], abs=1.0e-6
    )
    assert len(report["shellwise_comparison"]) == 2
    assert set(report["trianglewise_ingredient_audit"].keys()) == {
        "0.519902",
        "0.524333",
    }
    assert set(report["rowwise_ingredient_audit"].keys()) == {"0.519902", "0.524333"}
    assert report["first_material_magnitude_departure"]["call"] in {
        "radial_tilt_input",
        "corner_divergence_stencil_input",
        "div_raw",
        "div_eval",
        "geometric_prefactor",
        "combined_term",
        "local_contribution",
    }
