import pytest

from tools.diagnostics.curved_1disk_first_two_shell_magnitude_audit import (
    run_curved_1disk_first_two_shell_magnitude_audit,
)


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.exhaustive
def test_curved_1disk_first_two_shell_magnitude_audit_reports_target_shells() -> None:
    """The magnitude audit should isolate exactly the two target outer shells."""
    report = run_curved_1disk_first_two_shell_magnitude_audit()

    assert report["case"]["theta_B"] == pytest.approx(0.1845693593, abs=1.0e-12)
    radii = report["shell_selection"]["target_shell_radii"]
    assert len(radii) == 2
    assert float(radii[0]) < float(radii[1])
    assert len(report["shellwise_comparison"]) == 2
    triangle_keys = set(report["trianglewise_ingredient_audit"])
    row_keys = set(report["rowwise_ingredient_audit"])
    assert len(triangle_keys) == 2
    assert row_keys == triangle_keys
    assert report["first_material_magnitude_departure"]["call"] in {
        "radial_tilt_input",
        "corner_divergence_stencil_input",
        "div_raw",
        "div_eval",
        "geometric_prefactor",
        "combined_term",
        "local_contribution",
    }
