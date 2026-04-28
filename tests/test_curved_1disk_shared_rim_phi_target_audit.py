import pytest

from tools.diagnostics.curved_1disk_shared_rim_phi_target_audit import (
    run_curved_1disk_shared_rim_phi_target_audit,
)


@pytest.mark.benchmark
@pytest.mark.slow
def test_curved_1disk_shared_rim_phi_target_audit_reports_single_call() -> None:
    """The shared-rim phi-target audit should emit one final diagnosis call."""
    report = run_curved_1disk_shared_rim_phi_target_audit()

    assert report["case"]["theta_B"] == pytest.approx(0.1845693593, abs=1.0e-12)
    assert report["case"]["matching_mode"] == "shared_rim_staggered_v1"
    assert report["theory_reference"]["phi_star_theory"] > 0.0
    assert "secant_geometry" in report["shell_target_construction"]
    assert "phi_construction" in report["shell_target_construction"]
    assert "target_direction" in report["shell_target_construction"]
    assert report["diagnosis"]["call"] in {
        "wrong secant sign",
        "wrong phi target sign",
        "wrong normal/orientation convention",
        "another specific target-construction defect",
        "target direction outward",
    }
