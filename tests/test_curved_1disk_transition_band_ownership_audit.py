import math

from tools.diagnostics.curved_1disk_transition_band_ownership_audit import (
    ALLOWED_CLASSIFICATIONS,
    THETA_CANDIDATES,
    run_curved_1disk_transition_band_ownership_audit,
)


def test_transition_band_ownership_audit_reports_required_schema() -> None:
    report = run_curved_1disk_transition_band_ownership_audit()

    assert report["diagnosis"]["classification"] in ALLOWED_CLASSIFICATIONS
    assert report["diagnosis"]["no_energy_rescaling"] is True
    recommendation = report["diagnosis"]["recommendation"].lower()
    assert "rescale" not in recommendation
    assert "hidden weights" not in recommendation
    assert "tune coefficients" not in recommendation

    band = report["transition_band"]
    assert int(band["row_count"]) > 0
    assert "outer_support" in set(band["row_regions"])
    assert report["region_ownership"]["modules"]
    assert report["region_ownership"]["totals"]["gradient_transition_fraction"] >= 0.0


def test_transition_band_ownership_audit_reconciles_gradients_and_theta_rows() -> None:
    report = run_curved_1disk_transition_band_ownership_audit()

    residual = float(
        report["module_gradient_reconciliation"]["sum_projected_minus_full_norm"]
    )
    assert residual < 1.0e-8

    for row in report["region_ownership"]["modules"]:
        assert math.isfinite(float(row["projected_gradient_norm_total"]))
        assert math.isfinite(float(row["projected_gradient_norm_transition_band"]))
        assert math.isfinite(float(row["energy_total"]))
        assert math.isfinite(float(row["energy_transition_band"]))

    theta_rows = report["theta_candidate_ordering"]
    assert [float(row["theta_B"]) for row in theta_rows] == list(THETA_CANDIDATES)
    assert sum(bool(row["selected_by_total_energy"]) for row in theta_rows) == 1
    assert (
        sum(
            bool(row["selected_without_transition_band_attributed"])
            for row in theta_rows
        )
        == 1
    )


def test_transition_band_regularization_reduces_gradient_dominance_and_orders_theta() -> (
    None
):
    report = run_curved_1disk_transition_band_ownership_audit()

    totals = report["region_ownership"]["totals"]
    assert float(totals["gradient_transition_fraction"]) < 0.95
    assert (
        report["diagnosis"]["classification"]
        != "theta_ordering_depends_on_support_energy"
    )

    theta_rows = report["theta_candidate_ordering"]
    selected = [
        float(row["theta_B"]) for row in theta_rows if row["selected_by_total_energy"]
    ]
    assert selected and selected[0] > 0.12
