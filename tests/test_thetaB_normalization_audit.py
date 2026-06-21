from __future__ import annotations

import pytest

from tools.diagnostics import thetaB_normalization_audit as audit


def _row(theta: float) -> dict:
    elastic = 4.0 * theta * theta + 0.25
    contact = -2.0 * theta
    return {
        "thetaB_value": theta,
        "energy_breakdown": {
            "bending_tilt_in": theta * theta,
            "tilt_in": 3.0 * theta * theta,
            "tilt_thetaB_contact_in": contact,
        },
        "reduced_terms": {
            "elastic_measured": elastic,
            "contact_measured": contact,
            "total_measured": elastic + contact,
        },
    }


def test_summarize_fixed_theta_sweep_fits_quadratic_balance() -> None:
    sweep = {f"{theta:.2f}": _row(theta) for theta in audit.FIXED_THETA_SWEEP_VALUES}

    summary = audit.summarize_fixed_theta_sweep(sweep)

    assert summary["measured"]["elastic_A"] == pytest.approx(4.0)
    assert summary["measured"]["contact_B"] == pytest.approx(2.0)
    assert summary["measured"]["theta_min"] == pytest.approx(0.25)
    assert summary["module_fits"]["bending_tilt_in"]["quadratic"] == pytest.approx(1.0)
    assert summary["module_fits"]["tilt_in"]["quadratic"] == pytest.approx(3.0)
    assert summary["module_fits"]["tilt_thetaB_contact_in"][
        "contact_B"
    ] == pytest.approx(2.0)
    assert summary["classification"] == "insufficient_theory_reference"


def test_classify_balance_reports_contact_scale_high() -> None:
    classification = audit.classify_balance(
        ratios={"contact_B": 1.5, "elastic_A": 1.0, "theta_min": 1.4},
        measured={"total_quadratic": 1.0},
    )

    assert classification == "contact_scale_high"


def test_classify_balance_reports_elastic_response_low() -> None:
    classification = audit.classify_balance(
        ratios={"contact_B": 1.0, "elastic_A": 0.5, "theta_min": 2.0},
        measured={"total_quadratic": 1.0},
    )

    assert classification == "elastic_response_low"
