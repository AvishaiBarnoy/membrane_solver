import pytest

from tools.diagnostics import curved_1disk_energy_control_volume_audit as audit


def _fake_case(theta: float) -> dict[str, object]:
    return {
        "theta_B": float(theta),
        "energy_ratios": {
            "outer_numeric_over_tex": 8.4,
            "inner_numeric_over_tex": 0.03,
            "contact_numeric_over_tex": 1.0,
        },
        "shell_concentration": {
            "support_fraction_of_outer_shell_elastic": 0.18,
            "first_two_fraction_of_outer_shell_elastic": 0.18,
        },
        "shell_attribution_coverage": {
            "unattributed_fraction": 0.0,
        },
        "control_volume": {
            "ratios": {
                "outer_control_over_gap_annulus": 1.1,
            },
        },
    }


def test_curved_1disk_energy_control_volume_audit_ranks_remaining_causes(
    monkeypatch,
) -> None:
    """The report should rank the post-PR2 energy/ownership evidence."""
    monkeypatch.setattr(audit, "_run_case", _fake_case)

    report = audit.run_curved_1disk_energy_control_volume_audit((0.12,))

    assert report["scope"]["diagnosis_only"] is True
    assert report["scope"]["runtime_physics_changed"] is False
    assert report["theta_values"] == [0.12]
    assert report["recommended_next_pr"]["feature_contract_required"] is True

    causes = [row["cause"] for row in report["root_causes_ranked"]]
    assert causes[0] == "inner/outer leaflet elastic imbalance"
    assert "excess shared-rim/local-shell elastic cost" in causes
    assert "excessive shared-rim support control volume" in causes
    assert "outer energy attribution mismatch" in causes


def test_reconciled_energy_split_uses_runtime_modules_and_shell_outer() -> None:
    """Gate A: diagnostic split should reconcile shell outer energy to runtime totals."""
    legacy = {
        "total_numeric": -0.5,
        "inner_elastic_numeric": 0.01,
        "outer_elastic_numeric": 1.25,
        "contact_numeric": -1.5,
    }
    breakdown = {
        "tilt_in": 0.2,
        "tilt_out": 0.1,
        "bending_tilt_in": 0.3,
        "bending_tilt_out": 0.2,
        "tilt_thetaB_contact_in": -1.5,
    }
    shell = {
        "outer_membrane_elastic_total_from_shell_rows": 0.25,
    }

    split, reconciliation = audit._reconciled_runtime_energy_split(  # noqa: SLF001
        legacy_energy_split=legacy,
        runtime_breakdown=breakdown,
        shell_concentration=shell,
    )

    assert split["total_numeric"] == pytest.approx(-0.7)
    assert split["contact_numeric"] == pytest.approx(-1.5)
    assert split["outer_elastic_numeric"] == pytest.approx(0.25)
    assert split["inner_elastic_numeric"] == pytest.approx(0.55)
    assert reconciliation["elastic_residual"] == pytest.approx(0.0, abs=1.0e-12)
    assert reconciliation["legacy_outer_minus_reconciled_outer"] == pytest.approx(1.0)


@pytest.mark.benchmark
@pytest.mark.slow
def test_curved_1disk_energy_control_volume_audit_actual_selected_theta() -> None:
    """Benchmark diagnostic should emit selected-theta energy/control evidence."""
    report = audit.run_curved_1disk_energy_control_volume_audit((0.12,))

    case = report["cases"][0]
    assert case["theta_B"] == pytest.approx(0.12, abs=1.0e-12)
    assert case["energy_ratios"]["outer_numeric_over_tex"] < 3.0
    assert 0.5 < case["energy_ratios"]["inner_numeric_over_tex"] < 3.0
    assert case["energy_ratios"]["contact_numeric_over_tex"] == pytest.approx(
        1.0, rel=0.02
    )
    assert case["shell_attribution_coverage"]["unattributed_fraction"] == pytest.approx(
        0.0, abs=1.0e-9
    )
    assert case["runtime_energy_reconciliation"]["elastic_residual"] == pytest.approx(
        0.0, abs=1.0e-9
    )
    assert case["legacy_numeric_energy_split"]["outer_elastic_numeric"] > 1.0
    assert case["control_volume"]["ratios"]["rim_control_over_gap_annulus"] > 2.0
    assert case["shell_energy_rows"]
