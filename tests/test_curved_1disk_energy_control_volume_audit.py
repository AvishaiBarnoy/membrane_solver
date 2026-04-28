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
            "unattributed_fraction": 0.78,
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
    assert causes[:2] == [
        "inner/outer leaflet elastic imbalance",
        "outer energy attribution mismatch",
    ]
    assert "excess shared-rim/local-shell elastic cost" in causes
    assert "excessive shared-rim support control volume" in causes


@pytest.mark.benchmark
@pytest.mark.slow
def test_curved_1disk_energy_control_volume_audit_actual_selected_theta() -> None:
    """Benchmark diagnostic should emit selected-theta energy/control evidence."""
    report = audit.run_curved_1disk_energy_control_volume_audit((0.12,))

    case = report["cases"][0]
    assert case["theta_B"] == pytest.approx(0.12, abs=1.0e-12)
    assert case["energy_ratios"]["outer_numeric_over_tex"] > 5.0
    assert case["energy_ratios"]["inner_numeric_over_tex"] < 0.25
    assert case["energy_ratios"]["contact_numeric_over_tex"] == pytest.approx(
        1.0, rel=0.02
    )
    assert case["shell_attribution_coverage"]["unattributed_fraction"] > 0.5
    assert case["control_volume"]["ratios"]["rim_control_over_gap_annulus"] > 2.0
    assert case["shell_energy_rows"]
