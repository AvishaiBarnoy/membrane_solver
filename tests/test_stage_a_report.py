from __future__ import annotations

from tools.report_stage_a_emergent import build_stage_a_report


def test_stage_a_report_contains_all_cases_and_flat_reference() -> None:
    report = build_stage_a_report()

    assert set(report["cases"]) == {"base", "seeded", "continuation", "refined"}
    assert float(report["flat_reference"]["theta_star"]) > 0.0
    assert float(report["flat_reference"]["total_energy"]) < 0.0

    for metrics in report["cases"].values():
        assert metrics["branch"] in {"emergent_curved", "flat_control"}
        assert "energy_breakdown" in metrics
        assert "commands" in metrics
        assert "fixture" in metrics
