from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from tools.diagnostics import parity_acceptance_triage as triage_mod
from tools.diagnostics.parity_acceptance_triage import run_triage

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "diagnostics" / "parity_acceptance_triage.py"


def test_parity_acceptance_triage_schema_reports_known_cases() -> None:
    triage = run_triage(mode="schema")

    cases = {row["case"] for row in triage["cases"]}
    assert cases == {
        "ghost_shell_direct_interface",
        "generated_family_smoothness",
        "default_free_side_trace_continuation",
        "default_director_profile_parity",
    }
    assertions = triage["assertions"]
    assert len(assertions) == 4
    assert {row["case"] for row in assertions} == cases
    for row in assertions:
        assert row["metric_path"]
        assert row["condition"] in {">", "<", "abs<"}
        assert row["passed"] is None


def test_parity_acceptance_triage_cli_schema_writes_yaml(tmp_path) -> None:
    out_yaml = tmp_path / "triage.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "schema",
            "--out",
            str(out_yaml),
        ],
        check=True,
        cwd=str(ROOT),
    )

    triage = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert triage["meta"]["mode"] == "schema"
    assert len(triage["assertions"]) == 4


def test_fixed_theta_sweep_reports_runtime_energy_breakdown(
    tmp_path, monkeypatch
) -> None:
    fixture = tmp_path / "fixture.yaml"
    fixture.write_text(
        yaml.safe_dump({"global_parameters": {"tilt_thetaB_value": 0.0}}),
        encoding="utf-8",
    )

    def fake_run_report(path: Path) -> dict:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        theta = float(doc["global_parameters"]["tilt_thetaB_value"])
        assert doc["global_parameters"]["tilt_thetaB_optimize"] is False
        return {
            "metrics": {
                "final_energy": theta + 1.0,
                "breakdown": {
                    "bending_tilt_in": theta,
                    "tilt_thetaB_contact_in": -theta,
                },
                "reduced_terms": {
                    "elastic_measured": theta,
                    "contact_measured": -theta,
                    "total_measured": 1.0,
                },
                "diagnostics": {},
            }
        }

    monkeypatch.setattr(triage_mod, "_run_report", fake_run_report)
    monkeypatch.setattr(
        triage_mod,
        "_base_term_summary_for_fixture",
        lambda path, label: {"label": label},
    )

    sweep = triage_mod._run_fixed_theta_sweep(
        base_fixture=fixture, label="case", tmpdir=tmp_path
    )

    row = sweep["0.18"]
    assert row["thetaB_optimization_bypassed"] is True
    assert row["final_energy"] == 1.18
    assert row["energy_breakdown"]["bending_tilt_in"] == 0.18
    assert row["energy_breakdown"]["tilt_thetaB_contact_in"] == -0.18
    assert row["reduced_terms"]["total_measured"] == 1.0
    assert row["base_term_summary"]["label"] == "case_sweep_0.18"
