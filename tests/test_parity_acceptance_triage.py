from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

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
