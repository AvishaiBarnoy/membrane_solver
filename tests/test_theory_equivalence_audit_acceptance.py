import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = Path(__file__).resolve().parent.parent
REPRO_SCRIPT = ROOT / "tools" / "reproduce_theory_parity.py"
AUDIT_SCRIPT = ROOT / "tools" / "audit_theory_equivalence.py"


@pytest.mark.acceptance
def test_theory_equivalence_audit_fixed_lane_passes(tmp_path) -> None:
    report_out = tmp_path / "theory_parity_report.yaml"
    audit_out = tmp_path / "theory_equivalence_audit.yaml"
    subprocess.run(
        [sys.executable, str(REPRO_SCRIPT), "--out", str(report_out)],
        check=True,
        cwd=str(ROOT),
    )
    subprocess.run(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            "--report-out",
            str(report_out),
            "--out",
            str(audit_out),
        ],
        check=True,
        cwd=str(ROOT),
    )
    audit = yaml.safe_load(audit_out.read_text(encoding="utf-8"))
    assert audit["meta"]["format"] == "yaml"
    assert bool(audit["summary"]["all_pass"]) is True
    assert int(audit["summary"]["check_count"]) == 5
    assert len(audit["radius_sweep"]) == 3
