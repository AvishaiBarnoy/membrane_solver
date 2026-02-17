import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "audit_theory_scaling.py"


@pytest.mark.acceptance
def test_theory_scaling_audit_reports_expected_monotonic_trends(tmp_path) -> None:
    out_yaml = tmp_path / "theory_scaling_audit.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out",
            str(out_yaml),
            "--drive-values",
            "3.8",
            "4.286",
            "--kt-scales",
            "0.9",
            "1.1",
        ],
        check=True,
        cwd=str(ROOT),
    )
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert report["meta"]["format"] == "yaml"
    assert bool(report["summary"]["all_pass"]) is True
    assert int(report["summary"]["check_count"]) == 4
    assert len(report["sweeps"]["drive"]) == 2
    assert len(report["sweeps"]["tilt_modulus_scale"]) == 2
