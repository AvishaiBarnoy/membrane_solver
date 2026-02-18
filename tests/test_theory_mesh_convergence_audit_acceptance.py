import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "audit_theory_mesh_convergence.py"


@pytest.mark.acceptance
def test_theory_mesh_convergence_audit_writes_yaml_and_passes(tmp_path) -> None:
    out_yaml = tmp_path / "theory_mesh_convergence_audit.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out",
            str(out_yaml),
            "--refine-levels",
            "0",
            "1",
            "--ratio-drift-max",
            "0.25",
            "--energy-drift-max",
            "0.50",
        ],
        check=True,
        cwd=str(ROOT),
    )
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert report["meta"]["format"] == "yaml"
    assert isinstance(report["summary"]["all_pass"], bool)
    assert len(report["levels"]) == 2
    assert int(report["summary"]["check_count"]) == 4
