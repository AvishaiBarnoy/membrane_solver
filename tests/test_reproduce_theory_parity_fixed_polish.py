import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "reproduce_theory_parity.py"


@pytest.mark.acceptance
def test_reproduce_theory_parity_fixed_polish_keeps_finite_and_sign_relations(
    tmp_path,
) -> None:
    out_yaml = tmp_path / "theory_parity_report_polish2.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixed-polish-steps",
            "2",
            "--out",
            str(out_yaml),
        ],
        check=True,
        cwd=str(ROOT),
    )

    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert int(report["meta"]["fixed_polish_steps"]) == 2

    reduced = report["metrics"]["reduced_terms"]
    ratios = report["metrics"]["theory"]["ratios"]
    assert float(reduced["contact_measured"]) < 0.0
    assert float(reduced["elastic_measured"]) > 0.0
    assert float(reduced["total_measured"]) < 0.0

    vals = [
        float(report["metrics"]["final_energy"]),
        float(reduced["elastic_measured"]),
        float(reduced["contact_measured"]),
        float(reduced["total_measured"]),
    ]
    vals.extend(float(v) for v in ratios.values())
    assert all(math.isfinite(v) for v in vals)
