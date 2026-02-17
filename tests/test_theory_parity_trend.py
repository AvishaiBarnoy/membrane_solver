import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "theory_parity_trend.py"
TARGETS = ROOT / "tests" / "fixtures" / "theory_parity_targets.yaml"


def test_compute_ratio_trend_summary_counts() -> None:
    from tools.theory_parity_trend import compute_ratio_trend

    report = {
        "meta": {"fixture": "x.yaml", "protocol": ["g12"]},
        "metrics": {
            "theory": {
                "ratios": {
                    "theta_ratio": 2.1,
                    "elastic_ratio": 1.5,
                }
            }
        },
    }
    targets = {
        "targets": {
            "ratios": {
                "theta_ratio": {"expected": 2.0, "abs_tol": 0.2},
                "elastic_ratio": {"expected": 2.0, "abs_tol": 0.3},
            }
        }
    }
    trend = compute_ratio_trend(report=report, targets=targets)
    assert trend["summary"]["ratio_count"] == 2
    assert trend["summary"]["within_tolerance_count"] == 1
    assert trend["summary"]["all_within_tolerance"] is False
    assert trend["ratios"]["theta_ratio"]["within_tolerance"] is True
    assert trend["ratios"]["elastic_ratio"]["within_tolerance"] is False


@pytest.mark.acceptance
def test_theory_parity_trend_script_writes_yaml_artifact(tmp_path) -> None:
    report_out = tmp_path / "theory_parity_report.yaml"
    trend_out = tmp_path / "theory_parity_trend.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--report-out",
            str(report_out),
            "--out",
            str(trend_out),
            "--targets",
            str(TARGETS),
        ],
        check=True,
        cwd=str(ROOT),
    )
    assert report_out.exists()
    assert trend_out.exists()

    trend = yaml.safe_load(trend_out.read_text(encoding="utf-8"))
    assert trend["meta"]["format"] == "yaml"
    assert trend["summary"]["ratio_count"] == 4
    assert trend["summary"]["all_within_tolerance"] is True
