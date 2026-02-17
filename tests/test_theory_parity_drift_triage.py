import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.theory_parity_drift_triage import format_triage, top_offenders

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "theory_parity_drift_triage.py"


def _trend() -> dict:
    return {
        "summary": {
            "all_within_tolerance": False,
            "within_tolerance_count": 1,
            "ratio_count": 3,
        },
        "ratios": {
            "total_ratio": {
                "actual": 1.0,
                "expected": 2.0,
                "abs_tol": 0.3,
                "abs_delta": 1.0,
                "within_tolerance": False,
            },
            "elastic_ratio": {
                "actual": 1.7,
                "expected": 2.0,
                "abs_tol": 0.3,
                "abs_delta": 0.3,
                "within_tolerance": True,
            },
            "theta_ratio": {
                "actual": 0.8,
                "expected": 2.0,
                "abs_tol": 0.3,
                "abs_delta": 1.2,
                "within_tolerance": False,
            },
        },
    }


def test_top_offenders_sorted_by_abs_delta_desc() -> None:
    rows = top_offenders(_trend(), top=2)
    assert [name for name, _ in rows] == ["theta_ratio", "total_ratio"]


def test_format_triage_contains_summary_and_rows() -> None:
    out = format_triage(_trend(), top=2)
    assert "summary: all_within_tolerance=False within=1 total=3" in out
    assert "top_offenders(top=2):" in out
    assert "- theta_ratio:" in out
    assert "- total_ratio:" in out


@pytest.mark.acceptance
def test_drift_triage_script_reads_yaml_and_prints_top_rows(tmp_path) -> None:
    trend_path = tmp_path / "trend.yaml"
    trend_path.write_text(yaml.safe_dump(_trend(), sort_keys=False), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--trend", str(trend_path), "--top", "2"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "top_offenders(top=2):" in proc.stdout
    assert "- theta_ratio:" in proc.stdout
    assert "- total_ratio:" in proc.stdout
