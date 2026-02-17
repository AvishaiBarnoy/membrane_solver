import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ROOT / "tests" / "fixtures" / "theory_parity_targets.yaml"
SCRIPT = ROOT / "tools" / "reproduce_theory_parity.py"


def _get_path(dct: dict[str, Any], path: str) -> Any:
    cur: Any = dct
    for key in path.split("."):
        cur = cur[key]
    return cur


@pytest.mark.acceptance
def test_reproduce_theory_parity_matches_tex_targets_with_tolerances(tmp_path) -> None:
    out_yaml = tmp_path / "theory_parity_report.yaml"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out_yaml)],
        check=True,
        cwd=str(ROOT),
    )

    targets = yaml.safe_load(TARGETS.read_text(encoding="utf-8"))
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))

    assert report["meta"]["fixture"] == targets["meta"]["fixture"]
    assert report["meta"]["protocol"] == targets["meta"]["protocol"]
    assert report["meta"]["format"] == "yaml"

    ratios = targets["targets"]["ratios"]
    for name, cfg in ratios.items():
        actual = float(report["metrics"]["theory"]["ratios"][name])
        expected = float(cfg["expected"])
        abs_tol = float(cfg["abs_tol"])
        assert actual == pytest.approx(expected, abs=abs_tol), (
            f"{name}: expected {expected} +/- {abs_tol}, got {actual}"
        )

    rel = targets["targets"]["relations"]
    reduced = report["metrics"]["reduced_terms"]
    if bool(rel.get("contact_measured_negative", False)):
        assert float(reduced["contact_measured"]) < 0.0
    if bool(rel.get("elastic_measured_positive", False)):
        assert float(reduced["elastic_measured"]) > 0.0
    if bool(rel.get("total_measured_negative", False)):
        assert float(reduced["total_measured"]) < 0.0
