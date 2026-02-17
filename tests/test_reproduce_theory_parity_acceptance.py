import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "fixtures" / "theory_parity_baseline.yaml"
SCRIPT = ROOT / "tools" / "reproduce_theory_parity.py"


def _iter_leaf_scalars(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_leaf_scalars(value, path)
        return
    yield prefix, float(obj)


def _get_path(dct: dict[str, Any], path: str) -> float:
    cur: Any = dct
    for key in path.split("."):
        cur = cur[key]
    return float(cur)


@pytest.mark.acceptance
def test_reproduce_theory_parity_matches_yaml_baseline_with_tolerances(
    tmp_path,
) -> None:
    out_yaml = tmp_path / "theory_parity_report.yaml"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--out",
        str(out_yaml),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    baseline = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))

    assert report["meta"]["fixture"] == baseline["meta"]["fixture"]
    assert report["meta"]["protocol"] == baseline["meta"]["protocol"]
    assert report["meta"]["format"] == "yaml"

    for path, expected in _iter_leaf_scalars(baseline["metrics"]):
        tol = _get_path(baseline["tolerances"], path)
        actual = _get_path(report["metrics"], path)
        assert actual == pytest.approx(expected, abs=tol), (
            f"{path}: expected {expected} +/- {tol}, got {actual}"
        )
