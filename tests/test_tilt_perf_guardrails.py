from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _perf_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / "tilt_perf_guardrails.py"
    spec = importlib.util.spec_from_file_location("tilt_perf_guardrails", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_resolve_cases_and_unknown() -> None:
    perf = _perf_module()
    assert perf._resolve_cases("tilt_relax_nested") == [
        ("tilt_relax_nested", "benchmark_tilt_relaxation")
    ]
    with pytest.raises(ValueError, match="Unknown case"):
        perf._resolve_cases("missing")


def test_compare_marks_regression() -> None:
    perf = _perf_module()
    rows = perf._compare(
        [{"name": "tilt_relax_nested", "median_seconds": 1.2}],
        {"cases": [{"name": "tilt_relax_nested", "median_seconds": 1.0}]},
        max_regression_percent=5.0,
    )
    assert rows[0]["regressed"] is True


def test_main_writes_report(tmp_path: Path) -> None:
    perf = _perf_module()
    perf.CASE_REGISTRY = {"fake": "fake_mod"}
    perf._run_case = lambda name, module, warmups, runs: {
        "name": name,
        "module": module,
        "warmups": warmups,
        "runs": runs,
        "samples_seconds": [0.1, 0.2],
        "min_seconds": 0.1,
        "median_seconds": 0.15,
        "mean_seconds": 0.15,
        "p95_seconds": 0.195,
        "max_seconds": 0.2,
        "stdev_seconds": 0.05,
    }

    out = tmp_path / "report.json"
    rc = perf.main(["--cases", "fake", "--runs", "2", "--output-json", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["config"]["cases"] == ["fake"]
    assert data["cases"][0]["median_seconds"] == 0.15
