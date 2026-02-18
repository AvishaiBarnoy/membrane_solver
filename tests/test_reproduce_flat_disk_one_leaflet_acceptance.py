import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "reproduce_flat_disk_one_leaflet.py"
BASELINES = {
    "disabled": ROOT
    / "tests"
    / "fixtures"
    / "flat_disk_one_leaflet_disabled_baseline.yaml",
    "free": ROOT / "tests" / "fixtures" / "flat_disk_one_leaflet_free_baseline.yaml",
}


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
@pytest.mark.parametrize("mode", ["disabled", "free"])
def test_reproduce_flat_disk_one_leaflet_matches_yaml_baseline_with_tolerances(
    tmp_path, mode: str
) -> None:
    baseline_path = BASELINES[mode]
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))

    out_yaml = tmp_path / f"flat_disk_one_leaflet_{mode}.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            str(ROOT / baseline["meta"]["fixture"]),
            "--refine-level",
            str(int(baseline["meta"]["refine_level"])),
            "--outer-mode",
            str(baseline["meta"]["outer_mode"]),
            "--theta-min",
            str(float(baseline["meta"]["theta_min"])),
            "--theta-max",
            str(float(baseline["meta"]["theta_max"])),
            "--theta-count",
            str(int(baseline["meta"]["theta_count"])),
            "--output",
            str(out_yaml),
        ],
        check=True,
        cwd=str(ROOT),
    )
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))

    assert report["meta"]["fixture"] == baseline["meta"]["fixture"]
    assert int(report["meta"]["refine_level"]) == int(baseline["meta"]["refine_level"])
    assert report["meta"]["outer_mode"] == baseline["meta"]["outer_mode"]
    assert report["meta"]["theory_source"] == baseline["meta"]["theory_source"]
    assert float(report["scan"]["theta_min"]) == pytest.approx(
        float(baseline["meta"]["theta_min"]), abs=0.0
    )
    assert float(report["scan"]["theta_max"]) == pytest.approx(
        float(baseline["meta"]["theta_max"]), abs=0.0
    )
    assert int(report["scan"]["theta_count"]) == int(baseline["meta"]["theta_count"])

    for path, expected in _iter_leaf_scalars(baseline["metrics"]):
        tol = _get_path(baseline["tolerances"], path)
        actual = _get_path(report, path)
        assert actual == pytest.approx(expected, abs=tol), (
            f"{mode}:{path}: expected {expected} +/- {tol}, got {actual}"
        )

    assert bool(report["parity"]["meets_factor_2"]) is True
