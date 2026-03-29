import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "fixtures" / "theory_parity_baseline.yaml"
I50_BASELINE = ROOT / "tests" / "fixtures" / "theory_parity_i50_interface_baseline.yaml"
I60_BASELINE = ROOT / "tests" / "fixtures" / "theory_parity_i60_interface_baseline.yaml"
NEAR_EDGE_BASELINE = (
    ROOT / "tests" / "fixtures" / "theory_parity_near_edge_v1_baseline.yaml"
)
PRIMARY_BASELINE = (
    ROOT / "tests" / "fixtures" / "theory_parity_physical_edge_primary_baseline.yaml"
)
DEFAULT_BASELINE = (
    ROOT / "tests" / "fixtures" / "theory_parity_physical_edge_default_baseline.yaml"
)
SCRIPT = ROOT / "tools" / "reproduce_theory_parity.py"
I50_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_theory_parity_i50_interface.yaml"
)
I60_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_theory_parity_i60_interface.yaml"
)
NEAR_EDGE_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_theory_parity_near_edge_v1.yaml"
)
PRIMARY_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_primary.yaml"
)
DEFAULT_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml"
)


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


@pytest.mark.acceptance
def test_reproduce_theory_parity_i50_interface_matches_yaml_baseline_with_tolerances(
    tmp_path,
) -> None:
    out_yaml = tmp_path / "theory_parity_i50_report.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(I50_FIXTURE),
            "--out",
            str(out_yaml),
        ],
        check=True,
        cwd=str(ROOT),
    )

    baseline = yaml.safe_load(I50_BASELINE.read_text(encoding="utf-8"))
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))

    assert report["meta"]["fixture"] == baseline["meta"]["fixture"]
    assert report["meta"]["lane"] == baseline["meta"]["lane"]
    assert report["meta"]["protocol"] == baseline["meta"]["protocol"]
    assert report["meta"]["format"] == "yaml"

    for path, expected in _iter_leaf_scalars(baseline["metrics"]):
        tol = _get_path(baseline["tolerances"], path)
        actual = _get_path(report["metrics"], path)
        assert actual == pytest.approx(expected, abs=tol), (
            f"{path}: expected {expected} +/- {tol}, got {actual}"
        )


@pytest.mark.acceptance
def test_reproduce_theory_parity_i60_interface_matches_yaml_baseline_with_tolerances(
    tmp_path,
) -> None:
    out_yaml = tmp_path / "theory_parity_i60_report.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(I60_FIXTURE),
            "--out",
            str(out_yaml),
        ],
        check=True,
        cwd=str(ROOT),
    )

    baseline = yaml.safe_load(I60_BASELINE.read_text(encoding="utf-8"))
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))

    assert report["meta"]["fixture"] == baseline["meta"]["fixture"]
    assert report["meta"]["lane"] == baseline["meta"]["lane"]
    assert report["meta"]["protocol"] == baseline["meta"]["protocol"]
    assert report["meta"]["format"] == "yaml"

    for path, expected in _iter_leaf_scalars(baseline["metrics"]):
        tol = _get_path(baseline["tolerances"], path)
        actual = _get_path(report["metrics"], path)
        assert actual == pytest.approx(expected, abs=tol), (
            f"{path}: expected {expected} +/- {tol}, got {actual}"
        )


@pytest.mark.acceptance
def test_reproduce_theory_parity_near_edge_v1_matches_yaml_baseline_with_tolerances(
    tmp_path,
) -> None:
    out_yaml = tmp_path / "theory_parity_near_edge_v1_report.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(NEAR_EDGE_FIXTURE),
            "--out",
            str(out_yaml),
        ],
        check=True,
        cwd=str(ROOT),
    )

    baseline = yaml.safe_load(NEAR_EDGE_BASELINE.read_text(encoding="utf-8"))
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))

    assert report["meta"]["fixture"] == baseline["meta"]["fixture"]
    assert report["meta"]["lane"] == baseline["meta"]["lane"]
    assert report["meta"]["protocol"] == baseline["meta"]["protocol"]
    assert report["meta"]["format"] == "yaml"

    for path, expected in _iter_leaf_scalars(baseline["metrics"]):
        tol = _get_path(baseline["tolerances"], path)
        actual = _get_path(report["metrics"], path)
        assert actual == pytest.approx(expected, abs=tol), (
            f"{path}: expected {expected} +/- {tol}, got {actual}"
        )


@pytest.mark.acceptance
def test_reproduce_theory_parity_physical_edge_default_matches_yaml_baseline_with_tolerances(
    tmp_path,
) -> None:
    out_yaml = tmp_path / "theory_parity_physical_edge_default_report.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(DEFAULT_FIXTURE),
            "--out",
            str(out_yaml),
        ],
        check=True,
        cwd=str(ROOT),
    )

    baseline = yaml.safe_load(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))

    assert report["meta"]["fixture"] == baseline["meta"]["fixture"]
    assert report["meta"]["lane"] == baseline["meta"]["lane"]
    assert report["meta"]["protocol"] == baseline["meta"]["protocol"]
    assert report["meta"]["format"] == "yaml"

    for path, expected in _iter_leaf_scalars(baseline["metrics"]):
        tol = _get_path(baseline["tolerances"], path)
        actual = _get_path(report["metrics"], path)
        assert actual == pytest.approx(expected, abs=tol), (
            f"{path}: expected {expected} +/- {tol}, got {actual}"
        )


@pytest.mark.acceptance
def test_reproduce_theory_parity_physical_edge_primary_matches_yaml_baseline_with_tolerances(
    tmp_path,
) -> None:
    out_yaml = tmp_path / "theory_parity_physical_edge_primary_report.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(PRIMARY_FIXTURE),
            "--out",
            str(out_yaml),
        ],
        check=True,
        cwd=str(ROOT),
    )

    baseline = yaml.safe_load(PRIMARY_BASELINE.read_text(encoding="utf-8"))
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))

    assert report["meta"]["fixture"] == baseline["meta"]["fixture"]
    assert report["meta"]["lane"] == baseline["meta"]["lane"]
    assert report["meta"]["protocol"] == baseline["meta"]["protocol"]
    assert report["meta"]["format"] == "yaml"

    for path, expected in _iter_leaf_scalars(baseline["metrics"]):
        tol = _get_path(baseline["tolerances"], path)
        actual = _get_path(report["metrics"], path)
        assert actual == pytest.approx(expected, abs=tol), (
            f"{path}: expected {expected} +/- {tol}, got {actual}"
        )
