import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "reproduce_flat_disk_one_leaflet.py"
BASELINES = {
    "legacy_disabled": ROOT
    / "tests"
    / "fixtures"
    / "flat_disk_one_leaflet_disabled_baseline.yaml",
    "legacy_free": ROOT
    / "tests"
    / "fixtures"
    / "flat_disk_one_leaflet_free_baseline.yaml",
    "kh_physical_disabled": ROOT
    / "tests"
    / "fixtures"
    / "flat_disk_one_leaflet_kh_physical_disabled_baseline.yaml",
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
@pytest.mark.parametrize(
    "case_name", ["legacy_disabled", "legacy_free", "kh_physical_disabled"]
)
def test_reproduce_flat_disk_one_leaflet_matches_yaml_baseline_with_tolerances(
    tmp_path, case_name: str
) -> None:
    baseline_path = BASELINES[case_name]
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))

    meta = baseline["meta"]
    out_yaml = tmp_path / f"flat_disk_one_leaflet_{case_name}.yaml"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--fixture",
        str(ROOT / meta["fixture"]),
        "--refine-level",
        str(int(meta["refine_level"])),
        "--outer-mode",
        str(meta["outer_mode"]),
        "--output",
        str(out_yaml),
    ]
    if "smoothness_model" in meta:
        cmd.extend(["--smoothness-model", str(meta["smoothness_model"])])
    if "parameterization" in meta:
        cmd.extend(["--parameterization", str(meta["parameterization"])])
    if "optimize_preset" in meta:
        cmd.extend(["--optimize-preset", str(meta["optimize_preset"])])
    if "tilt_mass_mode_in" in meta:
        cmd.extend(["--tilt-mass-mode-in", str(meta["tilt_mass_mode_in"])])
    if "splay_modulus_scale_in" in meta:
        cmd.extend(
            ["--splay-modulus-scale-in", str(float(meta["splay_modulus_scale_in"]))]
        )
    for key in (
        "kappa_physical",
        "kappa_t_physical",
        "length_scale_nm",
        "radius_nm",
        "drive_physical",
    ):
        if key in meta:
            cmd.extend([f"--{key.replace('_', '-')}", str(float(meta[key]))])

    theta_mode = str(meta.get("theta_mode", "scan"))
    cmd.extend(["--theta-mode", theta_mode])
    if theta_mode == "scan":
        cmd.extend(
            [
                "--theta-min",
                str(float(meta["theta_min"])),
                "--theta-max",
                str(float(meta["theta_max"])),
                "--theta-count",
                str(int(meta["theta_count"])),
            ]
        )
    else:
        for key in (
            "theta_initial",
            "theta_optimize_steps",
            "theta_optimize_every",
            "theta_optimize_delta",
            "theta_optimize_inner_steps",
        ):
            if key in meta:
                cmd.extend([f"--{key.replace('_', '-')}", str(meta[key])])

    subprocess.run(cmd, check=True, cwd=str(ROOT))
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))

    assert report["meta"]["fixture"] == meta["fixture"]
    assert int(report["meta"]["refine_level"]) == int(meta["refine_level"])
    assert report["meta"]["outer_mode"] == meta["outer_mode"]
    assert report["meta"]["theory_source"] == meta["theory_source"]
    assert report["parity"]["lane"] == str(meta.get("parameterization", "legacy"))
    if str(meta.get("parameterization", "legacy")) == "kh_physical":
        assert report["meta"]["theory_model"] == "kh_physical_strict_kh"
    else:
        assert report["meta"]["theory_model"] == "legacy_scalar_reduced"
    if theta_mode == "scan":
        assert float(report["scan"]["theta_min"]) == pytest.approx(
            float(meta["theta_min"]), abs=0.0
        )
        assert float(report["scan"]["theta_max"]) == pytest.approx(
            float(meta["theta_max"]), abs=0.0
        )
        assert int(report["scan"]["theta_count"]) == int(meta["theta_count"])
    else:
        assert report["scan"] is None
        assert report["optimize"] is not None

    for path, expected in _iter_leaf_scalars(baseline["metrics"]):
        tol = _get_path(baseline["tolerances"], path)
        actual = _get_path(report, path)
        assert actual == pytest.approx(expected, abs=tol), (
            f"{case_name}:{path}: expected {expected} +/- {tol}, got {actual}"
        )

    expect_meets_factor_2 = bool(meta.get("expect_meets_factor_2", True))
    assert bool(report["parity"]["meets_factor_2"]) is expect_meets_factor_2
