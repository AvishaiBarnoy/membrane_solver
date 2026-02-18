import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data
from tools.reproduce_flat_disk_one_leaflet import (
    DEFAULT_FIXTURE,
    run_flat_disk_one_leaflet_benchmark,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "reproduce_flat_disk_one_leaflet.py"


@lru_cache(maxsize=4)
def _report_for_mode(mode: str) -> dict:
    return run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode=mode,
        theta_min=0.0,
        theta_max=0.0014,
        theta_count=8,
    )


@pytest.mark.acceptance
@pytest.mark.e2e
def test_flat_disk_one_leaflet_mesh_parity_outer_disabled_e2e() -> None:
    report = _report_for_mode("disabled")

    assert float(report["parity"]["theta_factor"]) <= 2.0
    assert float(report["parity"]["energy_factor"]) <= 2.0

    profile = report["mesh"]["profile"]
    rim = float(profile["rim_abs_median"])
    outer = float(profile["outer_abs_median"])
    assert rim > 1e-5
    assert outer < 0.7 * rim

    assert float(report["mesh"]["planarity_z_span"]) < 1e-12


@pytest.mark.acceptance
@pytest.mark.e2e
def test_flat_disk_one_leaflet_mesh_parity_outer_free_e2e() -> None:
    report = _report_for_mode("free")

    assert float(report["parity"]["theta_factor"]) <= 2.0
    assert float(report["parity"]["energy_factor"]) <= 2.0
    assert float(report["mesh"]["outer_tilt_max_free_rows"]) < 1e-9
    assert float(report["mesh"]["outer_decay_probe_max_before"]) > 1e-5
    assert float(report["mesh"]["outer_decay_probe_max_after"]) < 1e-9
    assert float(report["mesh"]["planarity_z_span"]) < 1e-12


@pytest.mark.regression
def test_flat_disk_preserves_planarity_during_tilt_relax() -> None:
    report_disabled = _report_for_mode("disabled")
    report_free = _report_for_mode("free")

    assert float(report_disabled["mesh"]["planarity_z_span"]) < 1e-12
    assert float(report_free["mesh"]["planarity_z_span"]) < 1e-12


@pytest.mark.regression
def test_outer_free_mode_does_not_shift_inner_theta_star() -> None:
    report_disabled = _report_for_mode("disabled")
    report_free = _report_for_mode("free")

    theta_disabled = float(report_disabled["mesh"]["theta_star"])
    theta_free = float(report_free["mesh"]["theta_star"])
    assert abs(theta_disabled) > 1e-12

    rel_shift = abs(theta_free - theta_disabled) / abs(theta_disabled)
    assert rel_shift < 0.10


@pytest.mark.regression
def test_flat_disk_splay_twist_mode_runs_with_zero_twist_default() -> None:
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_min=0.0,
        theta_max=0.0014,
        theta_count=8,
    )

    assert report["meta"]["smoothness_model"] == "splay_twist"
    breakdown = report["mesh"]["energy_breakdown"]
    assert "tilt_splay_twist_in" in breakdown
    assert float(report["mesh"]["planarity_z_span"]) < 1e-12
    assert float(report["parity"]["theta_factor"]) <= 2.0
    assert float(report["parity"]["energy_factor"]) <= 2.0


@pytest.mark.acceptance
def test_reproduce_flat_disk_one_leaflet_script_smoke(tmp_path) -> None:
    out_yaml = tmp_path / "flat_disk_one_leaflet_report.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(out_yaml),
            "--outer-mode",
            "disabled",
            "--refine-level",
            "1",
            "--theta-min",
            "0.0",
            "--theta-max",
            "0.0014",
            "--theta-count",
            "8",
        ],
        check=True,
        cwd=str(ROOT),
    )
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert report["meta"]["theory_source"] == "docs/tex/1_disk_flat.tex"
    assert report["meta"]["outer_mode"] == "disabled"
    assert float(report["theory"]["theta_star"]) > 0.0


@pytest.mark.regression
def test_flat_disk_empty_scan_bracket_raises_actionable_error() -> None:
    with pytest.raises(ValueError, match="minimum lies on theta scan boundary"):
        run_flat_disk_one_leaflet_benchmark(
            fixture=DEFAULT_FIXTURE,
            refine_level=1,
            outer_mode="disabled",
            theta_min=0.0,
            theta_max=1.0e-4,
            theta_count=4,
        )


@pytest.mark.regression
def test_flat_disk_missing_disk_group_raises_actionable_error(tmp_path) -> None:
    data = load_data(str(DEFAULT_FIXTURE))
    for vertex in data.get("vertices", []):
        if not (isinstance(vertex, list) and vertex and isinstance(vertex[-1], dict)):
            continue
        opts = vertex[-1]
        for key in (
            "rim_slope_match_group",
            "tilt_thetaB_group",
            "tilt_thetaB_group_in",
        ):
            if opts.get(key) == "disk":
                opts.pop(key, None)

    fixture_path = tmp_path / "flat_disk_missing_group.yaml"
    fixture_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(AssertionError, match="Missing or empty disk boundary group"):
        run_flat_disk_one_leaflet_benchmark(
            fixture=fixture_path,
            refine_level=0,
            outer_mode="disabled",
            theta_min=0.0,
            theta_max=0.0014,
            theta_count=8,
        )
