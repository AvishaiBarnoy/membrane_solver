import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools.theory_parity_interface_profiles import build_profiled_fixture

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ROOT / "tests" / "fixtures" / "theory_parity_targets.yaml"
SCRIPT = ROOT / "tools" / "reproduce_theory_parity.py"
BASE_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
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


def _get_path(dct: dict[str, Any], path: str) -> Any:
    cur: Any = dct
    for key in path.split("."):
        cur = cur[key]
    return cur


def _write_temp_fixture(doc: dict[str, Any], *, label: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{label}.yaml",
        prefix="theory_parity_acceptance_",
        delete=False,
        dir=str(ROOT / "tests" / "fixtures"),
        encoding="utf-8",
    )
    path = Path(handle.name)
    try:
        handle.write(yaml.safe_dump(doc, sort_keys=False))
    finally:
        handle.close()
    return path


def _build_physical_edge_profile_fixture(profile: str, lane: str) -> dict[str, Any]:
    base_doc = yaml.safe_load(BASE_FIXTURE.read_text(encoding="utf-8")) or {}
    doc = build_profiled_fixture(base_doc=base_doc, profile=profile, lane=lane)
    gp = dict(doc.get("global_parameters") or {})
    gp["rim_slope_match_mode"] = "physical_edge_staggered_v1"
    gp["tilt_solver"] = "cg"
    gp["tilt_cg_max_iters"] = 120
    gp["tilt_mass_mode_in"] = "consistent"
    doc["global_parameters"] = gp
    constraints = [str(x) for x in (doc.get("constraint_modules") or [])]
    doc["constraint_modules"] = [
        x for x in constraints if x != "tilt_thetaB_boundary_in"
    ]
    return doc


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

    tex = targets["targets"]["tex_benchmark"]
    for name in ("thetaB_star", "elastic_star", "contact_star", "total_star"):
        cfg = tex[name]
        actual = float(report["metrics"]["tex_benchmark"][name])
        expected = float(cfg["expected"])
        abs_tol = float(cfg["abs_tol"])
        assert actual == pytest.approx(expected, abs=abs_tol), (
            f"{name}: expected {expected} +/- {abs_tol}, got {actual}"
        )

    ratios = tex["ratios"]
    for name, cfg in ratios.items():
        actual = float(report["metrics"]["tex_benchmark"]["ratios"][name])
        expected = float(cfg["expected"])
        abs_tol = float(cfg["abs_tol"])
        assert actual == pytest.approx(expected, abs=abs_tol), (
            f"{name}: expected {expected} +/- {abs_tol}, got {actual}"
        )

    legacy_ratios = targets["targets"]["legacy_anchor"]["ratios"]
    for name, cfg in legacy_ratios.items():
        actual = float(report["metrics"]["legacy_anchor"]["ratios"][name])
        expected = float(cfg["expected"])
        abs_tol = float(cfg["abs_tol"])
        assert actual == pytest.approx(expected, abs=abs_tol), (
            f"legacy {name}: expected {expected} +/- {abs_tol}, got {actual}"
        )

    rel = targets["targets"]["relations"]
    reduced = report["metrics"]["reduced_terms"]
    if bool(rel.get("contact_measured_negative", False)):
        assert float(reduced["contact_measured"]) < 0.0
    if bool(rel.get("elastic_measured_positive", False)):
        assert float(reduced["elastic_measured"]) > 0.0
    if bool(rel.get("total_measured_negative", False)):
        assert float(reduced["total_measured"]) < 0.0


@pytest.mark.acceptance
def test_coarse_lane_reports_finite_outer_shell_geometry_for_parity(tmp_path) -> None:
    out_yaml = tmp_path / "coarse_outer_shell_report.yaml"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out_yaml)],
        check=True,
        cwd=str(ROOT),
    )

    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    geom = report["metrics"]["diagnostics"]["outer_shell_geometry"]

    assert bool(geom["available"])
    assert geom["construction_mode"] == "parity_disk_local_shell_measurement"
    assert float(geom["rim_radius"]) == pytest.approx(7.0 / 15.0, abs=5.0e-3)
    assert float(geom["outer_radius"]) < 1.0


@pytest.mark.acceptance
def test_physical_edge_profiled_lane_improves_thetaB_over_coarse_lane(tmp_path) -> None:
    coarse_out = tmp_path / "coarse_report.yaml"
    profiled_out = tmp_path / "profiled_report.yaml"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(coarse_out)],
        check=True,
        cwd=str(ROOT),
    )

    fixture_path = _write_temp_fixture(
        _build_physical_edge_profile_fixture("near_edge_v1", "near_edge_v1_acceptance"),
        label="near_edge_v1",
    )
    try:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--mesh",
                str(fixture_path),
                "--out",
                str(profiled_out),
            ],
            check=True,
            cwd=str(ROOT),
        )
    finally:
        if fixture_path.exists():
            fixture_path.unlink()

    coarse = yaml.safe_load(coarse_out.read_text(encoding="utf-8"))
    profiled = yaml.safe_load(profiled_out.read_text(encoding="utf-8"))

    assert profiled["meta"]["lane"] == "near_edge_v1_acceptance"
    assert float(profiled["metrics"]["thetaB_value"]) > float(
        coarse["metrics"]["thetaB_value"]
    )
    geom = profiled["metrics"]["diagnostics"]["outer_shell_geometry"]
    assert bool(geom["available"])
    assert geom["construction_mode"] == "physical_edge_local_shell"
    assert float(geom["rim_radius"]) == pytest.approx(7.0 / 15.0, abs=5.0e-3)
    assert float(geom["outer_radius"]) < 1.0


@pytest.mark.acceptance
def test_physical_edge_default_fixture_is_the_default_development_lane(
    tmp_path,
) -> None:
    coarse_out = tmp_path / "coarse_report.yaml"
    default_out = tmp_path / "default_report.yaml"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(coarse_out)],
        check=True,
        cwd=str(ROOT),
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(DEFAULT_FIXTURE),
            "--out",
            str(default_out),
        ],
        check=True,
        cwd=str(ROOT),
    )

    coarse = yaml.safe_load(coarse_out.read_text(encoding="utf-8"))
    default = yaml.safe_load(default_out.read_text(encoding="utf-8"))
    geom = default["metrics"]["diagnostics"]["outer_shell_geometry"]
    split = default["metrics"]["diagnostics"]["outer_split"]
    traces = default["metrics"]["diagnostics"]["interface_traces_at_R"]
    profile = default["metrics"]["diagnostics"]["outer_profile_parity"]

    default_theta = float(default["metrics"]["thetaB_value"])
    coarse_theta = float(coarse["metrics"]["thetaB_value"])
    tex_theta = float(default["metrics"]["tex_benchmark"]["thetaB_star"])
    default_total_ratio = float(
        default["metrics"]["tex_benchmark"]["ratios"]["total_ratio"]
    )
    coarse_total_ratio = float(
        coarse["metrics"]["tex_benchmark"]["ratios"]["total_ratio"]
    )
    default_elastic_ratio = float(
        default["metrics"]["tex_benchmark"]["ratios"]["elastic_ratio"]
    )
    coarse_elastic_ratio = float(
        coarse["metrics"]["tex_benchmark"]["ratios"]["elastic_ratio"]
    )

    assert default["meta"]["lane"] == "physical_edge_default"
    assert default_theta > float(coarse["metrics"]["thetaB_value"])
    assert abs(default_theta - tex_theta) < 0.01
    assert abs(default_theta - tex_theta) < abs(coarse_theta - tex_theta)
    assert abs(default_total_ratio - 1.0) < 0.01
    assert abs(default_total_ratio - 1.0) < abs(coarse_total_ratio - 1.0)
    assert abs(default_elastic_ratio - 1.0) < abs(coarse_elastic_ratio - 1.0)
    assert bool(geom["available"])
    assert geom["construction_mode"] == "physical_edge_local_shell"
    assert float(geom["rim_radius"]) == pytest.approx(7.0 / 15.0, abs=5.0e-3)
    assert float(geom["outer_radius"]) < 1.0
    assert float(split["phi_mean"]) > 0.0
    assert float(split["phi_over_half_theta"]) > 0.0
    assert abs(float(traces["disk_minus_phi_trace"])) < 0.01
    assert bool(traces["available"])
    assert bool(profile["available"])
    assert float(profile["sample_count"]) >= 10.0
    assert float(profile["phi_profile_rel_rmse"]) > 0.0
    assert float(profile["z_profile_rel_rmse"]) > 0.0


@pytest.mark.acceptance
def test_tracked_i50_fixture_improves_geometry_and_thetaB_over_coarse_lane(
    tmp_path,
) -> None:
    coarse_out = tmp_path / "coarse_report.yaml"
    i50_out = tmp_path / "i50_report.yaml"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(coarse_out)],
        check=True,
        cwd=str(ROOT),
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(I50_FIXTURE),
            "--out",
            str(i50_out),
        ],
        check=True,
        cwd=str(ROOT),
    )

    coarse = yaml.safe_load(coarse_out.read_text(encoding="utf-8"))
    i50 = yaml.safe_load(i50_out.read_text(encoding="utf-8"))

    coarse_theta = float(coarse["metrics"]["thetaB_value"])
    i50_theta = float(i50["metrics"]["thetaB_value"])
    geom = i50["metrics"]["diagnostics"]["outer_shell_geometry"]

    assert i50["meta"]["lane"] == "i50_interface_v1"
    assert i50_theta > coarse_theta
    assert bool(geom["available"])
    assert geom["construction_mode"] == "physical_edge_local_shell"
    assert float(geom["rim_radius"]) == pytest.approx(7.0 / 15.0, abs=5.0e-3)
    assert float(geom["outer_radius"]) < 1.0


@pytest.mark.acceptance
def test_tracked_i60_and_near_edge_fixtures_use_local_shell_physical_edge_mode(
    tmp_path,
) -> None:
    i60_out = tmp_path / "i60_report.yaml"
    near_edge_out = tmp_path / "near_edge_report.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(I60_FIXTURE),
            "--out",
            str(i60_out),
        ],
        check=True,
        cwd=str(ROOT),
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(NEAR_EDGE_FIXTURE),
            "--out",
            str(near_edge_out),
        ],
        check=True,
        cwd=str(ROOT),
    )

    i60 = yaml.safe_load(i60_out.read_text(encoding="utf-8"))
    near_edge = yaml.safe_load(near_edge_out.read_text(encoding="utf-8"))

    i60_geom = i60["metrics"]["diagnostics"]["outer_shell_geometry"]
    near_edge_geom = near_edge["metrics"]["diagnostics"]["outer_shell_geometry"]

    assert i60["meta"]["lane"] == "i60_interface_v1"
    assert near_edge["meta"]["lane"] == "near_edge_v1"
    assert i60_geom["construction_mode"] == "physical_edge_local_shell"
    assert near_edge_geom["construction_mode"] == "physical_edge_local_shell"
    assert float(i60_geom["rim_radius"]) == pytest.approx(7.0 / 15.0, abs=5.0e-3)
    assert float(near_edge_geom["rim_radius"]) == pytest.approx(7.0 / 15.0, abs=5.0e-3)
    assert float(i60_geom["outer_radius"]) < 1.0
    assert float(near_edge_geom["outer_radius"]) < 1.0


@pytest.mark.acceptance
def test_generated_physical_edge_family_varies_smoothly_around_primary(
    tmp_path,
) -> None:
    labels = [
        "default_lo",
        "default",
        "default_hi",
    ]
    reports = {}
    for label in labels:
        fixture_path = _write_temp_fixture(
            _build_physical_edge_profile_fixture(label, label),
            label=label,
        )
        out_yaml = tmp_path / f"{label}.yaml"
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--mesh",
                    str(fixture_path),
                    "--out",
                    str(out_yaml),
                ],
                check=True,
                cwd=str(ROOT),
            )
        finally:
            if fixture_path.exists():
                fixture_path.unlink()
        reports[label] = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))

    lo = reports["default_lo"]
    primary = reports["default"]
    hi = reports["default_hi"]

    lo_outer = float(
        lo["metrics"]["diagnostics"]["outer_shell_geometry"]["outer_radius"]
    )
    primary_outer = float(
        primary["metrics"]["diagnostics"]["outer_shell_geometry"]["outer_radius"]
    )
    hi_outer = float(
        hi["metrics"]["diagnostics"]["outer_shell_geometry"]["outer_radius"]
    )
    assert lo_outer > primary_outer > hi_outer

    lo_theta = float(lo["metrics"]["thetaB_value"])
    primary_theta = float(primary["metrics"]["thetaB_value"])
    hi_theta = float(hi["metrics"]["thetaB_value"])
    assert lo_theta <= primary_theta <= hi_theta

    lo_ratio = float(lo["metrics"]["tex_benchmark"]["ratios"]["total_ratio"])
    primary_ratio = float(primary["metrics"]["tex_benchmark"]["ratios"]["total_ratio"])
    hi_ratio = float(hi["metrics"]["tex_benchmark"]["ratios"]["total_ratio"])
    assert lo_ratio < primary_ratio
    assert hi_ratio > primary_ratio

    lo_split = lo["metrics"]["diagnostics"]["outer_split"]
    primary_split = primary["metrics"]["diagnostics"]["outer_split"]
    hi_split = hi["metrics"]["diagnostics"]["outer_split"]
    lo_traces = lo["metrics"]["diagnostics"]["interface_traces_at_R"]
    primary_traces = primary["metrics"]["diagnostics"]["interface_traces_at_R"]
    hi_traces = hi["metrics"]["diagnostics"]["interface_traces_at_R"]
    lo_profile = lo["metrics"]["diagnostics"]["outer_profile_parity"]
    primary_profile = primary["metrics"]["diagnostics"]["outer_profile_parity"]
    hi_profile = hi["metrics"]["diagnostics"]["outer_profile_parity"]

    lo_delta = float(lo["metrics"]["diagnostics"]["outer_shell_geometry"]["delta_r"])
    primary_delta = float(
        primary["metrics"]["diagnostics"]["outer_shell_geometry"]["delta_r"]
    )
    hi_delta = float(hi["metrics"]["diagnostics"]["outer_shell_geometry"]["delta_r"])
    assert lo_delta > primary_delta > hi_delta

    lo_phi = float(lo_split["phi_mean"])
    primary_phi = float(primary_split["phi_mean"])
    hi_phi = float(hi_split["phi_mean"])
    assert lo_phi < primary_phi <= hi_phi

    lo_phi_ratio = float(lo_split["phi_over_half_theta"])
    primary_phi_ratio = float(primary_split["phi_over_half_theta"])
    hi_phi_ratio = float(hi_split["phi_over_half_theta"])
    assert lo_phi_ratio < primary_phi_ratio <= hi_phi_ratio

    lo_tout = float(lo_split["t_out_mean"])
    primary_tout = float(primary_split["t_out_mean"])
    hi_tout = float(hi_split["t_out_mean"])
    assert abs(lo_tout) < 1.0e-6
    assert abs(primary_tout) < 1.0e-6
    assert abs(hi_tout) < 1.0e-6

    lo_tin = float(lo_split["t_in_mean"])
    primary_tin = float(primary_split["t_in_mean"])
    hi_tin = float(hi_split["t_in_mean"])
    assert max(lo_tin, primary_tin, hi_tin) - min(lo_tin, primary_tin, hi_tin) < 0.01

    trace_gap = [
        float(lo_traces["disk_minus_outer_trace"]),
        float(primary_traces["disk_minus_outer_trace"]),
        float(hi_traces["disk_minus_outer_trace"]),
    ]
    assert max(trace_gap) - min(trace_gap) < 0.02

    phi_trace = [
        float(lo_traces["phi_trace_at_R_plus"]),
        float(primary_traces["phi_trace_at_R_plus"]),
        float(hi_traces["phi_trace_at_R_plus"]),
    ]
    assert max(phi_trace) - min(phi_trace) < 0.02

    phi_rmse = [
        float(lo_profile["phi_profile_rel_rmse"]),
        float(primary_profile["phi_profile_rel_rmse"]),
        float(hi_profile["phi_profile_rel_rmse"]),
    ]
    z_rmse = [
        float(lo_profile["z_profile_rel_rmse"]),
        float(primary_profile["z_profile_rel_rmse"]),
        float(hi_profile["z_profile_rel_rmse"]),
    ]
    assert max(phi_rmse) - min(phi_rmse) < 0.1
    assert max(z_rmse) - min(z_rmse) < 0.05
