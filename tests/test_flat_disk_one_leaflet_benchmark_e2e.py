import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data
from tools.reproduce_flat_disk_one_leaflet import (
    DEFAULT_FIXTURE,
    BenchmarkOptimizeConfig,
    _build_minimizer,
    _configure_benchmark_mesh,
    _load_mesh_from_fixture,
    _radial_unit_vectors,
    _run_theta_optimize,
    _run_theta_relaxation,
    run_flat_disk_lane_comparison,
    run_flat_disk_one_leaflet_benchmark,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "reproduce_flat_disk_one_leaflet.py"


@lru_cache(maxsize=4)
def _report_for_mode(mode: str) -> dict:
    return run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=2,
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
    assert float(report["mesh"]["outer_decay_probe_max_after"]) < 2e-8
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
    assert float(report["parity"]["theta_factor"]) <= 2.5
    assert float(report["parity"]["energy_factor"]) <= 2.5


@pytest.mark.regression
def test_flat_disk_theta_mode_optimize_runs_and_reports_result() -> None:
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=2,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
    )

    assert report["meta"]["theta_mode"] == "optimize"
    assert report["meta"]["optimize_preset"] == "none"
    assert report["meta"]["optimize_preset_effective"] == "none"
    assert report["scan"] is None
    assert report["optimize"] is not None
    assert "optimize_theta_span" in report["optimize"]
    assert "hit_step_limit" in report["optimize"]
    assert isinstance(report["optimize"]["hit_step_limit"], bool)
    assert float(report["mesh"]["theta_star"]) > 0.0
    assert float(report["parity"]["theta_factor"]) <= 2.0
    assert float(report["parity"]["energy_factor"]) <= 2.0
    assert report["parity"]["recommended_mode_for_theory"] == "optimize"


@pytest.mark.regression
def test_flat_disk_theta_mode_optimize_full_runs_and_reports_polish() -> None:
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=2,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize_full",
        theta_polish_delta=1.0e-4,
        theta_polish_points=3,
    )

    assert report["meta"]["theta_mode"] == "optimize_full"
    assert report["scan"] is None
    assert report["optimize"] is not None
    opt = report["optimize"]
    polish = opt["polish"]
    assert polish is not None
    assert int(polish["polish_points"]) == 3
    assert float(polish["polish_delta"]) > 0.0
    assert len(polish["theta_values"]) == 3
    assert len(polish["energy_values"]) == 3
    assert opt["theta_star_raw"] is not None
    assert opt["theta_factor_raw"] is not None
    assert opt["energy_factor_raw"] is not None
    assert "optimize_theta_span" in opt
    assert "hit_step_limit" in opt
    assert isinstance(opt["hit_step_limit"], bool)
    assert float(report["mesh"]["theta_star"]) > 0.0
    assert float(report["parity"]["theta_factor"]) <= 2.0
    assert float(report["parity"]["energy_factor"]) <= 2.0
    assert report["parity"]["recommended_mode_for_theory"] in {
        "optimize",
        "optimize_full",
    }


@pytest.mark.regression
def test_flat_disk_optimize_preset_fast_r3_is_noop_below_refine3() -> None:
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        optimize_preset="fast_r3",
    )

    assert report["meta"]["optimize_preset"] == "fast_r3"
    assert report["meta"]["optimize_preset_effective"] == "fast_r3_inactive"
    opt = report["optimize"]
    assert opt is not None
    assert int(opt["optimize_steps"]) == 20
    assert int(opt["optimize_inner_steps"]) == 20
    assert float(opt["optimize_seconds"]) > 0.0


@pytest.mark.regression
def test_flat_disk_optimize_preset_full_accuracy_r3_is_noop_below_refine3() -> None:
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize_full",
        optimize_preset="full_accuracy_r3",
    )

    assert report["meta"]["optimize_preset"] == "full_accuracy_r3"
    assert report["meta"]["optimize_preset_effective"] == "full_accuracy_r3_inactive"
    opt = report["optimize"]
    assert opt is not None
    assert int(opt["optimize_steps"]) == 20
    assert int(opt["optimize_inner_steps"]) == 20
    assert opt["polish"] is not None


@pytest.mark.regression
def test_flat_disk_reports_splay_modulus_scale_meta() -> None:
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        splay_modulus_scale_in=0.5,
    )
    assert float(report["meta"]["splay_modulus_scale_in"]) == pytest.approx(0.5)


@pytest.mark.regression
def test_flat_disk_invalid_splay_modulus_scale_raises() -> None:
    with pytest.raises(ValueError, match="splay_modulus_scale_in must be > 0"):
        run_flat_disk_one_leaflet_benchmark(
            fixture=DEFAULT_FIXTURE,
            refine_level=1,
            outer_mode="disabled",
            smoothness_model="splay_twist",
            theta_mode="optimize",
            splay_modulus_scale_in=0.0,
        )


@pytest.mark.regression
def test_flat_disk_invalid_parameterization_raises() -> None:
    with pytest.raises(ValueError, match="parameterization must be"):
        run_flat_disk_one_leaflet_benchmark(
            fixture=DEFAULT_FIXTURE,
            refine_level=1,
            outer_mode="disabled",
            parameterization="invalid_mode",
        )


@pytest.mark.regression
def test_flat_disk_lane_comparison_reports_both_lanes() -> None:
    report = run_flat_disk_lane_comparison(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        legacy_theta_mode="scan",
        legacy_theta_min=0.0,
        legacy_theta_max=0.0014,
        legacy_theta_count=8,
        kh_theta_mode="optimize",
        kh_theta_initial=0.0,
        kh_theta_optimize_steps=6,
        kh_theta_optimize_every=1,
        kh_theta_optimize_delta=2.0e-4,
        kh_theta_optimize_inner_steps=5,
        kh_smoothness_model="splay_twist",
    )

    assert report["meta"]["mode"] == "compare_lanes"
    assert report["legacy"]["parity"]["lane"] == "legacy"
    assert report["kh_physical"]["parity"]["lane"] == "kh_physical"
    assert "comparison" in report
    comp = report["comparison"]
    assert float(comp["legacy_theta_star"]) > 0.0
    assert float(comp["kh_theta_star"]) > 0.0
    assert float(comp["kh_over_legacy_theta_star_ratio"]) > 1.0


@pytest.mark.regression
def test_flat_disk_kh_physical_parameterization_reports_unit_scaling() -> None:
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        theta_mode="optimize",
        parameterization="kh_physical",
        kappa_physical=10.0,
        kappa_t_physical=10.0,
        length_scale_nm=15.0,
        radius_nm=7.0,
        drive_physical=(2.0 / 0.7),
    )

    meta = report["meta"]
    theory = report["theory"]
    assert meta["parameterization"] == "kh_physical"
    assert bool(meta["using_physical_scaling"])
    assert float(meta["length_scale_nm"]) == pytest.approx(15.0, abs=1e-12)
    assert float(meta["radius_nm"]) == pytest.approx(7.0, abs=1e-12)
    assert float(meta["radius_dimless"]) == pytest.approx(7.0 / 15.0, abs=1e-12)
    assert float(theory["kappa"]) == pytest.approx(1.0, abs=1e-12)
    assert float(theory["kappa_t"]) == pytest.approx(225.0, abs=1e-12)
    assert float(theory["radius"]) == pytest.approx(7.0 / 15.0, abs=1e-12)


@pytest.mark.regression
def test_flat_disk_reports_rim_continuity_and_contact_diagnostics() -> None:
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=2,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
    )

    rim = report["mesh"]["rim_continuity"]
    assert int(rim["matched_bins"]) > 0
    assert np.isfinite(float(rim["jump_abs_median"]))
    assert np.isfinite(float(rim["jump_abs_max"]))

    contact = report["diagnostics"]["contact"]
    assert np.isfinite(float(contact["mesh_contact_energy"]))
    assert np.isfinite(float(contact["theory_contact_energy"]))
    assert np.isfinite(float(contact["mesh_contact_per_length"]))
    assert np.isfinite(float(contact["theory_contact_per_length"]))


@pytest.mark.regression
def test_flat_disk_optimize_mode_enforces_thetaB_on_full_disk_radius_ring() -> None:
    from runtime.refinement import refine_triangle_mesh
    from tools.diagnostics.flat_disk_one_leaflet_theory import tex_reference_params

    params = tex_reference_params()
    mesh = _load_mesh_from_fixture(Path(DEFAULT_FIXTURE))
    mesh = refine_triangle_mesh(mesh)
    _configure_benchmark_mesh(
        mesh,
        theory_params=params,
        parameterization="legacy",
        outer_mode="disabled",
        smoothness_model="splay_twist",
        splay_modulus_scale_in=1.0,
    )

    minim = _build_minimizer(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.project_tilts_to_tangent()

    theta_star = _run_theta_optimize(
        minim,
        optimize_cfg=BenchmarkOptimizeConfig(
            theta_initial=0.0,
            optimize_steps=60,
            optimize_every=1,
            optimize_delta=2.0e-4,
            optimize_inner_steps=60,
        ),
        reset_outer=True,
    )
    _run_theta_relaxation(minim, theta_value=float(theta_star), reset_outer=True)

    positions = mesh.positions_view()
    r, r_hat = _radial_unit_vectors(positions)
    rim_rows = np.flatnonzero(np.isclose(r, float(params.radius), atol=1e-6))
    assert rim_rows.size > 0

    theta_vals = np.einsum("ij,ij->i", mesh.tilts_in_view()[rim_rows], r_hat[rim_rows])
    assert float(np.max(np.abs(theta_vals - float(theta_star)))) < 1e-10


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


@pytest.mark.acceptance
def test_reproduce_flat_disk_one_leaflet_script_smoke_theta_optimize(tmp_path) -> None:
    out_yaml = tmp_path / "flat_disk_one_leaflet_report_opt.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(out_yaml),
            "--outer-mode",
            "disabled",
            "--smoothness-model",
            "splay_twist",
            "--theta-mode",
            "optimize",
            "--theta-initial",
            "0.0",
            "--theta-optimize-steps",
            "20",
            "--theta-optimize-every",
            "1",
            "--theta-optimize-delta",
            "0.0002",
            "--theta-optimize-inner-steps",
            "20",
        ],
        check=True,
        cwd=str(ROOT),
    )
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert report["meta"]["theta_mode"] == "optimize"
    assert report["scan"] is None
    assert report["optimize"] is not None
    assert float(report["mesh"]["theta_star"]) > 0.0
    assert float(report["parity"]["theta_factor"]) <= 2.0


@pytest.mark.acceptance
def test_reproduce_flat_disk_one_leaflet_script_smoke_theta_optimize_full(
    tmp_path,
) -> None:
    out_yaml = tmp_path / "flat_disk_one_leaflet_report_opt_full.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(out_yaml),
            "--outer-mode",
            "disabled",
            "--smoothness-model",
            "splay_twist",
            "--theta-mode",
            "optimize_full",
            "--theta-initial",
            "0.0",
            "--theta-optimize-steps",
            "20",
            "--theta-optimize-every",
            "1",
            "--theta-optimize-delta",
            "0.0002",
            "--theta-optimize-inner-steps",
            "20",
            "--theta-polish-delta",
            "0.0001",
            "--theta-polish-points",
            "3",
        ],
        check=True,
        cwd=str(ROOT),
    )
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert report["meta"]["theta_mode"] == "optimize_full"
    assert report["scan"] is None
    assert report["optimize"] is not None
    assert report["optimize"]["polish"] is not None
    assert float(report["mesh"]["theta_star"]) > 0.0
    assert float(report["parity"]["theta_factor"]) <= 2.0


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
