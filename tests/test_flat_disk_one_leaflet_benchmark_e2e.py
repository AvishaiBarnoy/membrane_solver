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
from tools.diagnostics.flat_disk_kh_term_audit import run_flat_disk_kh_term_audit
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


@lru_cache(maxsize=8)
def _kh_opt_report(
    *,
    refine_level: int,
    optimize_preset: str,
    tilt_mass_mode_in: str = "consistent",
) -> dict:
    preset = str(optimize_preset)
    # Strict presets force refine=1 in benchmark harness; normalize cache key so
    # tests reuse the same expensive run.
    strict_presets = {
        "kh_strict_refine",
        "kh_strict_fast",
        "kh_strict_balanced",
        "kh_strict_continuity",
        "kh_strict_energy_tight",
        "kh_strict_section_tight",
        "kh_strict_outerband_tight",
        "kh_strict_partition_tight",
        "kh_strict_robust",
    }
    if preset in strict_presets:
        effective_refine = (
            2
            if preset in {"kh_strict_section_tight", "kh_strict_outerband_tight"}
            else 1
        )
    else:
        effective_refine = int(refine_level)
    return run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=effective_refine,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        parameterization="kh_physical",
        optimize_preset=preset,
        tilt_mass_mode_in=str(tilt_mass_mode_in),
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

    assert float(report["parity"]["theta_factor"]) <= 2.2
    assert float(report["parity"]["energy_factor"]) <= 2.0
    assert float(report["mesh"]["outer_tilt_max_free_rows"]) < 1e-9
    assert float(report["mesh"]["outer_decay_probe_max_before"]) > 1e-5
    # Keep a strict near-zero post-relax bound while avoiding false failures
    # from tiny solver-floor variation in CI.
    assert float(report["mesh"]["outer_decay_probe_max_after"]) < 1e-5
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
        refine_level=1,
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
    assert float(report["parity"]["theta_factor"]) <= 2.2
    assert float(report["parity"]["energy_factor"]) <= 2.0
    assert report["parity"]["recommended_mode_for_theory"] == "optimize"


@pytest.mark.regression
def test_flat_disk_theta_mode_optimize_full_runs_and_reports_polish() -> None:
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize_full",
        theta_optimize_steps=16,
        theta_optimize_inner_steps=16,
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
    assert float(report["parity"]["theta_factor"]) <= 2.2
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


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_wide_expands_theta_span_for_kh_lane() -> None:
    report = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_wide",
    )

    assert report["meta"]["optimize_preset"] == "kh_wide"
    assert report["meta"]["optimize_preset_effective"] == "kh_wide"
    opt = report["optimize"]
    assert opt is not None
    assert int(opt["optimize_steps"]) == 120
    assert float(opt["optimize_delta"]) == pytest.approx(2.0e-3, abs=0.0)
    assert float(opt["optimize_theta_span"]) == pytest.approx(0.24, abs=1e-12)
    assert bool(opt["hit_step_limit"]) is False
    assert float(report["mesh"]["theta_star"]) > 0.02


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_strict_fast_is_opt_in_and_mesh_strict() -> None:
    report = _kh_opt_report(
        refine_level=3,  # should be overridden by strict-fast preset
        optimize_preset="kh_strict_fast",
    )

    assert report["meta"]["optimize_preset"] == "kh_strict_fast"
    assert report["meta"]["optimize_preset_effective"] == "kh_strict_fast"
    assert int(report["meta"]["refine_level"]) == 1
    assert int(report["meta"]["rim_local_refine_steps"]) == 1
    assert float(report["meta"]["rim_local_refine_band_lambda"]) == pytest.approx(4.0)

    opt = report["optimize"]
    assert opt is not None
    assert int(opt["optimize_steps"]) == 30
    assert int(opt["optimize_inner_steps"]) == 14
    assert int(opt["plateau_patience"]) == 12
    assert float(opt["plateau_abs_tol"]) == pytest.approx(1.0e-12, abs=0.0)
    assert int(opt["optimize_iterations_completed"]) <= int(opt["optimize_steps"])
    assert isinstance(opt["stopped_on_plateau"], bool)
    assert float(opt["optimize_delta"]) == pytest.approx(6.0e-3, abs=0.0)
    assert float(opt["optimize_theta_span"]) == pytest.approx(0.18, abs=1e-12)
    assert float(opt["optimize_theta_span_completed"]) <= float(
        opt["optimize_theta_span"]
    )
    assert bool(opt["hit_step_limit"]) is False
    assert opt["recommended_fallback_preset"] is None


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_strict_fast_respects_explicit_rim_overrides() -> (
    None
):
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=2,  # still forced to strict level
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        parameterization="kh_physical",
        optimize_preset="kh_strict_fast",
        rim_local_refine_steps=3,
        rim_local_refine_band_lambda=2.5,
    )

    assert report["meta"]["optimize_preset_effective"] == "kh_strict_fast"
    assert int(report["meta"]["refine_level"]) == 1
    assert int(report["meta"]["rim_local_refine_steps"]) == 3
    assert float(report["meta"]["rim_local_refine_band_lambda"]) == pytest.approx(2.5)


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_strict_continuity_improves_rim_metrics() -> None:
    fast = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_fast",
    )
    continuity = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_continuity",
    )

    assert continuity["meta"]["optimize_preset_effective"] == "kh_strict_continuity"
    assert int(continuity["meta"]["refine_level"]) == 1
    assert int(continuity["meta"]["rim_local_refine_steps"]) == 2
    assert float(continuity["parity"]["theta_factor"]) <= 1.2
    assert float(continuity["parity"]["energy_factor"]) <= 1.2

    jump_fast = float(fast["mesh"]["rim_continuity"]["jump_abs_median"]) / max(
        float(fast["mesh"]["profile"]["rim_abs_median"]), 1e-18
    )
    jump_cont = float(continuity["mesh"]["rim_continuity"]["jump_abs_median"]) / max(
        float(continuity["mesh"]["profile"]["rim_abs_median"]), 1e-18
    )
    leak_fast = float(fast["mesh"]["leakage"]["outer_tphi_over_trad_median"])
    leak_cont = float(continuity["mesh"]["leakage"]["outer_tphi_over_trad_median"])
    assert jump_cont < jump_fast
    assert leak_cont < leak_fast


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_strict_balanced_tradeoff_vs_fast() -> None:
    fast = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_fast",
    )
    balanced = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_balanced",
    )

    assert balanced["meta"]["optimize_preset_effective"] == "kh_strict_balanced"
    assert int(balanced["meta"]["refine_level"]) == 1
    assert int(balanced["meta"]["rim_local_refine_steps"]) == 2
    assert float(balanced["meta"]["rim_local_refine_band_lambda"]) == pytest.approx(3.0)
    assert balanced["optimize"]["parity_polish"] is not None

    score_fast = float(
        np.hypot(
            np.log(max(float(fast["parity"]["theta_factor"]), 1e-18)),
            np.log(max(float(fast["parity"]["energy_factor"]), 1e-18)),
        )
    )
    score_balanced = float(
        np.hypot(
            np.log(max(float(balanced["parity"]["theta_factor"]), 1e-18)),
            np.log(max(float(balanced["parity"]["energy_factor"]), 1e-18)),
        )
    )
    assert score_balanced <= score_fast
    assert float(balanced["parity"]["theta_factor"]) <= 1.2
    assert float(balanced["parity"]["energy_factor"]) <= 1.2


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_strict_energy_tight_controls() -> None:
    report = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_energy_tight",
    )

    assert report["meta"]["optimize_preset_effective"] == "kh_strict_energy_tight"
    assert int(report["meta"]["refine_level"]) == 1
    assert int(report["meta"]["rim_local_refine_steps"]) == 2
    assert float(report["meta"]["rim_local_refine_band_lambda"]) == pytest.approx(8.0)
    assert report["optimize"]["parity_polish"] is not None
    assert report["optimize"]["parity_polish"]["objective"] == "energy_factor"
    assert float(report["parity"]["theta_factor"]) <= 1.10
    assert float(report["parity"]["energy_factor"]) <= 1.10


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_strict_partition_tight_controls() -> None:
    report = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_partition_tight",
    )

    assert report["meta"]["optimize_preset_effective"] == "kh_strict_partition_tight"
    assert int(report["meta"]["refine_level"]) == 1
    assert int(report["meta"]["rim_local_refine_steps"]) == 2
    assert float(report["meta"]["rim_local_refine_band_lambda"]) == pytest.approx(10.0)
    assert report["optimize"]["parity_polish"] is not None
    assert float(report["parity"]["theta_factor"]) <= 1.2
    assert float(report["parity"]["energy_factor"]) <= 1.2


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_strict_section_tight_controls() -> None:
    report = _kh_opt_report(
        refine_level=1,  # should be overridden by section-tight preset
        optimize_preset="kh_strict_section_tight",
    )

    assert report["meta"]["optimize_preset_effective"] == "kh_strict_section_tight"
    assert int(report["meta"]["refine_level"]) == 2
    assert int(report["meta"]["rim_local_refine_steps"]) == 1
    assert float(report["meta"]["rim_local_refine_band_lambda"]) == pytest.approx(4.0)
    assert report["optimize"]["parity_polish"] is not None
    assert report["optimize"]["parity_polish"]["objective"] == "energy_factor"
    assert float(report["parity"]["theta_factor"]) <= 1.25
    assert float(report["parity"]["energy_factor"]) <= 1.25


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_strict_outerband_tight_balances_outer_inner_and_global() -> (
    None
):
    section_tight = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_section_tight",
    )
    outerband_tight = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_outerband_tight",
    )

    assert (
        outerband_tight["meta"]["optimize_preset_effective"]
        == "kh_strict_outerband_tight"
    )
    assert int(outerband_tight["meta"]["refine_level"]) == 2

    def _audit_row(report: dict, *, theta_value: float) -> dict:
        audit = run_flat_disk_kh_term_audit(
            fixture=DEFAULT_FIXTURE,
            refine_level=int(report["meta"]["refine_level"]),
            outer_mode="disabled",
            smoothness_model="splay_twist",
            kappa_physical=10.0,
            kappa_t_physical=10.0,
            radius_nm=7.0,
            length_scale_nm=15.0,
            drive_physical=(2.0 / 0.7),
            theta_values=(float(theta_value),),
            tilt_mass_mode_in="consistent",
            rim_local_refine_steps=int(report["meta"]["rim_local_refine_steps"]),
            rim_local_refine_band_lambda=float(
                report["meta"]["rim_local_refine_band_lambda"]
            ),
        )
        return audit["rows"][0]

    # Compare section decomposition at a fixed theta to isolate mesh/partition effects.
    section_row = _audit_row(section_tight, theta_value=0.138)
    outer_row = _audit_row(outerband_tight, theta_value=0.138)
    near_err_section = abs(
        float(section_row["internal_outer_near_ratio_mesh_over_theory"]) - 1.0
    )
    near_err_outerband = abs(
        float(outer_row["internal_outer_near_ratio_mesh_over_theory"]) - 1.0
    )
    far_err_section = abs(
        float(section_row["internal_outer_far_ratio_mesh_over_theory"]) - 1.0
    )
    far_err_outerband = abs(
        float(outer_row["internal_outer_far_ratio_mesh_over_theory"]) - 1.0
    )
    score_outer_section = float(
        np.hypot(
            np.log(
                max(
                    float(section_row["internal_outer_near_ratio_mesh_over_theory"]),
                    1e-18,
                )
            ),
            np.log(
                max(
                    float(section_row["internal_outer_far_ratio_mesh_over_theory"]),
                    1e-18,
                )
            ),
        )
    )
    score_outer_outerband = float(
        np.hypot(
            np.log(
                max(
                    float(outer_row["internal_outer_near_ratio_mesh_over_theory"]),
                    1e-18,
                )
            ),
            np.log(
                max(
                    float(outer_row["internal_outer_far_ratio_mesh_over_theory"]), 1e-18
                )
            ),
        )
    )
    assert score_outer_outerband <= score_outer_section
    assert (near_err_outerband < near_err_section) or (
        far_err_outerband < far_err_section
    )

    inner_err_section = abs(
        np.log(max(float(section_row["internal_disk_ratio_mesh_over_theory"]), 1e-18))
    )
    inner_err_outerband = abs(
        np.log(max(float(outer_row["internal_disk_ratio_mesh_over_theory"]), 1e-18))
    )
    assert inner_err_outerband <= inner_err_section

    assert float(outerband_tight["parity"]["theta_factor"]) <= (
        float(section_tight["parity"]["theta_factor"]) * 1.01
    )
    assert float(outerband_tight["parity"]["energy_factor"]) <= (
        float(section_tight["parity"]["energy_factor"]) * 1.01
    )


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_strict_outerband_tight_composite_score_non_worsening() -> (
    None
):
    section_tight = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_section_tight",
    )
    outerband_tight = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_outerband_tight",
    )

    def _audit_row(report: dict) -> dict:
        theta = float(report["mesh"]["theta_star"])
        audit = run_flat_disk_kh_term_audit(
            fixture=DEFAULT_FIXTURE,
            refine_level=int(report["meta"]["refine_level"]),
            outer_mode="disabled",
            smoothness_model="splay_twist",
            kappa_physical=10.0,
            kappa_t_physical=10.0,
            radius_nm=7.0,
            length_scale_nm=15.0,
            drive_physical=(2.0 / 0.7),
            theta_values=(theta,),
            tilt_mass_mode_in="consistent",
            rim_local_refine_steps=int(report["meta"]["rim_local_refine_steps"]),
            rim_local_refine_band_lambda=float(
                report["meta"]["rim_local_refine_band_lambda"]
            ),
        )
        return audit["rows"][0]

    sec_row = _audit_row(section_tight)
    out_row = _audit_row(outerband_tight)

    def _composite_score(report: dict, row: dict) -> float:
        ratios = np.asarray(
            [
                float(report["parity"]["theta_factor"]),
                float(report["parity"]["energy_factor"]),
                float(row["internal_disk_ratio_mesh_over_theory"]),
                float(row["internal_outer_near_ratio_mesh_over_theory"]),
                float(row["internal_outer_far_ratio_mesh_over_theory"]),
            ],
            dtype=float,
        )
        logs = np.log(np.maximum(ratios, 1e-18))
        return float(np.sqrt(np.mean(logs * logs)))

    score_sec = _composite_score(section_tight, sec_row)
    score_out = _composite_score(outerband_tight, out_row)
    assert score_out <= (score_sec * 1.005)


@pytest.mark.benchmark
def test_flat_disk_kh_strict_section_tight_non_worsening_vs_energy_tight() -> None:
    energy_tight = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_energy_tight",
    )
    section_tight = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_section_tight",
    )

    assert float(section_tight["parity"]["energy_factor"]) <= float(
        energy_tight["parity"]["energy_factor"]
    )

    def _section_scores(report: dict) -> tuple[float, float]:
        theta = float(report["mesh"]["theta_star"])
        audit = run_flat_disk_kh_term_audit(
            fixture=DEFAULT_FIXTURE,
            refine_level=int(report["meta"]["refine_level"]),
            outer_mode="disabled",
            smoothness_model="splay_twist",
            kappa_physical=10.0,
            kappa_t_physical=10.0,
            radius_nm=7.0,
            length_scale_nm=15.0,
            drive_physical=(2.0 / 0.7),
            theta_values=(theta,),
            tilt_mass_mode_in="consistent",
            rim_local_refine_steps=int(report["meta"]["rim_local_refine_steps"]),
            rim_local_refine_band_lambda=float(
                report["meta"]["rim_local_refine_band_lambda"]
            ),
        )
        row = audit["rows"][0]
        return (
            float(row["section_score_internal_split_l2_log"]),
            float(row["section_score_all_terms_l2_log"]),
        )

    energy_split, energy_all = _section_scores(energy_tight)
    section_split, section_all = _section_scores(section_tight)
    assert section_split <= energy_split
    assert section_all <= energy_all


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_strict_robust_non_worsening_score() -> None:
    fast = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_fast",
    )
    robust = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_robust",
    )

    score_fast = float(
        np.hypot(
            np.log(max(float(fast["parity"]["theta_factor"]), 1e-18)),
            np.log(max(float(fast["parity"]["energy_factor"]), 1e-18)),
        )
    )
    score_robust = float(
        np.hypot(
            np.log(max(float(robust["parity"]["theta_factor"]), 1e-18)),
            np.log(max(float(robust["parity"]["energy_factor"]), 1e-18)),
        )
    )
    assert robust["meta"]["optimize_preset_effective"] == "kh_strict_robust"
    assert robust["optimize"]["postcheck"] is not None
    assert score_robust <= 1.30 * score_fast


@pytest.mark.regression
def test_flat_disk_kh_consistent_mass_improves_energy_parity_lightweight() -> None:
    lumped = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        parameterization="kh_physical",
        optimize_preset="none",
        theta_optimize_steps=12,
        theta_optimize_every=1,
        theta_optimize_delta=8.0e-4,
        theta_optimize_inner_steps=8,
        tilt_mass_mode_in="lumped",
    )
    consistent = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        parameterization="kh_physical",
        optimize_preset="none",
        theta_optimize_steps=12,
        theta_optimize_every=1,
        theta_optimize_delta=8.0e-4,
        theta_optimize_inner_steps=8,
        tilt_mass_mode_in="consistent",
    )

    assert float(consistent["parity"]["energy_factor"]) < float(
        lumped["parity"]["energy_factor"]
    )
    assert np.isfinite(float(consistent["parity"]["energy_factor"]))
    assert np.isfinite(float(lumped["parity"]["energy_factor"]))


@pytest.mark.acceptance
@pytest.mark.e2e
def test_flat_disk_kh_optimize_profile_and_continuity_e2e() -> None:
    report = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_strict_fast",
    )

    assert report["meta"]["theta_mode"] == "optimize"
    assert report["meta"]["optimize_preset_effective"] == "kh_strict_fast"
    assert float(report["parity"]["theta_factor"]) <= 1.2
    assert float(report["parity"]["energy_factor"]) <= 1.2

    profile = report["mesh"]["profile"]
    inner = float(profile["inner_abs_median"])
    rim = float(profile["rim_abs_median"])
    outer = float(profile["outer_abs_median"])
    assert inner > 0.0
    assert rim > inner
    assert outer < inner

    continuity = report["mesh"]["rim_continuity"]
    assert int(continuity["matched_bins"]) > 0
    jump_ratio = float(continuity["jump_abs_median"]) / max(float(rim), 1e-18)
    # With the corrected inner-disk topology, strict KH parity improves while the
    # rim jump is finite due to non-axisymmetric local triangulation. Keep it bounded.
    assert jump_ratio < 0.30
    rim_bc = report["mesh"]["rim_boundary_realization"]
    assert float(rim_bc["rim_theta_error_abs_median"]) <= 1e-12
    assert float(report["mesh"]["planarity_z_span"]) < 1e-12


@pytest.mark.benchmark
def test_flat_disk_kh_optimize_parity_does_not_worsen_with_refinement() -> None:
    refine1 = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_wide",
    )
    refine2 = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=2,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        parameterization="kh_physical",
        optimize_preset="kh_wide",
    )

    t1 = float(refine1["parity"]["theta_factor"])
    e1 = float(refine1["parity"]["energy_factor"])
    t2 = float(refine2["parity"]["theta_factor"])
    e2 = float(refine2["parity"]["energy_factor"])
    assert t2 <= t1, (t1, t2)
    assert e2 <= e1, (e1, e2)


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


@pytest.mark.benchmark
def test_flat_disk_kh_default_splay_scale_stays_unmodified() -> None:
    report = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_wide",
    )
    assert float(report["meta"]["splay_modulus_scale_in"]) == pytest.approx(1.0)


@pytest.mark.benchmark
def test_flat_disk_kh_strict_preset_improves_score_vs_baseline() -> None:
    baseline = _kh_opt_report(
        refine_level=1,
        optimize_preset="kh_wide",
    )
    strict = _kh_opt_report(
        refine_level=2,  # should be overridden by strict preset
        optimize_preset="kh_strict_fast",
    )

    score_base = float(
        np.hypot(
            np.log(max(float(baseline["parity"]["theta_factor"]), 1e-18)),
            np.log(max(float(baseline["parity"]["energy_factor"]), 1e-18)),
        )
    )
    score_strict = float(
        np.hypot(
            np.log(max(float(strict["parity"]["theta_factor"]), 1e-18)),
            np.log(max(float(strict["parity"]["energy_factor"]), 1e-18)),
        )
    )

    assert int(strict["meta"]["refine_level"]) == 1
    assert int(strict["meta"]["rim_local_refine_steps"]) == 1
    assert score_strict <= score_base
    assert float(strict["parity"]["theta_factor"]) <= 1.2
    assert float(strict["parity"]["energy_factor"]) <= 1.2

    rim = strict["mesh"]["rim_boundary_realization"]
    assert int(rim["rim_samples"]) > 0
    assert np.isfinite(float(rim["rim_theta_error_abs_median"]))
    assert np.isfinite(float(rim["rim_theta_error_abs_max"]))
    assert float(rim["rim_theta_error_abs_median"]) <= 1e-12
    assert float(rim["rim_theta_error_abs_max"]) <= 1e-12

    leakage = strict["mesh"]["leakage"]
    assert np.isfinite(float(leakage["inner_tphi_over_trad_median"]))
    assert np.isfinite(float(leakage["outer_tphi_over_trad_median"]))
    # Regression caps from strict baseline (refine1 + rim-local step 1).
    assert float(leakage["inner_tphi_over_trad_median"]) <= 0.03
    assert float(leakage["outer_tphi_over_trad_median"]) <= 1.40


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
        refine_level=1,
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
        tilt_mass_mode_in="lumped",
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
