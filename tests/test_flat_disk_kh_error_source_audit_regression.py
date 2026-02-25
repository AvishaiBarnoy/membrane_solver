import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_kh_error_source_audit import (
    _rank_effects,
    run_flat_disk_kh_discrete_quality_safe_refine_trend,
    run_flat_disk_kh_discrete_quality_safe_sweep,
    run_flat_disk_kh_discrete_quality_sweep,
    run_flat_disk_kh_error_source_audit,
    run_flat_disk_kh_error_source_candidate_bakeoff,
    run_flat_disk_kh_fractional_refinement_trend,
)


@pytest.mark.unit
def test_flat_disk_kh_error_source_rank_effects_orders_descending() -> None:
    ranked = _rank_effects(
        {
            "partition_effect": 0.3,
            "mass_effect": 0.1,
            "resolution_effect": 0.2,
            "operator_effect": 0.05,
        }
    )
    assert ranked["dominant_source"] == "partition_effect"
    assert ranked["ranking"][0] == "partition_effect"
    assert 0.0 < float(ranked["confidence"]) <= 1.0


@pytest.mark.regression
def test_flat_disk_kh_error_source_audit_emits_finite_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_error_source_audit as mod

    def _fake_benchmark(**kwargs):
        refine = int(kwargs["refine_level"])
        preset = str(kwargs["optimize_preset"])
        return {
            "mesh": {
                "theta_star": 0.10
                + 0.01 * refine
                + (0.01 if "outerband" in preset else 0.0)
            }
        }

    def _fake_term_audit(**kwargs):
        refine = int(kwargs["refine_level"])
        mass = str(kwargs["tilt_mass_mode_in"])
        part = str(kwargs.get("partition_mode", "centroid"))
        near = 1.10 + 0.02 * (refine - 2)
        far = 0.95 - (0.01 if mass == "lumped" else 0.0)
        score = float(np.hypot(np.log(near), np.log(max(far, 1e-18))))
        if part == "fractional":
            score *= 0.9
        return {
            "rows": [
                {
                    "internal_disk_ratio_mesh_over_theory": 1.1,
                    "internal_outer_near_ratio_mesh_over_theory_finite": near,
                    "internal_outer_far_ratio_mesh_over_theory_finite": far,
                    "section_score_internal_bands_finite_outer_l2_log": score,
                    "proj_radial_internal_outer_near_abs_error_delta_vs_unprojected": 0.03,
                    "proj_radial_internal_outer_far_abs_error_delta_vs_unprojected": 0.02,
                }
            ]
        }

    monkeypatch.setattr(mod, "run_flat_disk_one_leaflet_benchmark", _fake_benchmark)
    monkeypatch.setattr(mod, "run_flat_disk_kh_term_audit", _fake_term_audit)

    report = run_flat_disk_kh_error_source_audit(
        refine_levels=(2, 3),
        mass_modes=("consistent", "lumped"),
        partition_modes=("centroid", "fractional"),
    )
    assert report["meta"]["mode"] == "kh_error_source_audit"
    assert report["meta"]["primary_partition_mode"] == "fractional"
    assert len(report["runs"]) == 16
    assert report["meta"]["partition_modes"] == ["centroid", "fractional"]
    for row in report["runs"]:
        assert np.isfinite(
            float(row["section_score_internal_bands_finite_outer_l2_log"])
        )
        assert row["partition_mode"] in {"centroid", "fractional"}
    assert report["attribution"]["dominant_source"] in {
        "partition_effect",
        "mass_effect",
        "resolution_effect",
        "operator_effect",
    }
    assert float(report["attribution"]["effect_sizes"]["partition_effect"]) > 0.0


@pytest.mark.regression
def test_flat_disk_kh_error_source_candidate_bakeoff_selects_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_error_source_audit as mod

    def _fake_benchmark(**kwargs):
        preset = str(kwargs["optimize_preset"])
        if "outerfield_averaged" in preset:
            return {"parity": {"theta_factor": 1.06, "energy_factor": 1.10}}
        if "outerband" in preset:
            return {"parity": {"theta_factor": 1.09, "energy_factor": 1.14}}
        return {"parity": {"theta_factor": 1.12, "energy_factor": 1.18}}

    def _fake_error_source_audit(**kwargs):
        preset = str(kwargs["primary_preset"])
        if "outerfield_averaged" in preset:
            near, far, part = 1.04, 0.98, 0.02
        elif "outerband" in preset:
            near, far, part = 1.08, 0.96, 0.03
        else:
            near, far, part = 1.12, 0.92, 0.04
        return {
            "runs": [
                {"outer_near_ratio": near, "outer_far_ratio": far},
                {"outer_near_ratio": near, "outer_far_ratio": far},
            ],
            "attribution": {
                "dominant_source": "partition_effect",
                "effect_sizes": {
                    "partition_effect": part,
                    "mass_effect": 0.01,
                    "resolution_effect": 0.02,
                    "operator_effect": 0.03,
                },
            },
        }

    monkeypatch.setattr(mod, "run_flat_disk_one_leaflet_benchmark", _fake_benchmark)
    monkeypatch.setattr(
        mod, "run_flat_disk_kh_error_source_audit", _fake_error_source_audit
    )

    report = run_flat_disk_kh_error_source_candidate_bakeoff(
        optimize_presets=(
            "kh_strict_outerfield_tight",
            "kh_strict_outerband_tight",
            "kh_strict_outerfield_averaged",
        ),
        refine_level=2,
    )
    assert report["meta"]["mode"] == "kh_error_source_candidate_bakeoff"
    rows = report["candidates"]
    assert len(rows) == 3
    for row in rows:
        assert np.isfinite(float(row["outer_section_score"]))
        assert np.isfinite(float(row["balanced_parity_score"]))
        assert np.isfinite(float(row["runtime_seconds"]))
    assert report["selected_best"]["optimize_preset"] == "kh_strict_outerfield_averaged"


@pytest.mark.regression
def test_flat_disk_kh_fractional_refinement_trend_emits_rows_and_monotonic_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_error_source_audit as mod

    def _fake_error_source_audit(**kwargs):
        level = int(kwargs["refine_levels"][0])
        score = {1: 0.22, 2: 0.14, 3: 0.10}[level]
        return {
            "runs": [
                {
                    "preset": str(kwargs["primary_preset"]),
                    "refine_level": int(level),
                    "tilt_mass_mode_in": "consistent",
                    "partition_mode": "fractional",
                    "disk_ratio": 1.1,
                    "outer_near_ratio": 1.05,
                    "outer_far_ratio": 0.98,
                    "section_score_internal_bands_finite_outer_l2_log": float(score),
                }
            ]
        }

    monkeypatch.setattr(
        mod, "run_flat_disk_kh_error_source_audit", _fake_error_source_audit
    )
    report = run_flat_disk_kh_fractional_refinement_trend(
        optimize_preset="kh_strict_outerfield_best",
        refine_levels=(1, 2, 3),
        mass_mode="consistent",
    )
    assert report["meta"]["mode"] == "kh_fractional_refinement_trend"
    assert report["meta"]["primary_partition_mode"] == "fractional"
    rows = report["trend"]["rows"]
    assert len(rows) == 3
    assert [int(r["refine_level"]) for r in rows] == [1, 2, 3]
    assert bool(report["trend"]["monotone_non_worsening"]) is True
    for row in rows:
        assert np.isfinite(
            float(row["section_score_internal_bands_finite_outer_l2_log"])
        )


@pytest.mark.regression
def test_flat_disk_kh_discrete_quality_sweep_emits_rows_and_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_error_source_audit as mod

    def _fake_benchmark(**kwargs):
        rmax = float(kwargs["outer_local_refine_rmax_lambda"])
        flips = int(kwargs["local_edge_flip_steps"])
        avg = int(kwargs["outer_local_vertex_average_steps"])
        theta_star = 0.132 + 0.002 * (rmax - 8.0) - 0.0005 * flips - 0.0003 * (avg - 2)
        theta_factor = 1.06 - 0.01 * flips - 0.004 * (avg - 2)
        energy_factor = 1.15 - 0.02 * flips - 0.01 * (avg - 2) - 0.002 * (rmax - 8.0)
        return {
            "mesh": {"theta_star": float(theta_star)},
            "parity": {
                "theta_factor": float(theta_factor),
                "energy_factor": float(energy_factor),
            },
            "meta": {
                "refine_level": int(kwargs["refine_level"]),
                "rim_local_refine_steps": 1,
                "rim_local_refine_band_lambda": 4.0,
                "outer_local_refine_steps": 1,
                "outer_local_refine_rmin_lambda": 1.0,
                "outer_local_refine_rmax_lambda": float(rmax),
                "local_edge_flip_steps": int(flips),
                "local_edge_flip_rmin_lambda": 2.0,
                "local_edge_flip_rmax_lambda": 6.0,
                "outer_local_vertex_average_steps": int(avg),
                "outer_local_vertex_average_rmin_lambda": 4.0,
                "outer_local_vertex_average_rmax_lambda": 12.0,
            },
        }

    def _fake_term_audit(**kwargs):
        rmax = float(kwargs["outer_local_refine_rmax_lambda"])
        flips = int(kwargs["local_edge_flip_steps"])
        avg = int(kwargs["outer_local_vertex_average_steps"])
        disk = 1.16 - 0.01 * flips - 0.005 * (avg - 2)
        outer_near = 1.13 - 0.02 * flips - 0.01 * (avg - 2)
        outer_far = 0.90 + 0.02 * flips + 0.005 * (avg - 2) + 0.003 * (rmax - 8.0)
        score = float(
            np.hypot(np.log(max(outer_near, 1e-18)), np.log(max(outer_far, 1e-18)))
        )
        return {
            "rows": [
                {
                    "internal_disk_ratio_mesh_over_theory": float(disk),
                    "internal_outer_near_ratio_mesh_over_theory_finite": float(
                        outer_near
                    ),
                    "internal_outer_far_ratio_mesh_over_theory_finite": float(
                        outer_far
                    ),
                    "section_score_internal_bands_finite_outer_l2_log": float(score),
                }
            ]
        }

    monkeypatch.setattr(mod, "run_flat_disk_one_leaflet_benchmark", _fake_benchmark)
    monkeypatch.setattr(mod, "run_flat_disk_kh_term_audit", _fake_term_audit)

    report = run_flat_disk_kh_discrete_quality_sweep(
        optimize_preset="kh_strict_outerfield_best",
        refine_level=2,
        outer_local_refine_rmax_lambda_values=(8.0, 10.0),
        local_edge_flip_steps_values=(0, 1),
        outer_local_vertex_average_steps_values=(2, 3),
    )
    assert report["meta"]["mode"] == "kh_discrete_quality_sweep"
    rows = report["rows"]
    assert len(rows) == 8
    baseline = next(
        row
        for row in rows
        if float(row["outer_local_refine_rmax_lambda"]) == 8.0
        and int(row["local_edge_flip_steps"]) == 0
        and int(row["outer_local_vertex_average_steps"]) == 2
    )
    assert abs(float(baseline["delta_theta_factor_vs_baseline"])) < 1e-12
    assert abs(float(baseline["delta_energy_factor_vs_baseline"])) < 1e-12
    assert abs(float(baseline["delta_section_score_vs_baseline"])) < 1e-12
    for row in rows:
        assert np.isfinite(float(row["theta_factor"]))
        assert np.isfinite(float(row["energy_factor"]))
        assert np.isfinite(
            float(row["section_score_internal_bands_finite_outer_l2_log"])
        )
        assert np.isfinite(float(row["disk_ratio"]))
        assert np.isfinite(float(row["outer_near_ratio"]))
        assert np.isfinite(float(row["outer_far_ratio"]))
        assert np.isfinite(float(row["delta_theta_factor_vs_baseline"]))
        assert np.isfinite(float(row["delta_energy_factor_vs_baseline"]))
        assert np.isfinite(float(row["delta_section_score_vs_baseline"]))
    assert int(report["selected_best"]["local_edge_flip_steps"]) == 1


@pytest.mark.regression
def test_flat_disk_kh_discrete_quality_safe_sweep_filters_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_error_source_audit as mod

    def _fake_quality_sweep(**kwargs):
        return {
            "meta": {
                "fixture": "tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml"
            },
            "rows": [
                {
                    "outer_local_refine_rmax_lambda": 8.0,
                    "outer_local_vertex_average_steps": 2,
                    "theta_factor": 1.03,
                    "energy_factor": 1.04,
                    "section_score_internal_bands_finite_outer_l2_log": 0.09,
                    "outer_far_ratio": 0.90,
                },
                {
                    "outer_local_refine_rmax_lambda": 10.0,
                    "outer_local_vertex_average_steps": 2,
                    "theta_factor": 1.03,
                    "energy_factor": 1.03,
                    "section_score_internal_bands_finite_outer_l2_log": 0.40,
                    "outer_far_ratio": 0.35,
                },
                {
                    "outer_local_refine_rmax_lambda": 9.0,
                    "outer_local_vertex_average_steps": 3,
                    "theta_factor": 1.02,
                    "energy_factor": 1.03,
                    "section_score_internal_bands_finite_outer_l2_log": 0.08,
                    "outer_far_ratio": 0.91,
                },
            ],
        }

    monkeypatch.setattr(
        mod, "run_flat_disk_kh_discrete_quality_sweep", _fake_quality_sweep
    )
    report = run_flat_disk_kh_discrete_quality_safe_sweep(
        optimize_preset="kh_strict_outerfield_best",
        refine_level=2,
        outer_far_floor=0.85,
    )
    assert report["meta"]["mode"] == "kh_discrete_quality_safe_sweep"
    assert float(report["meta"]["constraints"]["outer_far_floor"]) == 0.85
    rows = report["rows"]
    assert len(rows) == 3
    assert sum(1 for row in rows if bool(row["eligible"])) == 2
    assert report["selected_best"]["outer_local_refine_rmax_lambda"] == 9.0
    assert report["selected_best"]["outer_local_vertex_average_steps"] == 3


@pytest.mark.regression
def test_flat_disk_kh_discrete_quality_safe_refine_trend_emits_deltas_and_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_error_source_audit as mod

    def _fake_safe_sweep(**kwargs):
        lvl = int(kwargs["refine_level"])
        data = {
            2: {
                "realized_refine_level": 2,
                "theta_factor": 1.03,
                "energy_factor": 1.09,
                "section_score_internal_bands_finite_outer_l2_log": 0.12,
                "disk_ratio": 1.18,
                "outer_near_ratio": 1.11,
                "outer_far_ratio": 0.90,
                "eligible": True,
            },
            3: {
                "realized_refine_level": 3,
                "theta_factor": 1.02,
                "energy_factor": 1.06,
                "section_score_internal_bands_finite_outer_l2_log": 0.09,
                "disk_ratio": 1.15,
                "outer_near_ratio": 1.08,
                "outer_far_ratio": 0.93,
                "eligible": True,
            },
            4: {
                "realized_refine_level": 4,
                "theta_factor": 1.01,
                "energy_factor": 1.04,
                "section_score_internal_bands_finite_outer_l2_log": 0.08,
                "disk_ratio": 1.12,
                "outer_near_ratio": 1.05,
                "outer_far_ratio": 0.95,
                "eligible": True,
            },
        }[lvl]
        return {"selected_best": data}

    monkeypatch.setattr(
        mod, "run_flat_disk_kh_discrete_quality_safe_sweep", _fake_safe_sweep
    )
    report = run_flat_disk_kh_discrete_quality_safe_refine_trend(
        optimize_preset="kh_strict_outerfield_best",
        refine_levels=(2, 3, 4),
        outer_local_refine_rmax_lambda=8.0,
        outer_local_vertex_average_steps=2,
        outer_far_floor=0.85,
    )
    assert report["meta"]["mode"] == "kh_discrete_quality_safe_refine_trend"
    rows = report["trend"]["rows"]
    assert [int(r["refine_level"]) for r in rows] == [2, 3, 4]
    assert len(report["trend"]["deltas"]) == 2
    flags = report["trend"]["monotone_flags"]
    assert bool(flags["section_score_non_worsening"]) is True
    assert bool(flags["disk_abs_log_error_non_worsening"]) is True
    assert bool(flags["outer_near_abs_log_error_non_worsening"]) is True
    assert bool(flags["outer_far_abs_log_error_non_worsening"]) is True


@pytest.mark.regression
def test_flat_disk_kh_discrete_quality_safe_refine_trend_rejects_pinned_refine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_error_source_audit as mod

    def _fake_safe_sweep(**kwargs):
        return {
            "selected_best": {
                "realized_refine_level": 2,
                "theta_factor": 1.03,
                "energy_factor": 1.09,
                "section_score_internal_bands_finite_outer_l2_log": 0.12,
                "disk_ratio": 1.18,
                "outer_near_ratio": 1.11,
                "outer_far_ratio": 0.90,
                "eligible": True,
            }
        }

    monkeypatch.setattr(
        mod, "run_flat_disk_kh_discrete_quality_safe_sweep", _fake_safe_sweep
    )
    with pytest.raises(ValueError, match="realized refine_level"):
        run_flat_disk_kh_discrete_quality_safe_refine_trend(
            optimize_preset="kh_strict_outerfield_best",
            refine_levels=(2, 3),
            outer_local_refine_rmax_lambda=8.0,
            outer_local_vertex_average_steps=2,
            outer_far_floor=0.85,
        )
