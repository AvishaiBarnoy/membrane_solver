import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_kh_term_audit import (
    run_flat_disk_kh_outertail_characterization,
    run_flat_disk_kh_strict_preset_characterization,
    run_flat_disk_kh_strict_refinement_characterization,
    run_flat_disk_kh_term_audit,
    run_flat_disk_kh_term_audit_refine_sweep,
)


@pytest.mark.regression
def test_flat_disk_kh_term_audit_reports_finite_rows() -> None:
    report = run_flat_disk_kh_term_audit(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.0, 6.366e-4, 0.004),
        tilt_mass_mode_in="lumped",
    )

    assert report["meta"]["parameterization"] == "kh_physical"
    assert bool(report["meta"]["radial_projection_diagnostic"]) is False
    assert int(report["meta"]["outer_local_refine_steps"]) >= 0
    assert float(report["meta"]["outer_local_refine_rmin_lambda"]) >= 0.0
    assert float(report["meta"]["outer_local_refine_rmax_lambda"]) >= 0.0
    rows = report["rows"]
    assert len(rows) == 3

    for row in rows:
        assert np.isfinite(float(row["theta"]))
        assert np.isfinite(float(row["mesh_total"]))
        assert np.isfinite(float(row["mesh_contact"]))
        assert np.isfinite(float(row["mesh_internal"]))
        assert np.isfinite(float(row["theory_total"]))
        assert np.isfinite(float(row["theory_contact"]))
        assert np.isfinite(float(row["theory_internal"]))
        assert np.isfinite(float(row["total_error"]))
        assert np.isfinite(float(row["contact_error"]))
        assert np.isfinite(float(row["internal_error"]))
        assert np.isfinite(float(row["mesh_internal_disk"]))
        assert np.isfinite(float(row["mesh_internal_outer"]))
        assert np.isfinite(float(row["theory_internal_disk"]))
        assert np.isfinite(float(row["theory_internal_outer"]))
        assert np.isfinite(float(row["internal_disk_error"]))
        assert np.isfinite(float(row["internal_outer_error"]))
        assert np.isfinite(float(row["inner_tphi_over_trad_median"]))
        assert np.isfinite(float(row["outer_tphi_over_trad_median"]))
        assert np.isfinite(float(row["disk_core_tphi_abs_median"]))
        assert np.isfinite(float(row["disk_core_trad_abs_median"]))
        assert np.isfinite(float(row["disk_core_tphi_over_trad_median"]))
        assert np.isfinite(float(row["rim_band_tphi_abs_median"]))
        assert np.isfinite(float(row["rim_band_trad_abs_median"]))
        assert np.isfinite(float(row["rim_band_tphi_over_trad_median"]))
        assert np.isfinite(float(row["outer_near_tphi_abs_median"]))
        assert np.isfinite(float(row["outer_near_trad_abs_median"]))
        assert np.isfinite(float(row["outer_near_tphi_over_trad_median"]))
        assert np.isfinite(float(row["outer_far_tphi_abs_median"]))
        assert np.isfinite(float(row["outer_far_trad_abs_median"]))
        assert np.isfinite(float(row["outer_far_tphi_over_trad_median"]))
        assert np.isfinite(float(row["disk_core_hmax_over_hmin_mean"]))
        assert np.isfinite(float(row["rim_band_hmax_over_hmin_mean"]))
        assert np.isfinite(float(row["outer_near_hmax_over_hmin_mean"]))
        assert np.isfinite(float(row["outer_far_hmax_over_hmin_mean"]))
        assert np.isfinite(float(row["disk_core_edge_orientation_spread"]))
        assert np.isfinite(float(row["rim_band_edge_orientation_spread"]))
        assert np.isfinite(float(row["outer_near_edge_orientation_spread"]))
        assert np.isfinite(float(row["outer_far_edge_orientation_spread"]))
        corr_aspect = float(row["corr_hmax_over_hmin_vs_tphi_over_trad"])
        corr_orient = float(row["corr_orientation_spread_vs_tphi_over_trad"])
        assert np.isfinite(corr_aspect) or np.isnan(corr_aspect)
        assert np.isfinite(corr_orient) or np.isnan(corr_orient)
        assert np.isfinite(float(row["mesh_tilt_disk_core"]))
        assert np.isfinite(float(row["mesh_tilt_rim_band"]))
        assert np.isfinite(float(row["mesh_tilt_outer_near"]))
        assert np.isfinite(float(row["mesh_tilt_outer_far"]))
        assert np.isfinite(float(row["mesh_smooth_disk_core"]))
        assert np.isfinite(float(row["mesh_smooth_rim_band"]))
        assert np.isfinite(float(row["mesh_smooth_outer_near"]))
        assert np.isfinite(float(row["mesh_smooth_outer_far"]))
        assert np.isfinite(float(row["theory_tilt_disk_core"]))
        assert np.isfinite(float(row["theory_tilt_rim_band"]))
        assert np.isfinite(float(row["theory_tilt_outer_near"]))
        assert np.isfinite(float(row["theory_tilt_outer_far"]))
        assert np.isfinite(float(row["theory_smooth_disk_core"]))
        assert np.isfinite(float(row["theory_smooth_rim_band"]))
        assert np.isfinite(float(row["theory_smooth_outer_near"]))
        assert np.isfinite(float(row["theory_smooth_outer_far"]))
        assert np.isfinite(float(row["theory_tilt_outer_near_finite"]))
        assert np.isfinite(float(row["theory_tilt_outer_far_finite"]))
        assert np.isfinite(float(row["theory_smooth_outer_near_finite"]))
        assert np.isfinite(float(row["theory_smooth_outer_far_finite"]))
        assert np.isfinite(float(row["theory_internal_disk_core"]))
        assert np.isfinite(float(row["theory_internal_rim_band"]))
        assert np.isfinite(float(row["theory_internal_outer_near"]))
        assert np.isfinite(float(row["theory_internal_outer_far"]))
        assert np.isfinite(float(row["theory_internal_outer_near_finite"]))
        assert np.isfinite(float(row["theory_internal_outer_far_finite"]))
        assert np.isfinite(float(row["theory_outer_r_max"]))
        assert np.isfinite(float(row["theory_internal_total_from_bands"]))
        assert np.isfinite(float(row["theory_internal_bands_minus_closed_form"]))
        assert bool(row["radial_projection_diagnostic"]) is False
        assert np.isnan(float(row["proj_radial_mesh_internal"]))
        assert np.isnan(float(row["proj_radial_mesh_internal_outer_near"]))
        assert np.isnan(float(row["proj_radial_internal_outer_near_abs_error"]))
        assert np.isnan(
            float(row["proj_radial_internal_outer_near_abs_error_delta_vs_unprojected"])
        )
        if float(row["section_score_internal_split_count"]) > 0.0:
            assert np.isfinite(float(row["section_score_internal_split_l2_log"]))
            assert np.isfinite(float(row["section_score_internal_split_max_abs_log"]))
        if float(row["section_score_internal_bands_count"]) > 0.0:
            assert np.isfinite(float(row["section_score_internal_bands_l2_log"]))
            assert np.isfinite(float(row["section_score_internal_bands_max_abs_log"]))
        if float(row["section_score_internal_bands_finite_outer_count"]) > 0.0:
            assert np.isfinite(
                float(row["section_score_internal_bands_finite_outer_l2_log"])
            )
            assert np.isfinite(
                float(row["section_score_internal_bands_finite_outer_max_abs_log"])
            )
        if float(row["section_score_tilt_bands_count"]) > 0.0:
            assert np.isfinite(float(row["section_score_tilt_bands_l2_log"]))
            assert np.isfinite(float(row["section_score_tilt_bands_max_abs_log"]))
        if float(row["section_score_smooth_bands_count"]) > 0.0:
            assert np.isfinite(float(row["section_score_smooth_bands_l2_log"]))
            assert np.isfinite(float(row["section_score_smooth_bands_max_abs_log"]))
        if float(row["section_score_all_terms_count"]) > 0.0:
            assert np.isfinite(float(row["section_score_all_terms_l2_log"]))
            assert np.isfinite(float(row["section_score_all_terms_max_abs_log"]))
        assert float(row["mesh_internal_total_from_regions"]) == pytest.approx(
            float(row["mesh_internal"]), rel=0.0, abs=1e-12
        )
        assert (
            float(row["mesh_internal_disk_core"])
            + float(row["mesh_internal_rim_band"])
            + float(row["mesh_internal_outer_near"])
            + float(row["mesh_internal_outer_far"])
        ) == pytest.approx(float(row["mesh_internal"]), rel=0.0, abs=1e-10)
        assert (
            float(row["theory_internal_disk_core"])
            + float(row["theory_internal_rim_band"])
            + float(row["theory_internal_outer_near"])
            + float(row["theory_internal_outer_far"])
        ) == pytest.approx(
            float(row["theory_internal_total_from_bands"]), rel=0.0, abs=1e-12
        )

    theta0 = rows[0]
    assert float(theta0["theta"]) == pytest.approx(0.0, abs=1e-18)
    assert float(theta0["theory_contact"]) == pytest.approx(0.0, abs=1e-18)
    assert float(theta0["mesh_contact"]) == pytest.approx(0.0, abs=1e-15)
    assert int(theta0["rim_samples"]) > 0

    resolution = report["resolution"]
    assert int(resolution["rim_edge_count"]) > 0
    assert np.isfinite(float(resolution["rim_h_over_lambda_median"]))


@pytest.mark.regression
def test_flat_disk_kh_term_audit_radial_projection_diagnostic_emits_finite_rows() -> (
    None
):
    report = run_flat_disk_kh_term_audit(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.138,),
        tilt_mass_mode_in="consistent",
        radial_projection_diagnostic=True,
    )
    assert bool(report["meta"]["radial_projection_diagnostic"]) is True
    row = report["rows"][0]
    assert bool(row["radial_projection_diagnostic"]) is True
    assert np.isfinite(float(row["proj_radial_mesh_internal"]))
    assert np.isfinite(float(row["proj_radial_mesh_internal_disk"]))
    assert np.isfinite(float(row["proj_radial_mesh_internal_outer"]))
    assert np.isfinite(float(row["proj_radial_mesh_internal_disk_core"]))
    assert np.isfinite(float(row["proj_radial_mesh_internal_rim_band"]))
    assert np.isfinite(float(row["proj_radial_mesh_internal_outer_near"]))
    assert np.isfinite(float(row["proj_radial_mesh_internal_outer_far"]))
    assert np.isfinite(float(row["proj_radial_internal_disk_core_abs_error"]))
    assert np.isfinite(float(row["proj_radial_internal_rim_band_abs_error"]))
    assert np.isfinite(float(row["proj_radial_internal_outer_near_abs_error"]))
    assert np.isfinite(float(row["proj_radial_internal_outer_far_abs_error"]))
    assert np.isfinite(
        float(row["proj_radial_internal_disk_core_abs_error_delta_vs_unprojected"])
    )
    assert np.isfinite(
        float(row["proj_radial_internal_rim_band_abs_error_delta_vs_unprojected"])
    )
    assert np.isfinite(
        float(row["proj_radial_internal_outer_near_abs_error_delta_vs_unprojected"])
    )
    assert np.isfinite(
        float(row["proj_radial_internal_outer_far_abs_error_delta_vs_unprojected"])
    )


@pytest.mark.regression
def test_flat_disk_kh_term_audit_refine_sweep_shape() -> None:
    report = run_flat_disk_kh_term_audit_refine_sweep(
        refine_levels=(1, 2),
        theta_values=(0.0, 6.366e-4),
    )
    assert report["meta"]["mode"] == "refine_sweep"
    runs = report["runs"]
    assert len(runs) == 2
    for idx, run in enumerate(runs):
        assert int(run["meta"]["refine_level"]) == (idx + 1)
        assert run["meta"]["theory_model"] == "kh_physical_strict_kh"
        assert int(run["meta"]["outer_local_refine_steps"]) >= 0
        assert len(run["rows"]) == 2


@pytest.mark.regression
def test_flat_disk_kh_term_audit_local_rim_refine_changes_resolution() -> None:
    theta = 0.004
    base = run_flat_disk_kh_term_audit(
        refine_level=1,
        theta_values=(theta,),
        rim_local_refine_steps=0,
        rim_local_refine_band_lambda=0.0,
    )
    local = run_flat_disk_kh_term_audit(
        refine_level=1,
        theta_values=(theta,),
        rim_local_refine_steps=1,
        rim_local_refine_band_lambda=4.0,
    )

    base_row = base["rows"][0]
    local_row = local["rows"][0]
    assert np.isfinite(float(base_row["internal_outer_ratio_mesh_over_theory"]))
    assert np.isfinite(float(local_row["internal_outer_ratio_mesh_over_theory"]))
    base_h = float(base["resolution"]["rim_h_over_lambda_median"])
    local_h = float(local["resolution"]["rim_h_over_lambda_median"])
    assert local_h < base_h


@pytest.mark.regression
def test_flat_disk_kh_term_audit_internal_region_ratios_near_unity_under_strict_mesh() -> (
    None
):
    report = run_flat_disk_kh_term_audit(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.135,),
        tilt_mass_mode_in="consistent",
        rim_local_refine_steps=2,
        rim_local_refine_band_lambda=8.0,
    )
    row = report["rows"][0]
    disk_ratio = float(row["internal_disk_ratio_mesh_over_theory"])
    outer_ratio = float(row["internal_outer_ratio_mesh_over_theory"])
    assert disk_ratio <= 1.30
    assert disk_ratio >= 0.80
    assert outer_ratio <= 1.30
    assert outer_ratio >= 0.80


@pytest.mark.regression
def test_flat_disk_kh_term_audit_section_scores_deterministic_under_strict_mesh() -> (
    None
):
    report = run_flat_disk_kh_term_audit(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.135,),
        tilt_mass_mode_in="consistent",
        rim_local_refine_steps=2,
        rim_local_refine_band_lambda=8.0,
    )
    row = report["rows"][0]
    assert float(row["section_score_internal_split_l2_log"]) == pytest.approx(
        0.1429931326, rel=1e-6, abs=1e-10
    )
    assert float(row["section_score_internal_bands_l2_log"]) == pytest.approx(
        0.3829001961, rel=1e-6, abs=1e-10
    )
    assert float(row["section_score_tilt_bands_l2_log"]) == pytest.approx(
        0.8191650085, rel=1e-6, abs=1e-10
    )
    assert float(row["section_score_smooth_bands_l2_log"]) == pytest.approx(
        0.3391084545, rel=1e-6, abs=1e-10
    )
    assert float(row["section_score_all_terms_l2_log"]) == pytest.approx(
        0.6269074314, rel=1e-6, abs=1e-10
    )


@pytest.mark.regression
def test_flat_disk_kh_term_audit_finite_outer_reference_matches_infinite_at_rmax12() -> (
    None
):
    report = run_flat_disk_kh_term_audit(
        refine_level=2,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.138,),
        tilt_mass_mode_in="consistent",
        rim_local_refine_steps=1,
        rim_local_refine_band_lambda=4.0,
    )
    row = report["rows"][0]
    assert float(row["theory_outer_r_max"]) == pytest.approx(12.0, rel=0.0, abs=1e-10)
    assert float(row["theory_internal_outer_near_finite"]) == pytest.approx(
        float(row["theory_internal_outer_near"]), rel=0.0, abs=1e-15
    )
    assert float(row["theory_internal_outer_far_finite"]) == pytest.approx(
        float(row["theory_internal_outer_far"]), rel=0.0, abs=1e-15
    )


@pytest.mark.regression
def test_flat_disk_kh_term_audit_outerfield_tailmatch_improves_outer_far_section() -> (
    None
):
    quality = run_flat_disk_kh_term_audit(
        refine_level=2,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.138,),
        tilt_mass_mode_in="consistent",
        rim_local_refine_steps=1,
        rim_local_refine_band_lambda=3.0,
        outer_local_refine_steps=1,
        outer_local_refine_rmin_lambda=1.0,
        outer_local_refine_rmax_lambda=8.0,
        local_edge_flip_steps=1,
        local_edge_flip_rmin_lambda=2.0,
        local_edge_flip_rmax_lambda=6.0,
    )["rows"][0]
    tailmatch = run_flat_disk_kh_term_audit(
        refine_level=2,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.138,),
        tilt_mass_mode_in="consistent",
        rim_local_refine_steps=1,
        rim_local_refine_band_lambda=3.0,
        outer_local_refine_steps=1,
        outer_local_refine_rmin_lambda=1.0,
        outer_local_refine_rmax_lambda=8.0,
        local_edge_flip_steps=1,
        local_edge_flip_rmin_lambda=0.5,
        local_edge_flip_rmax_lambda=8.0,
    )["rows"][0]

    q_near = float(quality["internal_outer_near_ratio_mesh_over_theory"])
    q_far = float(quality["internal_outer_far_ratio_mesh_over_theory"])
    t_near = float(tailmatch["internal_outer_near_ratio_mesh_over_theory"])
    t_far = float(tailmatch["internal_outer_far_ratio_mesh_over_theory"])
    q_score = float(np.hypot(np.log(max(q_near, 1e-18)), np.log(max(q_far, 1e-18))))
    t_score = float(np.hypot(np.log(max(t_near, 1e-18)), np.log(max(t_far, 1e-18))))

    assert t_score <= (q_score + 2.0e-4)
    assert abs(t_far - 1.0) <= abs(q_far - 1.0)
    assert abs(t_near - 1.0) <= (abs(q_near - 1.0) + 2.0e-4)
    assert t_far >= 0.85
    assert t_near <= 1.13
    assert t_score <= 0.25


@pytest.mark.regression
def test_flat_disk_kh_strict_refinement_characterization_emits_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_term_audit as audit_mod

    def _fake_run_flat_disk_one_leaflet_benchmark(**kwargs):
        refine_level = int(kwargs.get("refine_level", 1))
        rim_steps = int(kwargs.get("rim_local_refine_steps", 0))
        if refine_level == 1 and rim_steps == 1:
            return {
                "parity": {
                    "theta_factor": 1.10,
                    "energy_factor": 1.10,
                    "meets_factor_2": True,
                },
                "optimize": {"optimize_steps": 120, "optimize_inner_steps": 20},
            }
        return {
            "parity": {
                "theta_factor": 1.30,
                "energy_factor": 1.20,
                "meets_factor_2": True,
            },
            "optimize": {"optimize_steps": 120, "optimize_inner_steps": 20},
        }

    def _fake_run_flat_disk_kh_term_audit(**kwargs):
        rim_steps = int(kwargs.get("rim_local_refine_steps", 0))
        rim_h = 0.6 if rim_steps > 0 else 1.2
        return {"resolution": {"rim_h_over_lambda_median": rim_h}}

    monkeypatch.setattr(
        "tools.reproduce_flat_disk_one_leaflet.run_flat_disk_one_leaflet_benchmark",
        _fake_run_flat_disk_one_leaflet_benchmark,
    )
    monkeypatch.setattr(
        audit_mod, "run_flat_disk_kh_term_audit", _fake_run_flat_disk_kh_term_audit
    )
    monkeypatch.setattr(audit_mod, "perf_counter", lambda: 1.0)

    report = run_flat_disk_kh_strict_refinement_characterization(
        optimize_preset="kh_wide",
        global_refine_levels=(1,),
        rim_local_steps=(0, 1),
    )
    rows = report["rows"]
    assert len(rows) == 2
    for row in rows:
        assert int(row["refine_level"]) >= 1
        assert int(row["rim_local_refine_steps"]) >= 0
        assert np.isfinite(float(row["theta_factor"]))
        assert np.isfinite(float(row["energy_factor"]))
        assert np.isfinite(float(row["balanced_parity_score"]))
        assert np.isfinite(float(row["rim_h_over_lambda_median"]))
    best = report["selected_best"]
    assert np.isfinite(float(best["theta_factor"]))
    assert int(best["rim_local_refine_steps"]) == 1


@pytest.mark.regression
def test_flat_disk_kh_strict_preset_characterization_emits_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_term_audit as audit_mod

    def _fake_run_flat_disk_one_leaflet_benchmark(**kwargs):
        preset = str(kwargs.get("optimize_preset"))
        if preset == "kh_strict_fast":
            return {
                "parity": {
                    "theta_factor": 1.11,
                    "energy_factor": 1.10,
                    "meets_factor_2": True,
                },
                "optimize": {
                    "optimize_steps": 30,
                    "optimize_inner_steps": 14,
                    "optimize_seconds": 10.0,
                },
                "mesh": {
                    "profile": {"rim_abs_median": 1.0},
                    "rim_continuity": {"jump_abs_median": 0.26},
                    "leakage": {"outer_tphi_over_trad_median": 0.88},
                },
            }
        if preset == "kh_strict_continuity":
            return {
                "parity": {
                    "theta_factor": 1.04,
                    "energy_factor": 1.08,
                    "meets_factor_2": True,
                },
                "optimize": {
                    "optimize_steps": 30,
                    "optimize_inner_steps": 14,
                    "optimize_seconds": 45.0,
                },
                "mesh": {
                    "profile": {"rim_abs_median": 1.0},
                    "rim_continuity": {"jump_abs_median": 0.18},
                    "leakage": {"outer_tphi_over_trad_median": 0.07},
                },
            }
        if preset == "kh_strict_balanced":
            return {
                "parity": {
                    "theta_factor": 1.06,
                    "energy_factor": 1.09,
                    "meets_factor_2": True,
                },
                "optimize": {
                    "optimize_steps": 34,
                    "optimize_inner_steps": 16,
                    "optimize_seconds": 20.0,
                },
                "mesh": {
                    "profile": {"rim_abs_median": 1.0},
                    "rim_continuity": {"jump_abs_median": 0.22},
                    "leakage": {"outer_tphi_over_trad_median": 0.33},
                },
            }
        if preset == "kh_strict_energy_tight":
            return {
                "parity": {
                    "theta_factor": 1.03,
                    "energy_factor": 1.08,
                    "meets_factor_2": True,
                },
                "optimize": {
                    "optimize_steps": 30,
                    "optimize_inner_steps": 14,
                    "optimize_seconds": 25.0,
                },
                "mesh": {
                    "profile": {"rim_abs_median": 1.0},
                    "rim_continuity": {"jump_abs_median": 0.20},
                    "leakage": {"outer_tphi_over_trad_median": 0.20},
                },
            }
        return {
            "parity": {
                "theta_factor": 1.13,
                "energy_factor": 1.11,
                "meets_factor_2": True,
            },
            "optimize": {
                "optimize_steps": 30,
                "optimize_inner_steps": 14,
                "optimize_seconds": 46.0,
            },
            "mesh": {
                "profile": {"rim_abs_median": 1.0},
                "rim_continuity": {"jump_abs_median": 0.26},
                "leakage": {"outer_tphi_over_trad_median": 0.88},
            },
        }

    monkeypatch.setattr(
        "tools.reproduce_flat_disk_one_leaflet.run_flat_disk_one_leaflet_benchmark",
        _fake_run_flat_disk_one_leaflet_benchmark,
    )
    monkeypatch.setattr(audit_mod, "perf_counter", lambda: 1.0)

    report = run_flat_disk_kh_strict_preset_characterization(
        optimize_presets=(
            "kh_strict_fast",
            "kh_strict_balanced",
            "kh_strict_energy_tight",
            "kh_strict_continuity",
            "kh_strict_robust",
        ),
        refine_level=1,
        rim_local_refine_steps=1,
        rim_local_refine_band_lambda=4.0,
    )
    assert report["meta"]["mode"] == "strict_preset_characterization"
    rows = report["rows"]
    assert len(rows) == 5
    for row in rows:
        assert np.isfinite(float(row["theta_factor"]))
        assert np.isfinite(float(row["energy_factor"]))
        assert np.isfinite(float(row["balanced_parity_score"]))
        assert np.isfinite(float(row["runtime_seconds"]))
        assert np.isfinite(float(row["optimize_seconds"]))
        assert np.isfinite(float(row["rim_jump_ratio"]))
        assert np.isfinite(float(row["outer_tphi_over_trad_median"]))
    best = report["selected_best"]
    assert best["optimize_preset"] == "kh_strict_energy_tight"


@pytest.mark.regression
def test_flat_disk_kh_outertail_characterization_emits_ranked_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_term_audit as audit_mod

    def _fake_run_flat_disk_one_leaflet_benchmark(**kwargs):
        rmax = float(kwargs.get("outer_local_refine_rmax_lambda", 8.0))
        theta_factor = 1.02 if rmax >= 10.0 else 1.03
        energy_factor = 1.04 if rmax >= 10.0 else 1.05
        return {
            "parity": {
                "theta_factor": theta_factor,
                "energy_factor": energy_factor,
                "meets_factor_2": True,
            },
            "mesh": {"theta_star": 0.138},
        }

    def _fake_run_flat_disk_kh_term_audit(**kwargs):
        rmax = float(kwargs.get("outer_local_refine_rmax_lambda", 8.0))
        near_ratio = 1.03 if rmax >= 10.0 else 1.15
        far_ratio = 0.96 if rmax >= 10.0 else 0.70
        return {
            "rows": [
                {
                    "internal_outer_near_ratio_mesh_over_theory_finite": near_ratio,
                    "internal_outer_far_ratio_mesh_over_theory_finite": far_ratio,
                    "mesh_internal_outer_near": 0.05,
                    "mesh_internal_outer_far": 0.001,
                    "theory_internal_outer_near_finite": 0.05,
                    "theory_internal_outer_far_finite": 0.001,
                }
            ]
        }

    monkeypatch.setattr(
        "tools.reproduce_flat_disk_one_leaflet.run_flat_disk_one_leaflet_benchmark",
        _fake_run_flat_disk_one_leaflet_benchmark,
    )
    monkeypatch.setattr(
        audit_mod, "run_flat_disk_kh_term_audit", _fake_run_flat_disk_kh_term_audit
    )

    report = run_flat_disk_kh_outertail_characterization(
        optimize_presets=("kh_strict_outerfield_tight",),
        outer_local_refine_steps_values=(1,),
        outer_local_refine_rmax_lambda_values=(8.0, 10.0),
    )
    assert len(report["rows"]) == 2
    best = report["selected_best"]
    assert float(best["outer_local_refine_rmax_lambda"]) == pytest.approx(10.0)
