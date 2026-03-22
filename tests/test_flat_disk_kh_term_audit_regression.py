import os
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT = Path(__file__).resolve().parent.parent

from tools.diagnostics.flat_disk_kh_term_audit import (
    run_flat_disk_kh_discrete_tilt_matrix,
    run_flat_disk_kh_disk_refinement_characterization,
    run_flat_disk_kh_outerfield_averaged_sweep,
    run_flat_disk_kh_outertail_characterization,
    run_flat_disk_kh_strict_preset_characterization,
    run_flat_disk_kh_strict_refinement_characterization,
    run_flat_disk_kh_term_audit,
    run_flat_disk_kh_term_audit_refine_sweep,
)


def _freeze_cache_value(value):
    """Return an lru-cache-safe representation for audit helper kwargs."""
    if isinstance(value, tuple):
        return tuple(_freeze_cache_value(v) for v in value)
    if isinstance(value, list):
        return tuple(_freeze_cache_value(v) for v in value)
    return value


@lru_cache(maxsize=64)
def _cached_term_audit_report(items):
    """Cache repeated flat-disk audit reports within this test module."""
    return run_flat_disk_kh_term_audit(**dict(items))


def _term_audit_report(**kwargs):
    """Return a defensive copy of a cached flat-disk term-audit report."""
    items = tuple(
        sorted((key, _freeze_cache_value(value)) for key, value in kwargs.items())
    )
    return deepcopy(_cached_term_audit_report(items))


def _strict_mesh_report() -> dict:
    """Return the shared strict-mesh audit report used by adjacent assertions."""
    return _term_audit_report(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.135,),
        tilt_mass_mode_in="consistent",
        rim_local_refine_steps=2,
        rim_local_refine_band_lambda=8.0,
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
    assert report["meta"]["partition_mode"] == "centroid"
    assert bool(report["meta"]["radial_projection_diagnostic"]) is False
    assert int(report["meta"]["outer_local_refine_steps"]) >= 0
    assert float(report["meta"]["outer_local_refine_rmin_lambda"]) >= 0.0
    assert float(report["meta"]["outer_local_refine_rmax_lambda"]) >= 0.0
    assert int(report["meta"]["outer_local_vertex_average_steps"]) >= 0
    assert float(report["meta"]["outer_local_vertex_average_rmin_lambda"]) >= 0.0
    assert float(report["meta"]["outer_local_vertex_average_rmax_lambda"]) >= 0.0
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
def test_flat_disk_kh_term_audit_fractional_partition_mode_runs() -> None:
    report = run_flat_disk_kh_term_audit(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.0,),
        tilt_mass_mode_in="consistent",
        partition_mode="fractional",
    )
    assert report["meta"]["partition_mode"] == "fractional"
    row = report["rows"][0]
    assert np.isfinite(float(row["mesh_internal_disk_core"]))
    assert np.isfinite(float(row["mesh_internal_outer_near"]))
    assert np.isfinite(float(row["mesh_internal_outer_far"]))
    assert (
        float(row["mesh_internal_disk_core"])
        + float(row["mesh_internal_rim_band"])
        + float(row["mesh_internal_outer_near"])
        + float(row["mesh_internal_outer_far"])
    ) == pytest.approx(float(row["mesh_internal"]), rel=0.0, abs=1e-8)


@pytest.mark.regression
def test_flat_disk_kh_term_audit_relax_projection_controls_emit_metadata() -> None:
    report = run_flat_disk_kh_term_audit(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.138,),
        tilt_mass_mode_in="consistent",
        tilt_projection_cadence="per_pass",
        tilt_projection_interval=3,
        tilt_post_relax_inner_steps=20,
        tilt_post_relax_step_size=0.02,
        tilt_post_relax_passes=2,
    )
    assert report["meta"]["tilt_projection_cadence"] == "per_pass"
    assert int(report["meta"]["tilt_projection_interval"]) == 3
    assert int(report["meta"]["tilt_post_relax_inner_steps"]) == 20
    assert float(report["meta"]["tilt_post_relax_step_size"]) == pytest.approx(0.02)
    assert int(report["meta"]["tilt_post_relax_passes"]) == 2


@pytest.mark.regression
def test_flat_disk_kh_term_audit_divergence_mode_emits_metadata() -> None:
    report = run_flat_disk_kh_term_audit(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.0,),
        tilt_mass_mode_in="consistent",
        tilt_divergence_mode_in="vertex_recovered",
    )
    assert report["meta"]["tilt_divergence_mode_in"] == "vertex_recovered"


@pytest.mark.regression
def test_flat_disk_kh_term_audit_transport_and_outer_mass_emit_metadata() -> None:
    report = run_flat_disk_kh_term_audit(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.0,),
        tilt_transport_model="connection_v1",
        tilt_mass_mode_out="consistent",
    )
    assert report["meta"]["tilt_transport_model"] == "connection_v1"
    assert report["meta"]["tilt_mass_mode_out"] == "consistent"


@pytest.mark.regression
def test_flat_disk_kh_term_audit_inner_coupled_update_mode_emits_metadata_and_row_stats() -> (
    None
):
    report = run_flat_disk_kh_term_audit(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.138,),
        inner_coupled_update_mode="rim_matched_radial_continuation_v1",
    )
    assert report["meta"]["inner_coupled_update_mode"] == (
        "rim_matched_radial_continuation_v1"
    )
    row = report["rows"][0]
    assert bool(row["inner_coupled_update_enabled"]) is True
    assert row["inner_coupled_update_mode"] == "rim_matched_radial_continuation_v1"
    assert int(row["inner_coupled_candidate_row_count"]) >= 0
    assert int(row["inner_coupled_capped_row_count"]) >= 0
    assert int(row["inner_coupled_rim_row_count"]) >= 0
    assert float(row["inner_coupled_cap_magnitude"]) >= 0.0


@pytest.mark.regression
def test_flat_disk_kh_term_audit_theta_relax_controls_emit_metadata() -> None:
    report = run_flat_disk_kh_term_audit(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.0, 0.138),
        tilt_mass_mode_in="consistent",
        theta_relax_mode="adaptive",
        theta_relax_max_repeats=3,
        theta_relax_energy_abs_tol=1.0e-6,
        theta_relax_plateau_patience=1,
    )
    assert report["meta"]["theta_relax_mode"] == "adaptive"
    assert int(report["meta"]["theta_relax_max_repeats"]) == 3
    assert float(report["meta"]["theta_relax_energy_abs_tol"]) == pytest.approx(1.0e-6)
    assert int(report["meta"]["theta_relax_plateau_patience"]) == 1
    for row in report["rows"]:
        assert row["theta_relax_mode"] == "adaptive"
        assert int(row["theta_relax_max_repeats"]) == 3
        assert 1 <= int(row["theta_relax_repeats_applied"]) <= 3
        assert isinstance(bool(row["theta_relax_converged"]), bool)


@pytest.mark.regression
def test_flat_disk_kh_term_audit_staged_v2_relax_path_matches_dirty_branch() -> None:
    report = run_flat_disk_kh_term_audit(
        refine_level=3,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.138,),
        tilt_mass_mode_in="consistent",
        tilt_divergence_mode_in="native",
        rim_local_refine_steps=1,
        rim_local_refine_band_lambda=1.5,
        outer_local_refine_steps=1,
        outer_local_refine_rmin_lambda=0.5,
        outer_local_refine_rmax_lambda=8.0,
        outer_local_vertex_average_steps=5,
        outer_local_vertex_average_rmin_lambda=8.0,
        outer_local_vertex_average_rmax_lambda=12.0,
        isotropy_pass="off",
        isotropy_iters=0,
        partition_mode="fractional",
        ratio_version="v2",
        theory_outer_mode="finite_bvp",
        parity_target="p10",
        axial_symmetry_gate="monitor",
        theta_relax_mode="adaptive",
        theta_relax_max_repeats=5,
        theta_relax_energy_abs_tol=1.0e-10,
        theta_relax_plateau_patience=2,
        tilt_projection_cadence="per_step",
        tilt_projection_interval=1,
        tilt_post_relax_inner_steps=40,
        tilt_post_relax_step_size=0.005,
        tilt_post_relax_passes=1,
    )

    row = report["rows"][0]
    assert int(row["theta_relax_repeats_applied"]) == 5
    assert int(row["tilt_projection_apply_count"]) == 40
    assert float(row["internal_disk_ratio_mesh_over_theory_v2"]) == pytest.approx(
        0.9839900707418473,
        rel=1.0e-6,
        abs=1.0e-9,
    )
    assert float(row["internal_outer_near_ratio_mesh_over_theory_v2"]) == pytest.approx(
        0.9794560277853616,
        rel=1.0e-6,
        abs=1.0e-9,
    )
    assert float(row["internal_outer_far_ratio_mesh_over_theory_v2"]) == pytest.approx(
        0.9681785253676642,
        rel=1.0e-6,
        abs=1.0e-9,
    )
    assert bool(row["meets_10pct_v2"]) is True
    assert bool(row["meets_parity_target_v2"]) is True


@pytest.mark.regression
def test_flat_disk_kh_term_audit_invalid_partition_mode_raises() -> None:
    with pytest.raises(ValueError, match="partition_mode"):
        run_flat_disk_kh_term_audit(
            refine_level=1,
            theta_values=(0.0,),
            partition_mode="bad_mode",
        )
    with pytest.raises(
        ValueError,
        match="tilt_transport_model must be 'ambient_v1' or 'connection_v1'",
    ):
        run_flat_disk_kh_term_audit(
            refine_level=1,
            theta_values=(0.0,),
            tilt_transport_model="bad_mode",
        )
    with pytest.raises(
        ValueError,
        match="tilt_mass_mode_out must be 'auto', 'lumped', or 'consistent'",
    ):
        run_flat_disk_kh_term_audit(
            refine_level=1,
            theta_values=(0.0,),
            tilt_mass_mode_out="bad_mode",
        )
    with pytest.raises(
        ValueError,
        match="inner_coupled_update_mode must be 'off' or 'rim_matched_radial_continuation_v1'",
    ):
        run_flat_disk_kh_term_audit(
            refine_level=1,
            theta_values=(0.0,),
            inner_coupled_update_mode="bad_mode",
        )


@pytest.mark.regression
def test_flat_disk_kh_term_audit_theta_relax_controls_validate() -> None:
    with pytest.raises(
        ValueError, match="theta_relax_mode must be 'fixed' or 'adaptive'"
    ):
        run_flat_disk_kh_term_audit(
            refine_level=1,
            theta_values=(0.0,),
            theta_relax_mode="bad_mode",
        )
    with pytest.raises(ValueError, match="theta_relax_max_repeats must be >= 1"):
        run_flat_disk_kh_term_audit(
            refine_level=1,
            theta_values=(0.0,),
            theta_relax_max_repeats=0,
        )


@pytest.mark.regression
def test_flat_disk_kh_term_audit_theory_outer_mode_emits_v2_metadata() -> None:
    report = run_flat_disk_kh_term_audit(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.138,),
        ratio_version="v2",
        theory_outer_mode="finite_bvp",
    )
    assert report["meta"]["theory_outer_mode_requested"] == "finite_bvp"
    assert report["meta"]["theory_outer_mode_v2_effective"] == "finite_bvp"
    assert report["meta"]["v2_ratio_semantics"] == "strict_raw"
    row = report["rows"][0]
    assert row["theory_outer_mode_v2"] == "finite_bvp"
    assert np.isfinite(float(row["theory_outer_r_max_v2"]))
    assert np.isfinite(float(row["internal_outer_near_ratio_mesh_over_theory_v2_raw"]))
    assert np.isfinite(float(row["internal_outer_far_ratio_mesh_over_theory_v2_raw"]))


@pytest.mark.regression
def test_flat_disk_kh_term_audit_theory_outer_mode_validates() -> None:
    with pytest.raises(
        ValueError, match="theory_outer_mode must be 'infinite' or 'finite_bvp'"
    ):
        run_flat_disk_kh_term_audit(
            refine_level=1,
            theta_values=(0.0,),
            theory_outer_mode="bad_mode",
        )


@pytest.mark.regression
def test_flat_disk_kh_term_audit_refine_sweep_relax_projection_controls_emit_metadata() -> (
    None
):
    report = run_flat_disk_kh_term_audit_refine_sweep(
        refine_levels=(1, 2),
        theta_values=(0.0,),
        tilt_projection_cadence="per_pass",
        tilt_projection_interval=3,
        tilt_post_relax_inner_steps=20,
        tilt_post_relax_step_size=0.02,
        tilt_post_relax_passes=2,
    )
    assert report["meta"]["tilt_projection_cadence"] == "per_pass"
    assert int(report["meta"]["tilt_projection_interval"]) == 3
    assert int(report["meta"]["tilt_post_relax_inner_steps"]) == 20
    assert float(report["meta"]["tilt_post_relax_step_size"]) == pytest.approx(0.02)
    assert int(report["meta"]["tilt_post_relax_passes"]) == 2
    for run in report["runs"]:
        assert run["meta"]["tilt_projection_cadence"] == "per_pass"
        assert int(run["meta"]["tilt_projection_interval"]) == 3
        assert int(run["meta"]["tilt_post_relax_inner_steps"]) == 20
        assert float(run["meta"]["tilt_post_relax_step_size"]) == pytest.approx(0.02)
        assert int(run["meta"]["tilt_post_relax_passes"]) == 2


@pytest.mark.regression
def test_flat_disk_kh_term_audit_refine_sweep_theta_relax_controls_emit_metadata() -> (
    None
):
    report = run_flat_disk_kh_term_audit_refine_sweep(
        refine_levels=(1, 2),
        theta_values=(0.0,),
        theta_relax_mode="adaptive",
        theta_relax_max_repeats=2,
        theta_relax_energy_abs_tol=1.0e-6,
        theta_relax_plateau_patience=1,
    )
    assert report["meta"]["theta_relax_mode"] == "adaptive"
    assert int(report["meta"]["theta_relax_max_repeats"]) == 2
    assert float(report["meta"]["theta_relax_energy_abs_tol"]) == pytest.approx(1.0e-6)
    assert int(report["meta"]["theta_relax_plateau_patience"]) == 1
    for run in report["runs"]:
        assert run["meta"]["theta_relax_mode"] == "adaptive"
        assert int(run["meta"]["theta_relax_max_repeats"]) == 2


@pytest.mark.regression
def test_flat_disk_kh_term_audit_refine_sweep_divergence_mode_emits_metadata() -> None:
    report = run_flat_disk_kh_term_audit_refine_sweep(
        refine_levels=(1, 2),
        theta_values=(0.0,),
        tilt_divergence_mode_in="vertex_recovered",
    )
    assert report["meta"]["tilt_divergence_mode_in"] == "vertex_recovered"
    for run in report["runs"]:
        assert run["meta"]["tilt_divergence_mode_in"] == "vertex_recovered"


@pytest.mark.regression
def test_flat_disk_kh_term_audit_refine_sweep_transport_and_outer_mass_emit_metadata() -> (
    None
):
    report = run_flat_disk_kh_term_audit_refine_sweep(
        refine_levels=(1, 2),
        theta_values=(0.0,),
        tilt_transport_model="connection_v1",
        tilt_mass_mode_out="consistent",
    )
    assert report["meta"]["tilt_transport_model"] == "connection_v1"
    assert report["meta"]["tilt_mass_mode_out"] == "consistent"
    for run in report["runs"]:
        assert run["meta"]["tilt_transport_model"] == "connection_v1"
        assert run["meta"]["tilt_mass_mode_out"] == "consistent"


@pytest.mark.regression
def test_flat_disk_kh_term_audit_refine_sweep_inner_coupled_update_mode_emits_metadata() -> (
    None
):
    report = run_flat_disk_kh_term_audit_refine_sweep(
        refine_levels=(1, 2),
        theta_values=(0.0,),
        inner_coupled_update_mode="rim_matched_radial_continuation_v1",
    )
    assert report["meta"]["inner_coupled_update_mode"] == (
        "rim_matched_radial_continuation_v1"
    )
    for run in report["runs"]:
        assert run["meta"]["inner_coupled_update_mode"] == (
            "rim_matched_radial_continuation_v1"
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
    report = _strict_mesh_report()
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
    report = _strict_mesh_report()
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


@pytest.mark.regression
def test_flat_disk_kh_outerfield_averaged_sweep_emits_matrix_and_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_term_audit as audit_mod

    def _fake_run_flat_disk_kh_term_audit(**kwargs):
        refine_steps = int(kwargs["outer_local_refine_steps"])
        refine_rmax = float(kwargs["outer_local_refine_rmax_lambda"])
        avg_rmin = float(kwargs["outer_local_vertex_average_rmin_lambda"])
        avg_rmax = float(kwargs["outer_local_vertex_average_rmax_lambda"])
        score = (
            (refine_steps - 1.0) ** 2
            + (refine_rmax - 9.0) ** 2
            + (avg_rmin - 4.0) ** 2
            + (avg_rmax - 11.0) ** 2
        )
        disk_ratio = 1.0 + 0.01 * (avg_rmin - 4.0)
        near_ratio = 1.0 + 0.02 * (refine_rmax - 9.0)
        far_ratio = 1.0 - 0.02 * (avg_rmax - 11.0)
        return {
            "rows": [
                {
                    "internal_disk_ratio_mesh_over_theory": disk_ratio,
                    "internal_outer_near_ratio_mesh_over_theory_finite": near_ratio,
                    "internal_outer_far_ratio_mesh_over_theory_finite": far_ratio,
                    "section_score_internal_bands_finite_outer_l2_log": score,
                }
            ]
        }

    monkeypatch.setattr(
        audit_mod, "run_flat_disk_kh_term_audit", _fake_run_flat_disk_kh_term_audit
    )

    report = run_flat_disk_kh_outerfield_averaged_sweep(
        outer_local_refine_steps_values=(1, 2),
        outer_local_refine_rmin_lambda_values=(1.0, 1.5),
        outer_local_refine_rmax_lambda_values=(8.0, 9.0),
        outer_local_vertex_average_steps_values=(1, 2),
        outer_local_vertex_average_rmin_lambda_values=(3.5, 4.0),
        outer_local_vertex_average_rmax_lambda_values=(11.0, 12.0),
    )
    assert report["meta"]["mode"] == "strict_outerfield_averaged_sweep"
    rows = report["rows"]
    assert len(rows) == 64
    for row in rows:
        assert int(row["outer_local_refine_steps"]) in (1, 2)
        assert np.isfinite(float(row["outer_local_refine_rmin_lambda"]))
        assert int(row["outer_local_vertex_average_steps"]) in (1, 2)
        assert np.isfinite(float(row["internal_disk_ratio_mesh_over_theory"]))
        assert np.isfinite(
            float(row["internal_outer_near_ratio_mesh_over_theory_finite"])
        )
        assert np.isfinite(
            float(row["internal_outer_far_ratio_mesh_over_theory_finite"])
        )
        assert np.isfinite(
            float(row["section_score_internal_bands_finite_outer_l2_log"])
        )
    best = report["selected_best"]
    assert int(best["outer_local_refine_steps"]) == 1
    assert float(best["outer_local_refine_rmax_lambda"]) == pytest.approx(9.0)
    assert float(best["outer_local_vertex_average_rmin_lambda"]) == pytest.approx(4.0)
    assert float(best["outer_local_vertex_average_rmax_lambda"]) == pytest.approx(11.0)


@pytest.mark.regression
def test_flat_disk_kh_outerfield_averaged_sweep_empty_values_raise() -> None:
    with pytest.raises(ValueError, match="outer_local_refine_steps_values"):
        run_flat_disk_kh_outerfield_averaged_sweep(
            outer_local_refine_steps_values=(),
        )
    with pytest.raises(ValueError, match="outer_local_refine_rmin_lambda_values"):
        run_flat_disk_kh_outerfield_averaged_sweep(
            outer_local_refine_rmin_lambda_values=(),
        )
    with pytest.raises(ValueError, match="outer_local_vertex_average_steps_values"):
        run_flat_disk_kh_outerfield_averaged_sweep(
            outer_local_vertex_average_steps_values=(),
        )
    with pytest.raises(ValueError, match="outer_local_refine_rmax_lambda_values"):
        run_flat_disk_kh_outerfield_averaged_sweep(
            outer_local_refine_rmax_lambda_values=(),
        )
    with pytest.raises(ValueError, match="outer_local_refine_steps_values must be > 0"):
        run_flat_disk_kh_outerfield_averaged_sweep(
            outer_local_refine_steps_values=(0,),
        )


@pytest.mark.regression
def test_flat_disk_kh_disk_refinement_characterization_emits_matrix_and_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_term_audit as audit_mod

    def _fake_run_flat_disk_kh_term_audit(**kwargs):
        refine = int(kwargs["refine_level"])
        rim_steps = int(kwargs["rim_local_refine_steps"])
        rim_band = float(kwargs["rim_local_refine_band_lambda"])
        disk_err = abs(refine - 3.0) + 0.5 * abs(rim_steps - 2.0) + abs(rim_band - 3.0)
        disk_ratio = float(np.exp(0.02 * disk_err))
        near_ratio = float(np.exp(0.04 * disk_err))
        far_ratio = float(np.exp(-0.03 * disk_err))
        return {
            "rows": [
                {
                    "internal_disk_ratio_mesh_over_theory": disk_ratio,
                    "internal_outer_near_ratio_mesh_over_theory": near_ratio,
                    "internal_outer_far_ratio_mesh_over_theory": far_ratio,
                    "section_score_internal_bands_finite_outer_l2_log": 0.1 * disk_err,
                    "disk_core_hmax_over_hmin_mean": 1.2 + 0.1 * disk_err,
                    "rim_band_hmax_over_hmin_mean": 1.1 + 0.05 * disk_err,
                }
            ]
        }

    monkeypatch.setattr(
        audit_mod, "run_flat_disk_kh_term_audit", _fake_run_flat_disk_kh_term_audit
    )

    report = run_flat_disk_kh_disk_refinement_characterization(
        refine_levels=(2, 3),
        rim_local_steps_values=(0, 1, 2),
        rim_local_band_lambda_values=(2.0, 3.0),
    )
    assert report["meta"]["mode"] == "strict_disk_refinement_characterization"
    rows = report["rows"]
    assert len(rows) == 10
    for row in rows:
        assert np.isfinite(float(row["internal_disk_ratio_mesh_over_theory"]))
        assert np.isfinite(float(row["internal_outer_near_ratio_mesh_over_theory"]))
        assert np.isfinite(float(row["internal_outer_far_ratio_mesh_over_theory"]))
        assert np.isfinite(
            float(row["section_score_internal_bands_finite_outer_l2_log"])
        )
        assert np.isfinite(float(row["disk_core_hmax_over_hmin_mean"]))
        assert np.isfinite(float(row["rim_band_hmax_over_hmin_mean"]))
    best = report["selected_best"]
    assert int(best["refine_level"]) == 3
    assert int(best["rim_local_refine_steps"]) == 2
    assert float(best["rim_local_refine_band_lambda"]) == pytest.approx(3.0)


def test_flat_disk_kh_disk_refinement_characterization_empty_values_raise() -> None:
    with pytest.raises(ValueError, match="refine_levels"):
        run_flat_disk_kh_disk_refinement_characterization(refine_levels=())


@pytest.mark.regression
def test_flat_disk_kh_discrete_tilt_matrix_runner_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_term_audit as audit_mod

    def _fake_refine_sweep(**kwargs):
        mass = str(kwargs.get("tilt_mass_mode_in", "consistent"))
        div = str(kwargs.get("tilt_divergence_mode_in", "native"))
        cadence = str(kwargs.get("tilt_projection_cadence", "per_step"))
        interval = int(kwargs.get("tilt_projection_interval", 1))
        iso = str(kwargs.get("isotropy_pass", "off"))
        avg = int(kwargs.get("outer_local_vertex_average_steps", 0))
        rmax = float(kwargs.get("outer_local_refine_rmax_lambda", 8.0))

        mass_penalty = 0.01 if mass == "lumped" else 0.0
        div_penalty = 0.02 if div == "vertex_recovered" else 0.0
        cadence_shift = (
            -0.01 if cadence == "per_pass" else (0.005 if interval > 1 else 0.0)
        )
        iso_shift = (
            -0.01
            if iso == "outer_far"
            else (-0.005 if iso == "outer_far_flip_only" else 0.0)
        )
        avg_shift = -0.002 * float(avg)
        rmax_shift = 0.004 * (float(rmax) - 8.0)

        far2 = (
            0.92
            - mass_penalty
            - div_penalty
            + cadence_shift
            + iso_shift
            + avg_shift
            + rmax_shift
        )
        far3 = (
            0.93
            - mass_penalty
            - div_penalty
            + cadence_shift
            + iso_shift
            + avg_shift
            + rmax_shift
        )
        disk2 = 0.98
        near2 = 0.99
        disk3 = 0.99
        near3 = 0.98
        err2 = max(abs(disk2 - 1.0), abs(near2 - 1.0), abs(far2 - 1.0))
        err3 = max(abs(disk3 - 1.0), abs(near3 - 1.0), abs(far3 - 1.0))
        return {
            "runs": [
                {
                    "meta": {"refine_level": 2},
                    "rows": [
                        {
                            "internal_disk_ratio_mesh_over_theory_v2": disk2,
                            "internal_outer_near_ratio_mesh_over_theory_v2": near2,
                            "internal_outer_far_ratio_mesh_over_theory_v2": far2,
                        }
                    ],
                },
                {
                    "meta": {"refine_level": 3},
                    "rows": [
                        {
                            "internal_disk_ratio_mesh_over_theory_v2": disk3,
                            "internal_outer_near_ratio_mesh_over_theory_v2": near3,
                            "internal_outer_far_ratio_mesh_over_theory_v2": far3,
                        }
                    ],
                },
            ],
            "err2_v2": float(err2),
            "err3_v2": float(err3),
            "adaptive_guard_pass": True,
            "adaptive_guard_reason": "pass",
        }

    monkeypatch.setattr(
        audit_mod, "run_flat_disk_kh_term_audit_refine_sweep", _fake_refine_sweep
    )

    matrix_fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "flat_disk_kh_discrete_tilt_matrix.yaml"
    )
    report_a = run_flat_disk_kh_discrete_tilt_matrix(matrix_fixture=matrix_fixture)
    report_b = run_flat_disk_kh_discrete_tilt_matrix(matrix_fixture=matrix_fixture)
    assert report_a["meta"]["mode"] == "discrete_tilt_matrix"
    assert len(report_a["phase1_rows"]) > 0
    assert len(report_a["phase1_top"]) > 0
    assert report_a["selected"] == report_b["selected"]


@lru_cache(maxsize=32)
def _cached_policy_report(items):
    """Cache repeated policy-audit report variants across regression tests."""
    return run_flat_disk_kh_term_audit(**dict(items))


def _run_policy_report(**overrides):
    base = {"refine_level": 1, "theta_values": (0.138,), "ratio_version": "v2"}
    base.update(overrides)
    items = tuple(
        sorted((key, _freeze_cache_value(value)) for key, value in base.items())
    )
    return deepcopy(_cached_policy_report(items))


@pytest.mark.acceptance
def test_flat_disk_kh_term_audit_v2_uses_strict_raw_ratio_semantics() -> None:
    target_cfg = yaml.safe_load(
        (
            ROOT / "tests" / "fixtures" / "flat_disk_kh_term_audit_v2_5pct_target.yaml"
        ).read_text(encoding="utf-8")
    )
    controls = dict(target_cfg["meta"]["controls"])
    controls.update(
        {
            "outer_mode": "disabled",
            "smoothness_model": "splay_twist",
            "tilt_mass_mode_in": "consistent",
            "tilt_divergence_mode_in": "native",
            "theta_values": (float(target_cfg["meta"]["theta_value"]),),
        }
    )
    t0 = perf_counter()
    report_v1 = run_flat_disk_kh_term_audit(
        **controls,
        ratio_version="v1",
        theory_outer_mode="infinite",
    )
    v1_seconds = float(perf_counter() - t0)

    t1 = perf_counter()
    report_v2 = run_flat_disk_kh_term_audit(
        **controls,
        ratio_version="v2",
        theory_outer_mode="infinite",
    )
    v2_seconds = float(perf_counter() - t1)

    row = report_v2["rows"][0]
    assert report_v2["meta"]["v2_ratio_semantics"] == "strict_raw"
    assert float(row["internal_disk_ratio_mesh_over_theory_v2"]) == pytest.approx(
        float(row["internal_disk_ratio_mesh_over_theory_v2_raw"]),
        rel=0.0,
        abs=1e-12,
    )
    assert float(row["internal_outer_near_ratio_mesh_over_theory_v2"]) == pytest.approx(
        float(row["internal_outer_near_ratio_mesh_over_theory_v2_raw"]),
        rel=0.0,
        abs=1e-12,
    )
    assert float(row["internal_outer_far_ratio_mesh_over_theory_v2"]) == pytest.approx(
        float(row["internal_outer_far_ratio_mesh_over_theory_v2_raw"]),
        rel=0.0,
        abs=1e-12,
    )
    meets_row_expected = bool(
        (0.95 <= float(row["internal_disk_ratio_mesh_over_theory_v2"]) <= 1.05)
        and (
            0.95 <= float(row["internal_outer_near_ratio_mesh_over_theory_v2"]) <= 1.05
        )
        and (0.95 <= float(row["internal_outer_far_ratio_mesh_over_theory_v2"]) <= 1.05)
    )
    assert bool(row["meets_5pct_v2"]) is meets_row_expected
    assert bool(report_v2["meets_5pct_v2"]) is meets_row_expected
    assert bool(row["meets_5pct_v2"]) is bool(target_cfg["expected"]["meets_5pct_v2"])

    runtime_ratio = float(v2_seconds / max(v1_seconds, 1e-12))
    assert runtime_ratio <= float(target_cfg["runtime_gate"]["max_v2_over_v1"])
    assert isinstance(bool(report_v1["meets_5pct_v2"]), bool)


@pytest.mark.acceptance
def test_flat_disk_kh_term_audit_staged_targets_match_fixture_expectations() -> None:
    cfg_p10 = yaml.safe_load(
        (
            ROOT / "tests" / "fixtures" / "flat_disk_kh_term_audit_v2_p10_target.yaml"
        ).read_text(encoding="utf-8")
    )
    cfg_p5 = yaml.safe_load(
        (
            ROOT / "tests" / "fixtures" / "flat_disk_kh_term_audit_v2_p5_target.yaml"
        ).read_text(encoding="utf-8")
    )
    controls = dict(cfg_p10["meta"]["controls"])
    controls.update(
        {
            "outer_mode": "disabled",
            "smoothness_model": "splay_twist",
            "tilt_mass_mode_in": "consistent",
            "tilt_divergence_mode_in": "native",
            "partition_mode": "fractional",
            "theta_values": (float(cfg_p10["meta"]["theta_value"]),),
            "ratio_version": "v2",
            "theory_outer_mode": "finite_bvp",
        }
    )

    report_p10 = run_flat_disk_kh_term_audit(
        **controls,
        parity_target="p10",
        axial_symmetry_gate=str(cfg_p10["meta"]["axial_symmetry_gate"]),
    )
    report_p5 = run_flat_disk_kh_term_audit(
        **controls,
        parity_target="p5",
        axial_symmetry_gate=str(cfg_p5["meta"]["axial_symmetry_gate"]),
    )
    assert bool(report_p10["meets_10pct_v2"]) is bool(
        cfg_p10["expected"]["meets_10pct_v2"]
    )
    assert bool(report_p10["meets_parity_target_v2"]) is bool(
        cfg_p10["expected"]["meets_parity_target_v2"]
    )
    assert bool(report_p5["meets_5pct_v2"]) is bool(cfg_p5["expected"]["meets_5pct_v2"])
    assert bool(report_p5["meets_parity_target_v2"]) is bool(
        cfg_p5["expected"]["meets_parity_target_v2"]
    )


@pytest.mark.regression
def test_flat_disk_kh_term_audit_parity_target_and_axial_defaults() -> None:
    report_p10 = _run_policy_report(
        parity_target="p10",
        rim_local_refine_steps=1,
        rim_local_refine_band_lambda=3.0,
    )
    report_p5 = _run_policy_report(
        parity_target="p5",
        rim_local_refine_steps=1,
        rim_local_refine_band_lambda=3.0,
    )
    assert report_p10["meta"]["parity_target"] == "p10"
    assert report_p10["meta"]["axial_symmetry_mode_effective"] == "monitor"
    assert report_p5["meta"]["parity_target"] == "p5"
    assert report_p5["meta"]["axial_symmetry_mode_effective"] == "hard"
    row_p10 = report_p10["rows"][0]
    row_p5 = report_p5["rows"][0]
    assert bool(row_p10["meets_parity_target_v2"]) is bool(row_p10["meets_10pct_v2"])
    assert bool(row_p5["meets_parity_target_v2"]) is bool(row_p5["meets_5pct_v2"])


@pytest.mark.regression
def test_flat_disk_kh_term_audit_axial_hard_gate_can_block_target() -> None:
    report = _run_policy_report(parity_target="p10", axial_symmetry_gate="hard")
    assert report["meta"]["axial_symmetry_mode_effective"] == "hard"
    row = report["rows"][0]
    assert row["axial_symmetry_mode"] == "hard"
    assert isinstance(bool(row["axial_symmetry_pass"]), bool)
    assert isinstance(bool(report["axial_symmetry_pass"]), bool)


@pytest.mark.regression
def test_flat_disk_kh_term_audit_axial_gate_off_always_passes() -> None:
    report = _run_policy_report(
        parity_target="p5",
        axial_symmetry_gate="off",
        rim_local_refine_steps=1,
        rim_local_refine_band_lambda=3.0,
    )
    assert report["meta"]["axial_symmetry_mode_effective"] == "off"
    assert bool(report["axial_symmetry_pass"]) is True
    assert bool(report["rows"][0]["axial_symmetry_pass"]) is True


@pytest.mark.regression
def test_flat_disk_kh_term_audit_isotropy_defaults_off() -> None:
    report = _run_policy_report()
    assert report["meta"]["isotropy_pass"] == "off"
    assert int(report["meta"]["isotropy_iters"]) == 0
    assert report["meta"]["isotropy_operator_mode"] == "flip_then_average"
    row = report["rows"][0]
    assert row["isotropy_pass"] == "off"
    assert int(row["isotropy_iterations_requested"]) == 0
    assert int(row["isotropy_iterations_applied"]) == 0
    assert int(row["isotropy_iterations_skipped"]) == 0
    assert row["isotropy_operator_mode"] == "flip_then_average"
    assert np.isfinite(float(row["isotropy_r_min"]))
    assert np.isfinite(float(row["isotropy_r_max"]))
    assert np.isfinite(float(row["isotropy_r_min_lambda"]))
    assert np.isfinite(float(row["isotropy_r_max_lambda"]))


@pytest.mark.regression
def test_flat_disk_kh_term_audit_isotropy_pass_outer_far_metadata() -> None:
    report = _run_policy_report(
        parity_target="p10",
        axial_symmetry_gate="monitor",
        isotropy_pass="outer_far",
        isotropy_iters=2,
        isotropy_rmin_lambda=6.0,
        isotropy_rmax_lambda=12.0,
    )
    assert report["meta"]["isotropy_pass"] == "outer_far"
    assert int(report["meta"]["isotropy_iters"]) == 2
    assert report["meta"]["isotropy_operator_mode"] == "flip_then_average"
    row = report["rows"][0]
    assert row["isotropy_pass"] == "outer_far"
    assert int(row["isotropy_iterations_requested"]) == 2
    assert int(row["isotropy_iterations_applied"]) >= 0
    assert int(row["isotropy_iterations_skipped"]) >= 0
    assert row["isotropy_operator_mode"] == "flip_then_average"


@pytest.mark.regression
def test_flat_disk_kh_term_audit_isotropy_pass_outer_far_flip_only_metadata() -> None:
    report = _run_policy_report(
        parity_target="p10",
        axial_symmetry_gate="monitor",
        isotropy_pass="outer_far_flip_only",
        isotropy_iters=2,
        isotropy_rmin_lambda=6.0,
        isotropy_rmax_lambda=12.0,
    )
    assert report["meta"]["isotropy_pass"] == "outer_far_flip_only"
    assert int(report["meta"]["isotropy_iters"]) == 2
    assert report["meta"]["isotropy_operator_mode"] == "flip_only"
    row = report["rows"][0]
    assert row["isotropy_pass"] == "outer_far_flip_only"
    assert int(row["isotropy_iterations_requested"]) == 2
    assert int(row["isotropy_iterations_applied"]) >= 0
    assert int(row["isotropy_iterations_skipped"]) >= 0
    assert row["isotropy_operator_mode"] == "flip_only"
