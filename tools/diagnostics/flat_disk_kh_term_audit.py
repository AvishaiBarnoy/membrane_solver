#!/usr/bin/env python3
"""Per-theta KH physical lane audit for flat one-leaflet disk benchmark."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import yaml

# Compatibility facade; numerical metric implementations are owned by flat_disk_kh_metrics.
from tools.diagnostics.flat_disk_kh_metrics import (
    _band_anisotropy_metrics,
    _boundary_realization_metrics,
    _leakage_metrics,
    _mesh_internal_band_split,
    _mesh_internal_region_split,
    _mesh_internal_triangle_terms,  # noqa: F401 - compatibility re-export
    _radial_frames,  # noqa: F401 - compatibility re-export
    _radial_projected_band_diagnostics,
    _resolution_metrics,
    _theory_term_band_split,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)


def _ensure_repo_root_on_sys_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


DEFAULT_OUT = (
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "flat_disk_kh_term_audit.yaml"
)
AXIAL_SYMMETRY_THRESHOLD_DEFAULT = 0.05
PARITY_TARGET_TO_ABS_TOL = {"p10": 0.10, "p5": 0.05}


def _validate_parity_target(parity_target: str) -> str:
    mode = str(parity_target).strip().lower()
    if mode not in PARITY_TARGET_TO_ABS_TOL:
        raise ValueError("parity_target must be 'p10' or 'p5'.")
    return mode


def _resolve_axial_symmetry_mode(
    *, parity_target: str, axial_symmetry_gate: str | None
) -> str:
    if axial_symmetry_gate is None:
        return "monitor" if str(parity_target) == "p10" else "hard"
    mode = str(axial_symmetry_gate).strip().lower()
    if mode not in {"monitor", "hard", "off"}:
        raise ValueError("axial_symmetry_gate must be one of: monitor, hard, off.")
    return mode


def _validate_isotropy_pass(isotropy_pass: str) -> str:
    mode = str(isotropy_pass).strip().lower()
    if mode not in {"off", "outer_far", "outer_far_flip_only"}:
        raise ValueError(
            "isotropy_pass must be one of: off, outer_far, outer_far_flip_only."
        )
    return mode


def _ratio_target_bounds(*, parity_target: str) -> tuple[float, float]:
    tol = float(PARITY_TARGET_TO_ABS_TOL[str(parity_target)])
    return (1.0 - tol, 1.0 + tol)


def _meets_ratio_target(
    *,
    disk_ratio: float,
    outer_near_ratio: float,
    outer_far_ratio: float,
    parity_target: str,
) -> bool:
    lo, hi = _ratio_target_bounds(parity_target=str(parity_target))
    return bool(
        (lo <= float(disk_ratio) <= hi)
        and (lo <= float(outer_near_ratio) <= hi)
        and (lo <= float(outer_far_ratio) <= hi)
    )


def _evaluate_axial_symmetry(
    *,
    disk_tphi_over_trad_median: float,
    outer_near_tphi_over_trad_median: float,
    outer_far_tphi_over_trad_median: float,
    mode: str,
) -> bool:
    if str(mode) in {"off", "monitor"}:
        return True
    threshold = float(AXIAL_SYMMETRY_THRESHOLD_DEFAULT)
    return bool(
        float(disk_tphi_over_trad_median) <= threshold
        and float(outer_near_tphi_over_trad_median) <= threshold
        and float(outer_far_tphi_over_trad_median) <= threshold
    )


def _theory_split_coeffs(theory: Any) -> tuple[float, float, float]:
    """Return (c_in, c_out, b_contact) for theory energy split at fixed theta."""
    k_splay = float(theory.kappa)
    c_in = float(
        np.pi
        * k_splay
        * float(theory.radius)
        * float(theory.lambda_inverse)
        * float(theory.ratio_i1_i0)
    )
    c_out = float(
        np.pi
        * k_splay
        * float(theory.radius)
        * float(theory.lambda_inverse)
        * float(theory.ratio_k1_k0)
    )
    b_contact = float(theory.coeff_B)
    return c_in, c_out, b_contact


def _kh_reference_profiles(*, smoothness_model: str) -> dict[str, str]:
    """Return explicit continuum reference labels for flat KH diagnostics."""
    from tools.flat_disk_benchmark_metrics import (
        _flat_benchmark_reference_profiles,
    )

    return _flat_benchmark_reference_profiles(
        parameterization="kh_physical", smoothness_model=smoothness_model
    )


def _run_single_level(
    *,
    fixture: Path,
    refine_level: int,
    outer_mode: str,
    smoothness_model: str,
    kappa_physical: float,
    kappa_t_physical: float,
    radius_nm: float,
    length_scale_nm: float,
    drive_physical: float,
    theta_values: Sequence[float],
    tilt_mass_mode_in: str,
    tilt_mass_mode_out: str = "auto",
    tilt_transport_model: str = "ambient_v1",
    tilt_divergence_mode_in: str = "native",
    rim_local_refine_steps: int,
    rim_local_refine_band_lambda: float,
    outer_local_refine_steps: int,
    outer_local_refine_rmin_lambda: float,
    outer_local_refine_rmax_lambda: float,
    local_edge_flip_steps: int,
    local_edge_flip_rmin_lambda: float,
    local_edge_flip_rmax_lambda: float,
    outer_local_vertex_average_steps: int,
    outer_local_vertex_average_rmin_lambda: float,
    outer_local_vertex_average_rmax_lambda: float,
    tilt_projection_cadence: str = "per_step",
    tilt_projection_interval: int = 1,
    tilt_post_relax_inner_steps: int = 0,
    tilt_post_relax_step_size: float = 0.0,
    tilt_post_relax_passes: int = 1,
    tilt_solver: str | None = "gd",
    mesh_quality_auto_repair_enabled: bool | None = False,
    inner_coupled_update_mode: str = "off",
    radial_projection_diagnostic: bool,
    partition_mode: str,
    ratio_version: str = "v1",
    theory_outer_mode: str = "infinite",
    parity_target: str = "p10",
    axial_symmetry_gate: str | None = None,
    isotropy_pass: str = "off",
    isotropy_iters: int = 0,
    isotropy_rmin_lambda: float = 0.0,
    isotropy_rmax_lambda: float = 0.0,
    theta_relax_mode: str = "fixed",
    theta_relax_max_repeats: int = 1,
    theta_relax_energy_abs_tol: float = 1.0e-10,
    theta_relax_plateau_patience: int = 2,
) -> dict[str, Any]:
    from tools.diagnostics.flat_disk_one_leaflet_theory import (
        compute_flat_disk_kh_physical_theory,
        physical_to_dimensionless_theory_params,
    )
    from tools.reproduce_flat_disk_one_leaflet import (
        _build_minimizer,
        _configure_benchmark_mesh,
        _prepare_benchmark_mesh,
        _resolve_inner_coupled_update_mode,
        _resolve_tilt_divergence_mode_in,
        _resolve_tilt_mass_mode,
        _resolve_tilt_transport_model,
        _run_theta_relaxation,
    )

    params = physical_to_dimensionless_theory_params(
        kappa_physical=float(kappa_physical),
        kappa_t_physical=float(kappa_t_physical),
        radius_physical=float(radius_nm),
        drive_physical=float(drive_physical),
        length_scale=float(length_scale_nm),
    )
    theory = compute_flat_disk_kh_physical_theory(params)
    c_in, c_out, b_contact = _theory_split_coeffs(theory)

    mesh = _prepare_benchmark_mesh(
        fixture=fixture,
        refine_level=int(refine_level),
        radius=float(theory.radius),
        lambda_value=float(theory.lambda_value),
        rim_local_refine=(rim_local_refine_steps, rim_local_refine_band_lambda),
        outer_local_refine=(
            outer_local_refine_steps,
            outer_local_refine_rmin_lambda,
            outer_local_refine_rmax_lambda,
        ),
        local_edge_flip=(
            local_edge_flip_steps,
            local_edge_flip_rmin_lambda,
            local_edge_flip_rmax_lambda,
        ),
        outer_local_vertex_average=(
            outer_local_vertex_average_steps,
            outer_local_vertex_average_rmin_lambda,
            outer_local_vertex_average_rmax_lambda,
        ),
    )
    positions = mesh.positions_view()
    mesh_r_max = float(np.max(np.linalg.norm(positions[:, :2], axis=1)))

    mass_mode = _resolve_tilt_mass_mode(
        tilt_mass_mode_in, leaflet="in", auto_default="consistent"
    )
    mass_mode_out = _resolve_tilt_mass_mode(
        tilt_mass_mode_out, leaflet="out", auto_default="lumped"
    )
    transport_model_value = _resolve_tilt_transport_model(tilt_transport_model)
    inner_coupled_update_mode_value = _resolve_inner_coupled_update_mode(
        inner_coupled_update_mode
    )
    div_mode = _resolve_tilt_divergence_mode_in(tilt_divergence_mode_in)
    ratio_version_mode = str(ratio_version).strip().lower()
    if ratio_version_mode not in {"v1", "v2"}:
        raise ValueError("ratio_version must be 'v1' or 'v2'.")
    theory_outer_mode_requested = str(theory_outer_mode).strip().lower()
    if theory_outer_mode_requested not in {"infinite", "finite_bvp"}:
        raise ValueError("theory_outer_mode must be 'infinite' or 'finite_bvp'.")
    parity_target_mode = _validate_parity_target(str(parity_target))
    axial_symmetry_mode = _resolve_axial_symmetry_mode(
        parity_target=parity_target_mode,
        axial_symmetry_gate=axial_symmetry_gate,
    )
    isotropy_pass_mode = _validate_isotropy_pass(str(isotropy_pass))
    isotropy_iters_value = int(isotropy_iters)
    if isotropy_iters_value < 0:
        raise ValueError("isotropy_iters must be >= 0.")
    isotropy_rmin_lambda_value = float(isotropy_rmin_lambda)
    isotropy_rmax_lambda_value = float(isotropy_rmax_lambda)
    theta_relax_mode_value = str(theta_relax_mode).strip().lower()
    if theta_relax_mode_value not in {"fixed", "adaptive"}:
        raise ValueError("theta_relax_mode must be 'fixed' or 'adaptive'.")
    theta_relax_max_repeats_value = int(theta_relax_max_repeats)
    if theta_relax_max_repeats_value < 1:
        raise ValueError("theta_relax_max_repeats must be >= 1.")
    theta_relax_energy_abs_tol_value = float(theta_relax_energy_abs_tol)
    if theta_relax_energy_abs_tol_value < 0.0:
        raise ValueError("theta_relax_energy_abs_tol must be >= 0.")
    theta_relax_plateau_patience_value = int(theta_relax_plateau_patience)
    if theta_relax_plateau_patience_value < 1:
        raise ValueError("theta_relax_plateau_patience must be >= 1.")
    isotropy_operator_mode = (
        "flip_only"
        if isotropy_pass_mode == "outer_far_flip_only"
        else "flip_then_average"
    )
    theory_outer_mode_v2 = (
        "finite_bvp"
        if ratio_version_mode == "v2" and theory_outer_mode_requested == "infinite"
        else theory_outer_mode_requested
    )
    theory_outer_r_max_v2 = float(mesh_r_max)
    if (
        theory_outer_mode_v2 == "finite_bvp"
        and float(outer_local_refine_rmax_lambda) > 0.0
    ):
        candidate_r_max = float(theory.radius) + float(
            outer_local_refine_rmax_lambda
        ) * float(theory.lambda_value)
        if candidate_r_max > float(theory.radius):
            theory_outer_r_max_v2 = float(candidate_r_max)
    isotropy_stats = {
        "iterations_requested": isotropy_iters_value,
        "iterations_applied": 0,
        "iterations_skipped": 0,
        "r_min": max(
            0.0,
            float(theory.radius)
            + float(isotropy_rmin_lambda_value) * float(theory.lambda_value),
        ),
        "r_max": max(
            0.0,
            float(theory.radius)
            + float(isotropy_rmax_lambda_value) * float(theory.lambda_value),
        ),
        "operator_mode": isotropy_operator_mode,
    }

    _configure_benchmark_mesh(
        mesh,
        theory_params=params,
        parameterization="kh_physical",
        outer_mode=outer_mode,
        smoothness_model=smoothness_model,
        splay_modulus_scale_in=1.0,
        tilt_mass_mode_in=str(mass_mode),
        tilt_mass_mode_out=str(mass_mode_out),
        tilt_transport_model=str(transport_model_value),
        tilt_divergence_mode_in=str(div_mode),
        tilt_solver=tilt_solver,
        tilt_projection_cadence=str(tilt_projection_cadence),
        tilt_projection_interval=int(tilt_projection_interval),
        tilt_post_relax_inner_steps=int(tilt_post_relax_inner_steps),
        tilt_post_relax_step_size=float(tilt_post_relax_step_size),
        tilt_post_relax_passes=int(tilt_post_relax_passes),
        mesh_quality_auto_repair_enabled=mesh_quality_auto_repair_enabled,
        inner_coupled_update_mode=str(inner_coupled_update_mode_value),
    )
    minim = _build_minimizer(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.project_tilts_to_tangent()

    rows: list[dict[str, float]] = []
    for theta in np.asarray(theta_values, dtype=float).tolist():
        theta_f = float(theta)
        repeats_planned = (
            int(theta_relax_max_repeats_value)
            if theta_relax_mode_value == "adaptive"
            else 1
        )
        repeats_applied = 0
        converged = False
        prev_energy = None
        plateau_count = 0
        mesh_total = float("nan")
        projection_apply_count = 0
        projection_norm_loss_outer_far = float("nan")
        inner_coupled_update_enabled = False
        inner_coupled_mode = "off"
        inner_coupled_candidate_row_count = 0
        inner_coupled_capped_row_count = 0
        inner_coupled_rim_row_count = 0
        inner_coupled_cap_magnitude = 0.0
        for relax_rep in range(repeats_planned):
            reset_this_rep = int(relax_rep) == 0
            mesh_total = float(
                _run_theta_relaxation(
                    minim,
                    theta_value=theta_f,
                    reset_outer=reset_this_rep,
                    reset_inner=reset_this_rep,
                )
            )
            repeats_applied = int(relax_rep) + 1
            projection_stats = getattr(minim, "_last_tilt_projection_stats", None)
            if isinstance(projection_stats, dict):
                projection_apply_count = int(
                    projection_stats.get(
                        "projection_apply_count", projection_apply_count
                    )
                )
                loss_val = float(
                    projection_stats.get(
                        "tilt_projection_norm_loss_outer_far",
                        projection_norm_loss_outer_far,
                    )
                )
                if np.isfinite(loss_val):
                    projection_norm_loss_outer_far = float(loss_val)
            inner_coupled_stats = getattr(
                minim, "_last_inner_coupled_update_mode_stats", None
            )
            if isinstance(inner_coupled_stats, dict):
                inner_coupled_update_enabled = bool(
                    inner_coupled_stats.get("enabled", inner_coupled_update_enabled)
                )
                inner_coupled_mode = str(
                    inner_coupled_stats.get("mode", inner_coupled_mode)
                )
                inner_coupled_candidate_row_count = int(
                    inner_coupled_stats.get(
                        "candidate_row_count", inner_coupled_candidate_row_count
                    )
                )
                inner_coupled_capped_row_count = int(
                    inner_coupled_stats.get(
                        "capped_row_count", inner_coupled_capped_row_count
                    )
                )
                inner_coupled_rim_row_count = int(
                    inner_coupled_stats.get(
                        "rim_row_count", inner_coupled_rim_row_count
                    )
                )
                inner_coupled_cap_magnitude = float(
                    inner_coupled_stats.get(
                        "cap_magnitude", inner_coupled_cap_magnitude
                    )
                )
            if theta_relax_mode_value == "adaptive":
                if prev_energy is not None:
                    delta = abs(float(mesh_total) - float(prev_energy))
                    if delta <= float(theta_relax_energy_abs_tol_value):
                        plateau_count += 1
                    else:
                        plateau_count = 0
                    if plateau_count >= int(theta_relax_plateau_patience_value):
                        converged = True
                        break
                prev_energy = float(mesh_total)
        breakdown = minim.compute_energy_breakdown()
        mesh_contact = float(breakdown.get("tilt_thetaB_contact_in", 0.0))
        mesh_internal = float(mesh_total - mesh_contact)
        mesh_region = _mesh_internal_region_split(
            mesh,
            smoothness_model=smoothness_model,
            radius=float(theory.radius),
        )
        mesh_bands = _mesh_internal_band_split(
            mesh,
            smoothness_model=smoothness_model,
            radius=float(theory.radius),
            lambda_value=float(theory.lambda_value),
            rim_half_width_lambda=1.0,
            outer_near_width_lambda=4.0,
            partition_mode=str(partition_mode),
        )

        th_in = float(c_in * theta_f * theta_f)
        th_out = float(c_out * theta_f * theta_f)
        th_contact = float(-b_contact * theta_f)
        th_total = float(th_in + th_out + th_contact)
        th_internal = float(th_in + th_out)
        th_bands = _theory_term_band_split(
            theta=theta_f,
            kappa=float(theory.kappa),
            kappa_t=float(theory.kappa_t),
            radius=float(theory.radius),
            lambda_value=float(theory.lambda_value),
            rim_half_width_lambda=1.0,
            outer_near_width_lambda=4.0,
        )
        th_bands_finite = _theory_term_band_split(
            theta=theta_f,
            kappa=float(theory.kappa),
            kappa_t=float(theory.kappa_t),
            radius=float(theory.radius),
            lambda_value=float(theory.lambda_value),
            rim_half_width_lambda=1.0,
            outer_near_width_lambda=4.0,
            outer_r_max=mesh_r_max,
            theory_outer_mode="infinite",
        )
        th_bands_v2 = _theory_term_band_split(
            theta=theta_f,
            kappa=float(theory.kappa),
            kappa_t=float(theory.kappa_t),
            radius=float(theory.radius),
            lambda_value=float(theory.lambda_value),
            rim_half_width_lambda=1.0,
            outer_near_width_lambda=4.0,
            outer_r_max=theory_outer_r_max_v2,
            theory_outer_mode=theory_outer_mode_v2,
        )

        def _ratio(mesh_val: float, theory_val: float) -> float:
            if abs(float(theory_val)) <= 1e-18:
                return float("nan")
            return float(float(mesh_val) / float(theory_val))

        boundary = _boundary_realization_metrics(
            mesh,
            radius=float(theory.radius),
            theta_value=theta_f,
        )
        leakage = _leakage_metrics(
            mesh,
            radius=float(theory.radius),
            lambda_value=float(theory.lambda_value),
            rim_half_width_lambda=1.0,
            outer_near_width_lambda=4.0,
        )
        anisotropy = _band_anisotropy_metrics(
            mesh,
            radius=float(theory.radius),
            lambda_value=float(theory.lambda_value),
            leakage=leakage,
        )
        if bool(radial_projection_diagnostic):
            proj_radial = _radial_projected_band_diagnostics(
                mesh,
                smoothness_model=smoothness_model,
                radius=float(theory.radius),
                lambda_value=float(theory.lambda_value),
                theory_bands=th_bands,
            )
        else:
            proj_radial = {
                "proj_radial_mesh_internal": float("nan"),
                "proj_radial_mesh_internal_disk": float("nan"),
                "proj_radial_mesh_internal_outer": float("nan"),
                "proj_radial_mesh_internal_disk_core": float("nan"),
                "proj_radial_mesh_internal_rim_band": float("nan"),
                "proj_radial_mesh_internal_outer_near": float("nan"),
                "proj_radial_mesh_internal_outer_far": float("nan"),
                "proj_radial_internal_disk_core_abs_error": float("nan"),
                "proj_radial_internal_rim_band_abs_error": float("nan"),
                "proj_radial_internal_outer_near_abs_error": float("nan"),
                "proj_radial_internal_outer_far_abs_error": float("nan"),
            }

        ratio_internal_disk = _ratio(float(mesh_region["mesh_internal_disk"]), th_in)
        ratio_internal_outer = _ratio(float(mesh_region["mesh_internal_outer"]), th_out)
        ratio_internal_disk_core = _ratio(
            float(mesh_bands["mesh_internal_disk_core"]),
            float(th_bands["theory_internal_disk_core"]),
        )
        ratio_internal_rim_band = _ratio(
            float(mesh_bands["mesh_internal_rim_band"]),
            float(th_bands["theory_internal_rim_band"]),
        )
        ratio_internal_outer_near = _ratio(
            float(mesh_bands["mesh_internal_outer_near"]),
            float(th_bands["theory_internal_outer_near"]),
        )
        ratio_internal_outer_far = _ratio(
            float(mesh_bands["mesh_internal_outer_far"]),
            float(th_bands["theory_internal_outer_far"]),
        )
        ratio_internal_outer_near_finite = _ratio(
            float(mesh_bands["mesh_internal_outer_near"]),
            float(th_bands_finite["theory_internal_outer_near"]),
        )
        ratio_internal_outer_far_finite = _ratio(
            float(mesh_bands["mesh_internal_outer_far"]),
            float(th_bands_finite["theory_internal_outer_far"]),
        )
        ratio_tilt_disk_core = _ratio(
            float(mesh_bands["mesh_tilt_disk_core"]),
            float(th_bands["theory_tilt_disk_core"]),
        )
        ratio_tilt_rim_band = _ratio(
            float(mesh_bands["mesh_tilt_rim_band"]),
            float(th_bands["theory_tilt_rim_band"]),
        )
        ratio_tilt_outer_near = _ratio(
            float(mesh_bands["mesh_tilt_outer_near"]),
            float(th_bands["theory_tilt_outer_near"]),
        )
        ratio_tilt_outer_far = _ratio(
            float(mesh_bands["mesh_tilt_outer_far"]),
            float(th_bands["theory_tilt_outer_far"]),
        )
        ratio_smooth_disk_core = _ratio(
            float(mesh_bands["mesh_smooth_disk_core"]),
            float(th_bands["theory_smooth_disk_core"]),
        )
        ratio_smooth_rim_band = _ratio(
            float(mesh_bands["mesh_smooth_rim_band"]),
            float(th_bands["theory_smooth_rim_band"]),
        )
        ratio_smooth_outer_near = _ratio(
            float(mesh_bands["mesh_smooth_outer_near"]),
            float(th_bands["theory_smooth_outer_near"]),
        )
        ratio_smooth_outer_far = _ratio(
            float(mesh_bands["mesh_smooth_outer_far"]),
            float(th_bands["theory_smooth_outer_far"]),
        )
        ratio_internal_disk_v2_raw = _ratio(
            float(mesh_bands["mesh_internal_disk_core"]),
            float(th_bands_v2["theory_internal_disk_core"]),
        )
        ratio_internal_outer_near_v2_raw = _ratio(
            float(mesh_bands["mesh_internal_outer_near"]),
            float(th_bands_v2["theory_internal_outer_near"]),
        )
        ratio_internal_outer_far_v2_raw = _ratio(
            float(mesh_bands["mesh_internal_outer_far"]),
            float(th_bands_v2["theory_internal_outer_far"]),
        )
        ratio_internal_disk_v2 = float(ratio_internal_disk_v2_raw)
        ratio_internal_outer_near_v2 = float(ratio_internal_outer_near_v2_raw)
        ratio_internal_outer_far_v2 = float(ratio_internal_outer_far_v2_raw)

        meets_10pct_v2 = _meets_ratio_target(
            disk_ratio=float(ratio_internal_disk_v2),
            outer_near_ratio=float(ratio_internal_outer_near_v2),
            outer_far_ratio=float(ratio_internal_outer_far_v2),
            parity_target="p10",
        )
        meets_5pct_v2 = _meets_ratio_target(
            disk_ratio=float(ratio_internal_disk_v2),
            outer_near_ratio=float(ratio_internal_outer_near_v2),
            outer_far_ratio=float(ratio_internal_outer_far_v2),
            parity_target="p5",
        )
        meets_parity_target_v2 = _meets_ratio_target(
            disk_ratio=float(ratio_internal_disk_v2),
            outer_near_ratio=float(ratio_internal_outer_near_v2),
            outer_far_ratio=float(ratio_internal_outer_far_v2),
            parity_target=str(parity_target_mode),
        )
        axial_symmetry_pass = _evaluate_axial_symmetry(
            disk_tphi_over_trad_median=float(
                leakage["disk_core_tphi_over_trad_median"]
            ),
            outer_near_tphi_over_trad_median=float(
                leakage["outer_near_tphi_over_trad_median"]
            ),
            outer_far_tphi_over_trad_median=float(
                leakage["outer_far_tphi_over_trad_median"]
            ),
            mode=str(axial_symmetry_mode),
        )

        score_internal_split = _ratio_distance_score(
            ratio_internal_disk,
            ratio_internal_outer,
        )
        score_internal_bands = _ratio_distance_score(
            ratio_internal_disk_core,
            ratio_internal_rim_band,
            ratio_internal_outer_near,
            ratio_internal_outer_far,
        )
        score_internal_bands_finite_outer = _ratio_distance_score(
            ratio_internal_disk_core,
            ratio_internal_rim_band,
            ratio_internal_outer_near_finite,
            ratio_internal_outer_far_finite,
        )
        score_tilt_bands = _ratio_distance_score(
            ratio_tilt_disk_core,
            ratio_tilt_rim_band,
            ratio_tilt_outer_near,
            ratio_tilt_outer_far,
        )
        score_smooth_bands = _ratio_distance_score(
            ratio_smooth_disk_core,
            ratio_smooth_rim_band,
            ratio_smooth_outer_near,
            ratio_smooth_outer_far,
        )
        score_all_terms = _ratio_distance_score(
            ratio_tilt_disk_core,
            ratio_tilt_rim_band,
            ratio_tilt_outer_near,
            ratio_tilt_outer_far,
            ratio_smooth_disk_core,
            ratio_smooth_rim_band,
            ratio_smooth_outer_near,
            ratio_smooth_outer_far,
        )
        err_internal_disk_core = float(
            float(mesh_bands["mesh_internal_disk_core"])
            - float(th_bands["theory_internal_disk_core"])
        )
        err_internal_rim_band = float(
            float(mesh_bands["mesh_internal_rim_band"])
            - float(th_bands["theory_internal_rim_band"])
        )
        err_internal_outer_near = float(
            float(mesh_bands["mesh_internal_outer_near"])
            - float(th_bands["theory_internal_outer_near"])
        )
        err_internal_outer_far = float(
            float(mesh_bands["mesh_internal_outer_far"])
            - float(th_bands["theory_internal_outer_far"])
        )

        rows.append(
            {
                "theta": theta_f,
                "radial_projection_diagnostic": bool(radial_projection_diagnostic),
                "mesh_total": mesh_total,
                "mesh_contact": mesh_contact,
                "mesh_internal": mesh_internal,
                "mesh_internal_disk": float(mesh_region["mesh_internal_disk"]),
                "mesh_internal_outer": float(mesh_region["mesh_internal_outer"]),
                "mesh_internal_total_from_regions": float(
                    mesh_region["mesh_internal_total_from_regions"]
                ),
                "mesh_tilt_disk": float(mesh_region["mesh_tilt_disk"]),
                "mesh_tilt_outer": float(mesh_region["mesh_tilt_outer"]),
                "mesh_smooth_disk": float(mesh_region["mesh_smooth_disk"]),
                "mesh_smooth_outer": float(mesh_region["mesh_smooth_outer"]),
                "theory_total": th_total,
                "theory_contact": th_contact,
                "theory_internal": th_internal,
                "theory_internal_disk": th_in,
                "theory_internal_outer": th_out,
                "total_error": float(mesh_total - th_total),
                "contact_error": float(mesh_contact - th_contact),
                "internal_error": float(mesh_internal - th_internal),
                "internal_disk_error": float(mesh_region["mesh_internal_disk"] - th_in),
                "internal_outer_error": float(
                    mesh_region["mesh_internal_outer"] - th_out
                ),
                "contact_ratio_mesh_over_theory": _ratio(mesh_contact, th_contact),
                "internal_disk_ratio_mesh_over_theory": ratio_internal_disk,
                "internal_disk_ratio_mesh_over_theory_v2": ratio_internal_disk_v2,
                "internal_outer_near_ratio_mesh_over_theory_v2": (
                    ratio_internal_outer_near_v2
                ),
                "internal_outer_far_ratio_mesh_over_theory_v2": (
                    ratio_internal_outer_far_v2
                ),
                "internal_disk_ratio_mesh_over_theory_v2_raw": ratio_internal_disk_v2_raw,
                "internal_outer_near_ratio_mesh_over_theory_v2_raw": (
                    ratio_internal_outer_near_v2_raw
                ),
                "internal_outer_far_ratio_mesh_over_theory_v2_raw": (
                    ratio_internal_outer_far_v2_raw
                ),
                "internal_disk_ratio_mesh_over_theory_v2_leakage_adjusted": (
                    ratio_internal_disk_v2_raw
                ),
                "internal_outer_near_ratio_mesh_over_theory_v2_leakage_adjusted": (
                    ratio_internal_outer_near_v2_raw
                ),
                "internal_outer_far_ratio_mesh_over_theory_v2_leakage_adjusted": (
                    ratio_internal_outer_far_v2_raw
                ),
                "v2_disk_leakage_correction_factor": 1.0,
                "v2_outer_near_leakage_correction_factor": 1.0,
                "v2_outer_far_leakage_correction_factor": 1.0,
                "internal_outer_ratio_mesh_over_theory": ratio_internal_outer,
                "mesh_internal_disk_core": float(mesh_bands["mesh_internal_disk_core"]),
                "mesh_internal_rim_band": float(mesh_bands["mesh_internal_rim_band"]),
                "mesh_internal_outer_near": float(
                    mesh_bands["mesh_internal_outer_near"]
                ),
                "mesh_internal_outer_far": float(mesh_bands["mesh_internal_outer_far"]),
                "mesh_tilt_disk_core": float(mesh_bands["mesh_tilt_disk_core"]),
                "mesh_tilt_rim_band": float(mesh_bands["mesh_tilt_rim_band"]),
                "mesh_tilt_outer_near": float(mesh_bands["mesh_tilt_outer_near"]),
                "mesh_tilt_outer_far": float(mesh_bands["mesh_tilt_outer_far"]),
                "mesh_smooth_disk_core": float(mesh_bands["mesh_smooth_disk_core"]),
                "mesh_smooth_rim_band": float(mesh_bands["mesh_smooth_rim_band"]),
                "mesh_smooth_outer_near": float(mesh_bands["mesh_smooth_outer_near"]),
                "mesh_smooth_outer_far": float(mesh_bands["mesh_smooth_outer_far"]),
                **th_bands,
                "theory_tilt_outer_near_finite": float(
                    th_bands_finite["theory_tilt_outer_near"]
                ),
                "theory_tilt_outer_far_finite": float(
                    th_bands_finite["theory_tilt_outer_far"]
                ),
                "theory_smooth_outer_near_finite": float(
                    th_bands_finite["theory_smooth_outer_near"]
                ),
                "theory_smooth_outer_far_finite": float(
                    th_bands_finite["theory_smooth_outer_far"]
                ),
                "theory_internal_outer_near_finite": float(
                    th_bands_finite["theory_internal_outer_near"]
                ),
                "theory_internal_outer_far_finite": float(
                    th_bands_finite["theory_internal_outer_far"]
                ),
                "theory_internal_disk_core_v2": float(
                    th_bands_v2["theory_internal_disk_core"]
                ),
                "theory_internal_outer_near_v2": float(
                    th_bands_v2["theory_internal_outer_near"]
                ),
                "theory_internal_outer_far_v2": float(
                    th_bands_v2["theory_internal_outer_far"]
                ),
                "theory_outer_mode_v2": str(th_bands_v2["theory_outer_mode"]),
                "theory_outer_r_max_v2": float(th_bands_v2["theory_outer_r_max"]),
                "theory_outer_r_max": float(th_bands_finite["theory_outer_r_max"]),
                "theory_internal_total_from_bands": float(
                    float(th_bands["theory_internal_disk_core"])
                    + float(th_bands["theory_internal_rim_band"])
                    + float(th_bands["theory_internal_outer_near"])
                    + float(th_bands["theory_internal_outer_far"])
                ),
                "theory_internal_bands_minus_closed_form": float(
                    (
                        float(th_bands["theory_internal_disk_core"])
                        + float(th_bands["theory_internal_rim_band"])
                        + float(th_bands["theory_internal_outer_near"])
                        + float(th_bands["theory_internal_outer_far"])
                    )
                    - th_internal
                ),
                "internal_disk_core_ratio_mesh_over_theory": ratio_internal_disk_core,
                "internal_rim_band_ratio_mesh_over_theory": ratio_internal_rim_band,
                "internal_outer_near_ratio_mesh_over_theory": ratio_internal_outer_near,
                "internal_outer_far_ratio_mesh_over_theory": ratio_internal_outer_far,
                "internal_outer_near_ratio_mesh_over_theory_finite": (
                    ratio_internal_outer_near_finite
                ),
                "internal_outer_far_ratio_mesh_over_theory_finite": (
                    ratio_internal_outer_far_finite
                ),
                "meets_10pct_v2": bool(meets_10pct_v2),
                "meets_5pct_v2": bool(meets_5pct_v2),
                "meets_parity_target_v2": bool(meets_parity_target_v2),
                "axial_symmetry_mode": str(axial_symmetry_mode),
                "axial_symmetry_pass": bool(axial_symmetry_pass),
                "theta_relax_mode": str(theta_relax_mode_value),
                "theta_relax_max_repeats": int(theta_relax_max_repeats_value),
                "theta_relax_repeats_applied": int(repeats_applied),
                "theta_relax_converged": bool(converged),
                "tilt_projection_apply_count": int(projection_apply_count),
                "tilt_projection_norm_loss_outer_far": float(
                    projection_norm_loss_outer_far
                ),
                "inner_coupled_update_enabled": bool(inner_coupled_update_enabled),
                "inner_coupled_update_mode": str(inner_coupled_mode),
                "inner_coupled_candidate_row_count": int(
                    inner_coupled_candidate_row_count
                ),
                "inner_coupled_capped_row_count": int(inner_coupled_capped_row_count),
                "inner_coupled_rim_row_count": int(inner_coupled_rim_row_count),
                "inner_coupled_cap_magnitude": float(inner_coupled_cap_magnitude),
                "isotropy_pass": str(isotropy_pass_mode),
                "isotropy_iterations_requested": int(
                    isotropy_stats["iterations_requested"]
                ),
                "isotropy_iterations_applied": int(
                    isotropy_stats["iterations_applied"]
                ),
                "isotropy_iterations_skipped": int(
                    isotropy_stats["iterations_skipped"]
                ),
                "isotropy_r_min": float(isotropy_stats["r_min"]),
                "isotropy_r_max": float(isotropy_stats["r_max"]),
                "isotropy_operator_mode": str(isotropy_stats["operator_mode"]),
                "isotropy_r_min_lambda": float(isotropy_rmin_lambda_value),
                "isotropy_r_max_lambda": float(isotropy_rmax_lambda_value),
                "tilt_disk_core_ratio_mesh_over_theory": ratio_tilt_disk_core,
                "tilt_rim_band_ratio_mesh_over_theory": ratio_tilt_rim_band,
                "tilt_outer_near_ratio_mesh_over_theory": ratio_tilt_outer_near,
                "tilt_outer_far_ratio_mesh_over_theory": ratio_tilt_outer_far,
                "smooth_disk_core_ratio_mesh_over_theory": ratio_smooth_disk_core,
                "smooth_rim_band_ratio_mesh_over_theory": ratio_smooth_rim_band,
                "smooth_outer_near_ratio_mesh_over_theory": ratio_smooth_outer_near,
                "smooth_outer_far_ratio_mesh_over_theory": ratio_smooth_outer_far,
                "section_score_internal_split_l2_log": float(
                    score_internal_split["l2_log"]
                ),
                "section_score_internal_split_max_abs_log": float(
                    score_internal_split["max_abs_log"]
                ),
                "section_score_internal_split_count": float(
                    score_internal_split["count"]
                ),
                "section_score_internal_bands_l2_log": float(
                    score_internal_bands["l2_log"]
                ),
                "section_score_internal_bands_max_abs_log": float(
                    score_internal_bands["max_abs_log"]
                ),
                "section_score_internal_bands_count": float(
                    score_internal_bands["count"]
                ),
                "section_score_internal_bands_finite_outer_l2_log": float(
                    score_internal_bands_finite_outer["l2_log"]
                ),
                "section_score_internal_bands_finite_outer_max_abs_log": float(
                    score_internal_bands_finite_outer["max_abs_log"]
                ),
                "section_score_internal_bands_finite_outer_count": float(
                    score_internal_bands_finite_outer["count"]
                ),
                "section_score_tilt_bands_l2_log": float(score_tilt_bands["l2_log"]),
                "section_score_tilt_bands_max_abs_log": float(
                    score_tilt_bands["max_abs_log"]
                ),
                "section_score_tilt_bands_count": float(score_tilt_bands["count"]),
                "section_score_smooth_bands_l2_log": float(
                    score_smooth_bands["l2_log"]
                ),
                "section_score_smooth_bands_max_abs_log": float(
                    score_smooth_bands["max_abs_log"]
                ),
                "section_score_smooth_bands_count": float(score_smooth_bands["count"]),
                "section_score_all_terms_l2_log": float(score_all_terms["l2_log"]),
                "section_score_all_terms_max_abs_log": float(
                    score_all_terms["max_abs_log"]
                ),
                "section_score_all_terms_count": float(score_all_terms["count"]),
                "rim_band_tri_count": float(mesh_bands["rim_band_tri_count"]),
                "rim_band_h_over_lambda_median": float(
                    mesh_bands["rim_band_h_over_lambda_median"]
                ),
                "proj_radial_internal_disk_core_abs_error_delta_vs_unprojected": float(
                    float(proj_radial["proj_radial_internal_disk_core_abs_error"])
                    - abs(err_internal_disk_core)
                ),
                "proj_radial_internal_rim_band_abs_error_delta_vs_unprojected": float(
                    float(proj_radial["proj_radial_internal_rim_band_abs_error"])
                    - abs(err_internal_rim_band)
                ),
                "proj_radial_internal_outer_near_abs_error_delta_vs_unprojected": float(
                    float(proj_radial["proj_radial_internal_outer_near_abs_error"])
                    - abs(err_internal_outer_near)
                ),
                "proj_radial_internal_outer_far_abs_error_delta_vs_unprojected": float(
                    float(proj_radial["proj_radial_internal_outer_far_abs_error"])
                    - abs(err_internal_outer_far)
                ),
                **boundary,
                **leakage,
                **anisotropy,
                **proj_radial,
            }
        )

    resolution = _resolution_metrics(
        mesh,
        radius=float(theory.radius),
        lambda_value=float(theory.lambda_value),
    )
    meets_10pct_v2_all = bool(
        len(rows) > 0 and all(bool(r.get("meets_10pct_v2", False)) for r in rows)
    )
    meets_5pct_v2_all = bool(
        len(rows) > 0 and all(bool(r.get("meets_5pct_v2", False)) for r in rows)
    )
    meets_parity_target_v2_all = bool(
        len(rows) > 0
        and all(bool(r.get("meets_parity_target_v2", False)) for r in rows)
    )
    axial_symmetry_pass_all = bool(
        len(rows) > 0 and all(bool(r.get("axial_symmetry_pass", False)) for r in rows)
    )

    return {
        "meta": {
            "fixture": str(fixture.relative_to(ROOT)),
            "refine_level": int(refine_level),
            "outer_mode": str(outer_mode),
            "smoothness_model": str(smoothness_model),
            "parameterization": "kh_physical",
            "theory_model": "kh_physical_strict_kh",
            **_kh_reference_profiles(smoothness_model=str(smoothness_model)),
            "kappa_physical": float(kappa_physical),
            "kappa_t_physical": float(kappa_t_physical),
            "radius_nm": float(radius_nm),
            "length_scale_nm": float(length_scale_nm),
            "drive_physical": float(drive_physical),
            "tilt_mass_mode_in": str(mass_mode),
            "tilt_mass_mode_out": str(mass_mode_out),
            "tilt_transport_model": str(transport_model_value),
            "tilt_divergence_mode_in": str(div_mode),
            "rim_local_refine_steps": int(rim_local_refine_steps),
            "rim_local_refine_band_lambda": float(rim_local_refine_band_lambda),
            "outer_local_refine_steps": int(outer_local_refine_steps),
            "outer_local_refine_rmin_lambda": float(outer_local_refine_rmin_lambda),
            "outer_local_refine_rmax_lambda": float(outer_local_refine_rmax_lambda),
            "outer_local_vertex_average_steps": int(outer_local_vertex_average_steps),
            "outer_local_vertex_average_rmin_lambda": float(
                outer_local_vertex_average_rmin_lambda
            ),
            "outer_local_vertex_average_rmax_lambda": float(
                outer_local_vertex_average_rmax_lambda
            ),
            "tilt_projection_cadence": str(tilt_projection_cadence),
            "tilt_projection_interval": int(tilt_projection_interval),
            "tilt_post_relax_inner_steps": int(tilt_post_relax_inner_steps),
            "tilt_post_relax_step_size": float(tilt_post_relax_step_size),
            "tilt_post_relax_passes": int(tilt_post_relax_passes),
            "inner_coupled_update_mode": str(inner_coupled_update_mode_value),
            "isotropy_pass": str(isotropy_pass_mode),
            "isotropy_iters": int(isotropy_iters_value),
            "isotropy_operator_mode": str(isotropy_operator_mode),
            "radial_projection_diagnostic": bool(radial_projection_diagnostic),
            "partition_mode": str(partition_mode),
            "ratio_version": str(ratio_version_mode),
            "ratio_version_requested": str(ratio_version_mode),
            "parity_target": str(parity_target_mode),
            "parity_target_bounds": list(
                _ratio_target_bounds(parity_target=parity_target_mode)
            ),
            "v2_ratio_semantics": "strict_raw",
            "axial_symmetry_mode_effective": str(axial_symmetry_mode),
            "theory_outer_mode_requested": str(theory_outer_mode_requested),
            "theory_outer_mode_v2_effective": str(theory_outer_mode_v2),
            "theory_outer_r_max_v2_effective": float(theory_outer_r_max_v2),
            "theta_relax_mode": str(theta_relax_mode_value),
            "theta_relax_max_repeats": int(theta_relax_max_repeats_value),
            "theta_relax_energy_abs_tol": float(theta_relax_energy_abs_tol_value),
            "theta_relax_plateau_patience": int(theta_relax_plateau_patience_value),
        },
        "theory": {
            "kappa": float(theory.kappa),
            "kappa_t": float(theory.kappa_t),
            "radius": float(theory.radius),
            "lambda_value": float(theory.lambda_value),
            "lambda_inverse": float(theory.lambda_inverse),
            "coeff_B": float(theory.coeff_B),
            "coeff_c_inner": float(c_in),
            "coeff_c_outer": float(c_out),
        },
        "resolution": resolution,
        "rows": rows,
        "meets_10pct_v2": bool(meets_10pct_v2_all),
        "meets_5pct_v2": bool(meets_5pct_v2_all),
        "meets_parity_target_v2": bool(meets_parity_target_v2_all),
        "axial_symmetry_pass": bool(axial_symmetry_pass_all),
    }


def run_flat_disk_kh_screening_resolution_characterization(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    outer_mode: str = "disabled",
    smoothness_model: str = "splay_twist",
    kappa_physical: float = 10.0,
    kappa_t_physical: float = 10.0,
    radius_nm: float = 7.0,
    length_scale_nm: float = 15.0,
    drive_physical: float = (2.0 / 0.7),
    tilt_mass_mode_in: str = "consistent",
    optimize_preset: str = "kh_strict_outerfield_tight",
    refine_level: int = 1,
    rim_local_refine_steps: int = 1,
    rim_local_refine_band_lambda: float = 3.0,
    outer_local_refine_steps_values: Sequence[int] = (0, 1, 2),
    outer_local_refine_rmin_lambda_values: Sequence[float] = (0.25, 0.5, 1.0),
    outer_local_refine_rmax_lambda_values: Sequence[float] = (1.5, 2.0, 4.0),
) -> dict[str, Any]:
    """Characterize near-rim shell spacing for the combined flat KH benchmark."""
    from tools.diagnostics.flat_disk_one_leaflet_theory import (
        compute_flat_disk_kh_physical_theory,
        physical_to_dimensionless_theory_params,
    )
    from tools.reproduce_flat_disk_one_leaflet import (
        run_flat_disk_one_leaflet_benchmark,
    )

    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    params = physical_to_dimensionless_theory_params(
        kappa_physical=float(kappa_physical),
        kappa_t_physical=float(kappa_t_physical),
        radius_physical=float(radius_nm),
        drive_physical=float(drive_physical),
        length_scale=float(length_scale_nm),
    )
    theory = compute_flat_disk_kh_physical_theory(params)
    lam = float(theory.lambda_value)

    step_values = [int(x) for x in outer_local_refine_steps_values]
    rmin_values = [float(x) for x in outer_local_refine_rmin_lambda_values]
    rmax_values = [float(x) for x in outer_local_refine_rmax_lambda_values]
    if len(step_values) == 0 or len(rmin_values) == 0 or len(rmax_values) == 0:
        raise ValueError("screening characterization inputs must be non-empty.")

    rows: list[dict[str, Any]] = []
    complexity_rank = 0
    for outer_steps in step_values:
        candidates = (
            [(0.0, 0.0)]
            if int(outer_steps) == 0
            else [
                (float(rmin), float(rmax))
                for rmin in rmin_values
                for rmax in rmax_values
                if float(rmax) > float(rmin)
            ]
        )
        for rmin_lambda, rmax_lambda in candidates:
            t0 = perf_counter()
            bench = run_flat_disk_one_leaflet_benchmark(
                fixture=fixture_path,
                refine_level=int(refine_level),
                outer_mode=str(outer_mode),
                smoothness_model=str(smoothness_model),
                theta_mode="optimize",
                optimize_preset=str(optimize_preset),
                parameterization="kh_physical",
                kappa_physical=float(kappa_physical),
                kappa_t_physical=float(kappa_t_physical),
                radius_nm=float(radius_nm),
                length_scale_nm=float(length_scale_nm),
                drive_physical=float(drive_physical),
                tilt_mass_mode_in=str(tilt_mass_mode_in),
                rim_local_refine_steps=int(rim_local_refine_steps),
                rim_local_refine_band_lambda=float(rim_local_refine_band_lambda),
                outer_local_refine_steps=int(outer_steps),
                outer_local_refine_rmin_lambda=float(rmin_lambda),
                outer_local_refine_rmax_lambda=float(rmax_lambda),
            )
            runtime_seconds = float(perf_counter() - t0)

            boundary = dict(bench["parity"]["boundary_at_R"])
            rim_radius = float(boundary["rim_radius"])
            outer_radius = float(boundary["outer_radius"])
            dr = max(0.0, outer_radius - rim_radius)
            theta_factor = float(bench["parity"]["theta_factor"])
            energy_factor = float(bench["parity"]["energy_factor"])
            rows.append(
                {
                    "outer_local_refine_steps": int(outer_steps),
                    "outer_local_refine_rmin_lambda": float(rmin_lambda),
                    "outer_local_refine_rmax_lambda": float(rmax_lambda),
                    "theta_star": float(bench["mesh"]["theta_star"]),
                    "total_energy": float(bench["mesh"]["total_energy"]),
                    "theta_factor": float(theta_factor),
                    "energy_factor": float(energy_factor),
                    "balanced_parity_score": float(
                        _balanced_parity_score(theta_factor, energy_factor)
                    ),
                    "rim_radius": float(rim_radius),
                    "outer_radius": float(outer_radius),
                    "first_outer_dr": float(dr),
                    "first_outer_dr_over_lambda": float(dr / max(lam, 1.0e-18)),
                    "runtime_seconds": float(runtime_seconds),
                    "complexity_rank": int(complexity_rank),
                }
            )
            complexity_rank += 1

    selected = min(
        rows,
        key=lambda row: (
            abs(float(row["first_outer_dr_over_lambda"]) - 1.0),
            float(row["balanced_parity_score"]),
            float(row["runtime_seconds"]),
            int(row["complexity_rank"]),
        ),
    )
    return {
        "meta": {
            "mode": "kh_screening_resolution_characterization",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "parameterization": "kh_physical",
            **_kh_reference_profiles(smoothness_model=str(smoothness_model)),
            "optimize_preset": str(optimize_preset),
            "outer_mode": str(outer_mode),
            "smoothness_model": str(smoothness_model),
            "tilt_mass_mode_in": str(tilt_mass_mode_in),
            "refine_level": int(refine_level),
            "rim_local_refine_steps": int(rim_local_refine_steps),
            "rim_local_refine_band_lambda": float(rim_local_refine_band_lambda),
            "lambda_value": float(lam),
            "target_first_outer_dr_over_lambda": 1.0,
        },
        "rows": rows,
        "selected_best": selected,
    }


def run_flat_disk_kh_term_audit(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    refine_level: int = 1,
    outer_mode: str = "disabled",
    smoothness_model: str = "splay_twist",
    kappa_physical: float = 10.0,
    kappa_t_physical: float = 10.0,
    radius_nm: float = 7.0,
    length_scale_nm: float = 15.0,
    drive_physical: float = (2.0 / 0.7),
    theta_values: Sequence[float] = (0.0, 6.366e-4, 0.004),
    tilt_mass_mode_in: str = "auto",
    tilt_mass_mode_out: str = "auto",
    tilt_transport_model: str = "ambient_v1",
    tilt_divergence_mode_in: str = "native",
    rim_local_refine_steps: int = 0,
    rim_local_refine_band_lambda: float = 0.0,
    outer_local_refine_steps: int = 0,
    outer_local_refine_rmin_lambda: float = 0.0,
    outer_local_refine_rmax_lambda: float = 0.0,
    local_edge_flip_steps: int = 0,
    local_edge_flip_rmin_lambda: float = -1.0,
    local_edge_flip_rmax_lambda: float = 4.0,
    outer_local_vertex_average_steps: int = 0,
    outer_local_vertex_average_rmin_lambda: float = 0.0,
    outer_local_vertex_average_rmax_lambda: float = 0.0,
    tilt_projection_cadence: str = "per_step",
    tilt_projection_interval: int = 1,
    tilt_post_relax_inner_steps: int = 0,
    tilt_post_relax_step_size: float = 0.0,
    tilt_post_relax_passes: int = 1,
    tilt_solver: str | None = "gd",
    mesh_quality_auto_repair_enabled: bool | None = False,
    inner_coupled_update_mode: str = "off",
    radial_projection_diagnostic: bool = False,
    partition_mode: str = "centroid",
    ratio_version: str = "v1",
    theory_outer_mode: str = "infinite",
    parity_target: str = "p10",
    axial_symmetry_gate: str | None = None,
    isotropy_pass: str = "off",
    isotropy_iters: int = 0,
    isotropy_rmin_lambda: float = 0.0,
    isotropy_rmax_lambda: float = 0.0,
    theta_relax_mode: str = "fixed",
    theta_relax_max_repeats: int = 1,
    theta_relax_energy_abs_tol: float = 1.0e-10,
    theta_relax_plateau_patience: int = 2,
) -> dict[str, Any]:
    """Evaluate per-theta mesh/theory split terms in KH physical lane."""
    _ensure_repo_root_on_sys_path()

    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    return _run_single_level(
        fixture=fixture_path,
        refine_level=int(refine_level),
        outer_mode=outer_mode,
        smoothness_model=smoothness_model,
        kappa_physical=float(kappa_physical),
        kappa_t_physical=float(kappa_t_physical),
        radius_nm=float(radius_nm),
        length_scale_nm=float(length_scale_nm),
        drive_physical=float(drive_physical),
        theta_values=theta_values,
        tilt_mass_mode_in=str(tilt_mass_mode_in),
        tilt_mass_mode_out=str(tilt_mass_mode_out),
        tilt_transport_model=str(tilt_transport_model),
        tilt_divergence_mode_in=str(tilt_divergence_mode_in),
        rim_local_refine_steps=int(rim_local_refine_steps),
        rim_local_refine_band_lambda=float(rim_local_refine_band_lambda),
        outer_local_refine_steps=int(outer_local_refine_steps),
        outer_local_refine_rmin_lambda=float(outer_local_refine_rmin_lambda),
        outer_local_refine_rmax_lambda=float(outer_local_refine_rmax_lambda),
        local_edge_flip_steps=int(local_edge_flip_steps),
        local_edge_flip_rmin_lambda=float(local_edge_flip_rmin_lambda),
        local_edge_flip_rmax_lambda=float(local_edge_flip_rmax_lambda),
        outer_local_vertex_average_steps=int(outer_local_vertex_average_steps),
        outer_local_vertex_average_rmin_lambda=float(
            outer_local_vertex_average_rmin_lambda
        ),
        outer_local_vertex_average_rmax_lambda=float(
            outer_local_vertex_average_rmax_lambda
        ),
        tilt_projection_cadence=str(tilt_projection_cadence),
        tilt_projection_interval=int(tilt_projection_interval),
        tilt_post_relax_inner_steps=int(tilt_post_relax_inner_steps),
        tilt_post_relax_step_size=float(tilt_post_relax_step_size),
        tilt_post_relax_passes=int(tilt_post_relax_passes),
        tilt_solver=tilt_solver,
        mesh_quality_auto_repair_enabled=mesh_quality_auto_repair_enabled,
        inner_coupled_update_mode=str(inner_coupled_update_mode),
        radial_projection_diagnostic=bool(radial_projection_diagnostic),
        partition_mode=str(partition_mode),
        ratio_version=str(ratio_version),
        theory_outer_mode=str(theory_outer_mode),
        parity_target=str(parity_target),
        axial_symmetry_gate=axial_symmetry_gate,
        isotropy_pass=str(isotropy_pass),
        isotropy_iters=int(isotropy_iters),
        isotropy_rmin_lambda=float(isotropy_rmin_lambda),
        isotropy_rmax_lambda=float(isotropy_rmax_lambda),
        theta_relax_mode=str(theta_relax_mode),
        theta_relax_max_repeats=int(theta_relax_max_repeats),
        theta_relax_energy_abs_tol=float(theta_relax_energy_abs_tol),
        theta_relax_plateau_patience=int(theta_relax_plateau_patience),
    )


def run_flat_disk_kh_term_audit_refine_sweep(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    refine_levels: Sequence[int] = (1, 2, 3),
    outer_mode: str = "disabled",
    smoothness_model: str = "splay_twist",
    kappa_physical: float = 10.0,
    kappa_t_physical: float = 10.0,
    radius_nm: float = 7.0,
    length_scale_nm: float = 15.0,
    drive_physical: float = (2.0 / 0.7),
    theta_values: Sequence[float] = (0.0, 6.366e-4, 0.004),
    tilt_mass_mode_in: str = "auto",
    tilt_mass_mode_out: str = "auto",
    tilt_transport_model: str = "ambient_v1",
    tilt_divergence_mode_in: str = "native",
    rim_local_refine_steps: int = 0,
    rim_local_refine_band_lambda: float = 0.0,
    outer_local_refine_steps: int = 0,
    outer_local_refine_rmin_lambda: float = 0.0,
    outer_local_refine_rmax_lambda: float = 0.0,
    local_edge_flip_steps: int = 0,
    local_edge_flip_rmin_lambda: float = -1.0,
    local_edge_flip_rmax_lambda: float = 4.0,
    outer_local_vertex_average_steps: int = 0,
    outer_local_vertex_average_rmin_lambda: float = 0.0,
    outer_local_vertex_average_rmax_lambda: float = 0.0,
    tilt_projection_cadence: str = "per_step",
    tilt_projection_interval: int = 1,
    tilt_post_relax_inner_steps: int = 0,
    tilt_post_relax_step_size: float = 0.0,
    tilt_post_relax_passes: int = 1,
    tilt_solver: str | None = "gd",
    mesh_quality_auto_repair_enabled: bool | None = False,
    inner_coupled_update_mode: str = "off",
    radial_projection_diagnostic: bool = False,
    partition_mode: str = "centroid",
    theory_outer_mode: str = "infinite",
    theta_relax_mode: str = "fixed",
    theta_relax_max_repeats: int = 1,
    theta_relax_energy_abs_tol: float = 1.0e-10,
    theta_relax_plateau_patience: int = 2,
) -> dict[str, Any]:
    """Run KH term audit across multiple refinement levels."""
    _ensure_repo_root_on_sys_path()

    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    levels = [int(x) for x in refine_levels]
    if len(levels) == 0:
        raise ValueError("refine_levels must be non-empty.")

    runs = [
        _run_single_level(
            fixture=fixture_path,
            refine_level=int(level),
            outer_mode=outer_mode,
            smoothness_model=smoothness_model,
            kappa_physical=float(kappa_physical),
            kappa_t_physical=float(kappa_t_physical),
            radius_nm=float(radius_nm),
            length_scale_nm=float(length_scale_nm),
            drive_physical=float(drive_physical),
            theta_values=theta_values,
            tilt_mass_mode_in=str(tilt_mass_mode_in),
            tilt_mass_mode_out=str(tilt_mass_mode_out),
            tilt_transport_model=str(tilt_transport_model),
            tilt_divergence_mode_in=str(tilt_divergence_mode_in),
            rim_local_refine_steps=int(rim_local_refine_steps),
            rim_local_refine_band_lambda=float(rim_local_refine_band_lambda),
            outer_local_refine_steps=int(outer_local_refine_steps),
            outer_local_refine_rmin_lambda=float(outer_local_refine_rmin_lambda),
            outer_local_refine_rmax_lambda=float(outer_local_refine_rmax_lambda),
            local_edge_flip_steps=int(local_edge_flip_steps),
            local_edge_flip_rmin_lambda=float(local_edge_flip_rmin_lambda),
            local_edge_flip_rmax_lambda=float(local_edge_flip_rmax_lambda),
            outer_local_vertex_average_steps=int(outer_local_vertex_average_steps),
            outer_local_vertex_average_rmin_lambda=float(
                outer_local_vertex_average_rmin_lambda
            ),
            outer_local_vertex_average_rmax_lambda=float(
                outer_local_vertex_average_rmax_lambda
            ),
            tilt_solver=tilt_solver,
            tilt_projection_cadence=str(tilt_projection_cadence),
            tilt_projection_interval=int(tilt_projection_interval),
            tilt_post_relax_inner_steps=int(tilt_post_relax_inner_steps),
            tilt_post_relax_step_size=float(tilt_post_relax_step_size),
            tilt_post_relax_passes=int(tilt_post_relax_passes),
            mesh_quality_auto_repair_enabled=mesh_quality_auto_repair_enabled,
            inner_coupled_update_mode=str(inner_coupled_update_mode),
            radial_projection_diagnostic=bool(radial_projection_diagnostic),
            partition_mode=str(partition_mode),
            theory_outer_mode=str(theory_outer_mode),
            theta_relax_mode=str(theta_relax_mode),
            theta_relax_max_repeats=int(theta_relax_max_repeats),
            theta_relax_energy_abs_tol=float(theta_relax_energy_abs_tol),
            theta_relax_plateau_patience=int(theta_relax_plateau_patience),
        )
        for level in levels
    ]
    return {
        "meta": {
            "mode": "refine_sweep",
            "refine_levels": levels,
            "outer_mode": str(outer_mode),
            "smoothness_model": str(smoothness_model),
            "parameterization": "kh_physical",
            "theory_model": "kh_physical_strict_kh",
            "tilt_mass_mode_in": str(tilt_mass_mode_in).strip().lower(),
            "tilt_mass_mode_out": (
                "lumped"
                if str(tilt_mass_mode_out).strip().lower() == "auto"
                else str(tilt_mass_mode_out).strip().lower()
            ),
            "tilt_transport_model": str(tilt_transport_model).strip().lower(),
            "tilt_divergence_mode_in": str(tilt_divergence_mode_in).strip().lower(),
            "rim_local_refine_steps": int(rim_local_refine_steps),
            "rim_local_refine_band_lambda": float(rim_local_refine_band_lambda),
            "outer_local_refine_steps": int(outer_local_refine_steps),
            "outer_local_refine_rmin_lambda": float(outer_local_refine_rmin_lambda),
            "outer_local_refine_rmax_lambda": float(outer_local_refine_rmax_lambda),
            "outer_local_vertex_average_steps": int(outer_local_vertex_average_steps),
            "outer_local_vertex_average_rmin_lambda": float(
                outer_local_vertex_average_rmin_lambda
            ),
            "outer_local_vertex_average_rmax_lambda": float(
                outer_local_vertex_average_rmax_lambda
            ),
            "tilt_projection_cadence": str(tilt_projection_cadence),
            "tilt_projection_interval": int(tilt_projection_interval),
            "tilt_post_relax_inner_steps": int(tilt_post_relax_inner_steps),
            "tilt_post_relax_step_size": float(tilt_post_relax_step_size),
            "tilt_post_relax_passes": int(tilt_post_relax_passes),
            "inner_coupled_update_mode": str(inner_coupled_update_mode).strip().lower(),
            "radial_projection_diagnostic": bool(radial_projection_diagnostic),
            "partition_mode": str(partition_mode),
            "theory_outer_mode_requested": str(theory_outer_mode),
            "theta_relax_mode": str(theta_relax_mode).strip().lower(),
            "theta_relax_max_repeats": int(theta_relax_max_repeats),
            "theta_relax_energy_abs_tol": float(theta_relax_energy_abs_tol),
            "theta_relax_plateau_patience": int(theta_relax_plateau_patience),
        },
        "runs": runs,
    }


def _balanced_parity_score(theta_factor: float, energy_factor: float) -> float:
    """Return balanced parity score from theta/energy factors."""
    return float(
        np.hypot(
            np.log(max(float(theta_factor), 1e-18)),
            np.log(max(float(energy_factor), 1e-18)),
        )
    )


def _ratio_distance_score(*ratios: float) -> dict[str, float]:
    """Return distance-to-1 scores for ratio tuples using log-space metrics."""
    vals = np.asarray([float(x) for x in ratios], dtype=float)
    good = np.isfinite(vals) & (vals > 0.0)
    if not np.any(good):
        return {"l2_log": float("nan"), "max_abs_log": float("nan"), "count": 0.0}
    logs = np.log(vals[good])
    return {
        "l2_log": float(np.sqrt(np.mean(logs * logs))),
        "max_abs_log": float(np.max(np.abs(logs))),
        "count": float(logs.size),
    }


def run_flat_disk_kh_strict_refinement_characterization(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    outer_mode: str = "disabled",
    smoothness_model: str = "splay_twist",
    kappa_physical: float = 10.0,
    kappa_t_physical: float = 10.0,
    radius_nm: float = 7.0,
    length_scale_nm: float = 15.0,
    drive_physical: float = (2.0 / 0.7),
    tilt_mass_mode_in: str = "auto",
    optimize_preset: str = "kh_wide",
    rim_band_lambda: float = 4.0,
    global_refine_levels: Sequence[int] = (1, 2, 3),
    rim_local_steps: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """Characterize strict-KH parity across global and rim-local refinement."""
    from tools.reproduce_flat_disk_one_leaflet import (
        run_flat_disk_one_leaflet_benchmark,
    )

    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    global_levels = [int(x) for x in global_refine_levels]
    local_steps = [int(x) for x in rim_local_steps]
    if len(global_levels) == 0:
        raise ValueError("global_refine_levels must be non-empty.")
    if len(local_steps) == 0:
        raise ValueError("rim_local_steps must be non-empty.")

    candidates: list[dict[str, int]] = []
    for level in global_levels:
        candidates.append({"refine_level": int(level), "rim_local_refine_steps": 0})
    for steps in local_steps:
        if int(steps) > 0:
            candidates.append({"refine_level": 1, "rim_local_refine_steps": int(steps)})

    rows: list[dict[str, float | int | bool | str]] = []
    for cand in candidates:
        refine_level = int(cand["refine_level"])
        rim_steps = int(cand["rim_local_refine_steps"])

        t0 = perf_counter()
        bench = run_flat_disk_one_leaflet_benchmark(
            fixture=fixture_path,
            refine_level=refine_level,
            outer_mode=outer_mode,
            smoothness_model=smoothness_model,
            theta_mode="optimize",
            optimize_preset=str(optimize_preset),
            parameterization="kh_physical",
            kappa_physical=float(kappa_physical),
            kappa_t_physical=float(kappa_t_physical),
            radius_nm=float(radius_nm),
            length_scale_nm=float(length_scale_nm),
            drive_physical=float(drive_physical),
            splay_modulus_scale_in=1.0,
            tilt_mass_mode_in=str(tilt_mass_mode_in),
            rim_local_refine_steps=rim_steps,
            rim_local_refine_band_lambda=float(rim_band_lambda)
            if rim_steps > 0
            else 0.0,
        )
        runtime_seconds = float(perf_counter() - t0)

        theta_factor = float(bench["parity"]["theta_factor"])
        energy_factor = float(bench["parity"]["energy_factor"])
        score = _balanced_parity_score(theta_factor, energy_factor)

        # Reuse existing audit resolution metric for h/lambda at the same mesh setup.
        audit = run_flat_disk_kh_term_audit(
            fixture=fixture_path,
            refine_level=refine_level,
            outer_mode=outer_mode,
            smoothness_model=smoothness_model,
            kappa_physical=float(kappa_physical),
            kappa_t_physical=float(kappa_t_physical),
            radius_nm=float(radius_nm),
            length_scale_nm=float(length_scale_nm),
            drive_physical=float(drive_physical),
            theta_values=(0.0,),
            tilt_mass_mode_in=str(tilt_mass_mode_in),
            rim_local_refine_steps=rim_steps,
            rim_local_refine_band_lambda=float(rim_band_lambda)
            if rim_steps > 0
            else 0.0,
        )
        rim_h_over_lambda = float(audit["resolution"]["rim_h_over_lambda_median"])

        rows.append(
            {
                "refine_level": refine_level,
                "rim_local_refine_steps": rim_steps,
                "rim_local_refine_band_lambda": (
                    float(rim_band_lambda) if rim_steps > 0 else 0.0
                ),
                "theta_factor": theta_factor,
                "energy_factor": energy_factor,
                "balanced_parity_score": score,
                "runtime_seconds": runtime_seconds,
                "rim_h_over_lambda_median": rim_h_over_lambda,
                "meets_factor_2": bool(bench["parity"]["meets_factor_2"]),
            }
        )

    if len(rows) == 0:
        raise ValueError("Strict refinement characterization produced no candidates.")

    selected = min(
        rows,
        key=lambda row: (
            float(row["balanced_parity_score"]),
            float(row["runtime_seconds"]),
            int(row["refine_level"]),
            int(row["rim_local_refine_steps"]),
        ),
    )

    return {
        "meta": {
            "mode": "strict_refinement_characterization",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "parameterization": "kh_physical",
            "splay_modulus_scale_in": 1.0,
            "tilt_mass_mode_in": str(tilt_mass_mode_in),
            "theta_mode": "optimize",
            "optimize_preset": str(optimize_preset),
            "outer_mode": str(outer_mode),
            "smoothness_model": str(smoothness_model),
            "rim_band_lambda": float(rim_band_lambda),
        },
        "rows": rows,
        "selected_best": selected,
    }


def run_flat_disk_kh_strict_preset_characterization(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    outer_mode: str = "disabled",
    smoothness_model: str = "splay_twist",
    kappa_physical: float = 10.0,
    kappa_t_physical: float = 10.0,
    radius_nm: float = 7.0,
    length_scale_nm: float = 15.0,
    drive_physical: float = (2.0 / 0.7),
    tilt_mass_mode_in: str = "auto",
    optimize_presets: Sequence[str] = (
        "kh_strict_fast",
        "kh_strict_balanced",
        "kh_strict_energy_tight",
        "kh_strict_section_tight",
        "kh_strict_outerfield_tight",
        "kh_strict_outerfield_averaged",
        "kh_strict_continuity",
        "kh_strict_robust",
    ),
    refine_level: int = 1,
    rim_local_refine_steps: int = 1,
    rim_local_refine_band_lambda: float = 4.0,
    outer_local_refine_steps: int = 0,
    outer_local_refine_rmin_lambda: float = 0.0,
    outer_local_refine_rmax_lambda: float = 0.0,
) -> dict[str, Any]:
    """Characterize strict-KH optimize preset candidates on a fixed strict mesh."""
    from tools.reproduce_flat_disk_one_leaflet import (
        run_flat_disk_one_leaflet_benchmark,
    )

    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    presets = [str(x) for x in optimize_presets]
    if len(presets) == 0:
        raise ValueError("optimize_presets must be non-empty.")

    rows: list[dict[str, float | int | bool | str]] = []
    for complexity_rank, preset in enumerate(presets):
        t0 = perf_counter()
        bench = run_flat_disk_one_leaflet_benchmark(
            fixture=fixture_path,
            refine_level=int(refine_level),
            outer_mode=outer_mode,
            smoothness_model=smoothness_model,
            theta_mode="optimize",
            optimize_preset=str(preset),
            parameterization="kh_physical",
            kappa_physical=float(kappa_physical),
            kappa_t_physical=float(kappa_t_physical),
            radius_nm=float(radius_nm),
            length_scale_nm=float(length_scale_nm),
            drive_physical=float(drive_physical),
            splay_modulus_scale_in=1.0,
            tilt_mass_mode_in=str(tilt_mass_mode_in),
            rim_local_refine_steps=int(rim_local_refine_steps),
            rim_local_refine_band_lambda=float(rim_local_refine_band_lambda),
            outer_local_refine_steps=int(outer_local_refine_steps),
            outer_local_refine_rmin_lambda=float(outer_local_refine_rmin_lambda),
            outer_local_refine_rmax_lambda=float(outer_local_refine_rmax_lambda),
        )
        runtime_seconds = float(perf_counter() - t0)

        theta_factor = float(bench["parity"]["theta_factor"])
        energy_factor = float(bench["parity"]["energy_factor"])
        score = _balanced_parity_score(theta_factor, energy_factor)
        opt = bench["optimize"] or {}
        mesh = bench.get("mesh") or {}
        profile = mesh.get("profile") or {}
        continuity = mesh.get("rim_continuity") or {}
        leakage = mesh.get("leakage") or {}
        rim_abs = float(profile.get("rim_abs_median", 0.0) or 0.0)
        jump_abs = float(continuity.get("jump_abs_median", float("nan")))
        jump_ratio = (
            float(jump_abs / max(rim_abs, 1e-18))
            if np.isfinite(jump_abs)
            else float("nan")
        )
        rows.append(
            {
                "optimize_preset": str(preset),
                "theta_factor": theta_factor,
                "energy_factor": energy_factor,
                "balanced_parity_score": score,
                "runtime_seconds": runtime_seconds,
                "optimize_seconds": float(opt.get("optimize_seconds", float("nan"))),
                "meets_factor_2": bool(bench["parity"]["meets_factor_2"]),
                "optimize_steps": int(opt.get("optimize_steps", 0) or 0),
                "optimize_inner_steps": int(opt.get("optimize_inner_steps", 0) or 0),
                "rim_jump_ratio": jump_ratio,
                "outer_tphi_over_trad_median": float(
                    leakage.get("outer_tphi_over_trad_median", float("nan"))
                ),
                "complexity_rank": int(complexity_rank),
            }
        )

    selected = min(
        rows,
        key=lambda row: (
            float(row["balanced_parity_score"]),
            float(row["runtime_seconds"]),
            int(row["complexity_rank"]),
        ),
    )
    return {
        "meta": {
            "mode": "strict_preset_characterization",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "parameterization": "kh_physical",
            "splay_modulus_scale_in": 1.0,
            "tilt_mass_mode_in": str(tilt_mass_mode_in),
            "theta_mode": "optimize",
            "outer_mode": str(outer_mode),
            "smoothness_model": str(smoothness_model),
            "refine_level": int(refine_level),
            "rim_local_refine_steps": int(rim_local_refine_steps),
            "rim_local_refine_band_lambda": float(rim_local_refine_band_lambda),
            "outer_local_refine_steps": int(outer_local_refine_steps),
            "outer_local_refine_rmin_lambda": float(outer_local_refine_rmin_lambda),
            "outer_local_refine_rmax_lambda": float(outer_local_refine_rmax_lambda),
        },
        "rows": rows,
        "selected_best": selected,
    }


def run_flat_disk_kh_outertail_characterization(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    outer_mode: str = "disabled",
    smoothness_model: str = "splay_twist",
    kappa_physical: float = 10.0,
    kappa_t_physical: float = 10.0,
    radius_nm: float = 7.0,
    length_scale_nm: float = 15.0,
    drive_physical: float = (2.0 / 0.7),
    tilt_mass_mode_in: str = "consistent",
    optimize_presets: Sequence[str] = (
        "kh_strict_outerfield_tight",
        "kh_strict_outerband_tight",
    ),
    refine_level: int = 2,
    rim_local_refine_steps: int = 1,
    rim_local_refine_band_lambda: float = 3.0,
    outer_local_refine_steps_values: Sequence[int] = (1, 2),
    outer_local_refine_rmin_lambda: float = 1.0,
    outer_local_refine_rmax_lambda_values: Sequence[float] = (8.0, 10.0),
) -> dict[str, Any]:
    """Characterize strict KH outer-tail parity over a compact mesh-control matrix."""
    from tools.reproduce_flat_disk_one_leaflet import (
        run_flat_disk_one_leaflet_benchmark,
    )

    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    presets = [str(x) for x in optimize_presets]
    outer_steps_values = [int(x) for x in outer_local_refine_steps_values]
    outer_rmax_values = [float(x) for x in outer_local_refine_rmax_lambda_values]
    if len(presets) == 0:
        raise ValueError("optimize_presets must be non-empty.")
    if len(outer_steps_values) == 0:
        raise ValueError("outer_local_refine_steps_values must be non-empty.")
    if len(outer_rmax_values) == 0:
        raise ValueError("outer_local_refine_rmax_lambda_values must be non-empty.")

    rows: list[dict[str, float | int | bool | str]] = []
    complexity_rank = 0

    def _pick(row: dict[str, Any], primary: str, fallback: str) -> float:
        if primary in row:
            return float(row[primary])
        return float(row[fallback])

    for preset in presets:
        for outer_steps in outer_steps_values:
            for outer_rmax in outer_rmax_values:
                if float(outer_rmax) <= float(outer_local_refine_rmin_lambda):
                    raise ValueError(
                        "outer_local_refine_rmax_lambda must be > "
                        "outer_local_refine_rmin_lambda in characterization."
                    )
                t0 = perf_counter()
                bench = run_flat_disk_one_leaflet_benchmark(
                    fixture=fixture_path,
                    refine_level=int(refine_level),
                    outer_mode=outer_mode,
                    smoothness_model=smoothness_model,
                    theta_mode="optimize",
                    optimize_preset=str(preset),
                    parameterization="kh_physical",
                    kappa_physical=float(kappa_physical),
                    kappa_t_physical=float(kappa_t_physical),
                    radius_nm=float(radius_nm),
                    length_scale_nm=float(length_scale_nm),
                    drive_physical=float(drive_physical),
                    splay_modulus_scale_in=1.0,
                    tilt_mass_mode_in=str(tilt_mass_mode_in),
                    rim_local_refine_steps=int(rim_local_refine_steps),
                    rim_local_refine_band_lambda=float(rim_local_refine_band_lambda),
                    outer_local_refine_steps=int(outer_steps),
                    outer_local_refine_rmin_lambda=float(
                        outer_local_refine_rmin_lambda
                    ),
                    outer_local_refine_rmax_lambda=float(outer_rmax),
                )
                runtime_seconds = float(perf_counter() - t0)
                theta_star = float(bench["mesh"]["theta_star"])
                theta_factor = float(bench["parity"]["theta_factor"])
                energy_factor = float(bench["parity"]["energy_factor"])
                balanced_parity_score = _balanced_parity_score(
                    theta_factor, energy_factor
                )

                audit = run_flat_disk_kh_term_audit(
                    fixture=fixture_path,
                    refine_level=int(refine_level),
                    outer_mode=outer_mode,
                    smoothness_model=smoothness_model,
                    kappa_physical=float(kappa_physical),
                    kappa_t_physical=float(kappa_t_physical),
                    radius_nm=float(radius_nm),
                    length_scale_nm=float(length_scale_nm),
                    drive_physical=float(drive_physical),
                    theta_values=(theta_star,),
                    tilt_mass_mode_in=str(tilt_mass_mode_in),
                    rim_local_refine_steps=int(rim_local_refine_steps),
                    rim_local_refine_band_lambda=float(rim_local_refine_band_lambda),
                    outer_local_refine_steps=int(outer_steps),
                    outer_local_refine_rmin_lambda=float(
                        outer_local_refine_rmin_lambda
                    ),
                    outer_local_refine_rmax_lambda=float(outer_rmax),
                )
                if len(audit.get("rows", [])) == 0:
                    raise ValueError(
                        "run_flat_disk_kh_term_audit returned empty rows for "
                        f"candidate preset={preset} outer_steps={outer_steps} "
                        f"outer_rmax={outer_rmax}"
                    )
                arow = audit["rows"][0]
                outer_near_ratio = _pick(
                    arow,
                    "internal_outer_near_ratio_mesh_over_theory_finite",
                    "internal_outer_near_ratio_mesh_over_theory",
                )
                outer_far_ratio = _pick(
                    arow,
                    "internal_outer_far_ratio_mesh_over_theory_finite",
                    "internal_outer_far_ratio_mesh_over_theory",
                )
                outer_tail_score = _ratio_distance_score(
                    outer_near_ratio, outer_far_ratio
                )
                if not np.isfinite(outer_tail_score["l2_log"]) or not np.isfinite(
                    outer_tail_score["max_abs_log"]
                ):
                    raise ValueError(
                        "Non-finite outer-tail score for "
                        f"preset={preset} outer_steps={outer_steps} "
                        f"outer_rmax={outer_rmax}"
                    )

                rows.append(
                    {
                        "optimize_preset": str(preset),
                        "outer_local_refine_steps": int(outer_steps),
                        "outer_local_refine_rmax_lambda": float(outer_rmax),
                        "theta_factor": theta_factor,
                        "energy_factor": energy_factor,
                        "balanced_parity_score": float(balanced_parity_score),
                        "runtime_seconds": runtime_seconds,
                        "outer_near_ratio_mesh_over_theory": float(outer_near_ratio),
                        "outer_far_ratio_mesh_over_theory": float(outer_far_ratio),
                        "outer_tail_balance_score": float(outer_tail_score["l2_log"]),
                        "outer_tail_max_abs_log": float(
                            outer_tail_score["max_abs_log"]
                        ),
                        "complexity_rank": int(complexity_rank),
                    }
                )
                complexity_rank += 1

    if len(rows) == 0:
        raise ValueError("Strict outer-tail characterization produced no candidates.")

    selected = min(
        rows,
        key=lambda row: (
            float(row["outer_tail_balance_score"]),
            float(row["balanced_parity_score"]),
            float(row["runtime_seconds"]),
            int(row["complexity_rank"]),
        ),
    )
    return {
        "meta": {
            "mode": "strict_outertail_characterization",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "parameterization": "kh_physical",
            "splay_modulus_scale_in": 1.0,
            "tilt_mass_mode_in": str(tilt_mass_mode_in),
            "theta_mode": "optimize",
            "outer_mode": str(outer_mode),
            "smoothness_model": str(smoothness_model),
            "refine_level": int(refine_level),
            "rim_local_refine_steps": int(rim_local_refine_steps),
            "rim_local_refine_band_lambda": float(rim_local_refine_band_lambda),
            "outer_local_refine_rmin_lambda": float(outer_local_refine_rmin_lambda),
        },
        "rows": rows,
        "selected_best": selected,
    }


def run_flat_disk_kh_outerfield_averaged_sweep(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    outer_mode: str = "disabled",
    smoothness_model: str = "splay_twist",
    kappa_physical: float = 10.0,
    kappa_t_physical: float = 10.0,
    radius_nm: float = 7.0,
    length_scale_nm: float = 15.0,
    drive_physical: float = (2.0 / 0.7),
    tilt_mass_mode_in: str = "consistent",
    theta_value: float = 0.138,
    refine_level: int = 2,
    rim_local_refine_steps: int = 1,
    rim_local_refine_band_lambda: float = 3.0,
    outer_local_refine_steps_values: Sequence[int] | None = None,
    outer_local_refine_rmin_lambda: float = 1.0,
    outer_local_refine_rmin_lambda_values: Sequence[float] | None = None,
    outer_local_refine_rmax_lambda_values: Sequence[float] = (8.0, 9.0, 10.0),
    outer_local_vertex_average_steps: int = 2,
    outer_local_vertex_average_steps_values: Sequence[int] | None = None,
    outer_local_vertex_average_rmin_lambda_values: Sequence[float] = (3.5, 4.0, 4.5),
    outer_local_vertex_average_rmax_lambda_values: Sequence[float] = (10.0, 11.0, 12.0),
    ratio_version: str = "v1",
) -> dict[str, Any]:
    """Sweep averaged outer-field mesh controls on a fixed-theta strict KH lane."""
    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    refine_rmin_values = (
        [float(x) for x in outer_local_refine_rmin_lambda_values]
        if outer_local_refine_rmin_lambda_values is not None
        else [float(outer_local_refine_rmin_lambda)]
    )
    refine_steps_values = (
        [int(x) for x in outer_local_refine_steps_values]
        if outer_local_refine_steps_values is not None
        else [1]
    )
    refine_rmax_values = [float(x) for x in outer_local_refine_rmax_lambda_values]
    avg_steps_values = (
        [int(x) for x in outer_local_vertex_average_steps_values]
        if outer_local_vertex_average_steps_values is not None
        else [int(outer_local_vertex_average_steps)]
    )
    avg_rmin_values = [float(x) for x in outer_local_vertex_average_rmin_lambda_values]
    avg_rmax_values = [float(x) for x in outer_local_vertex_average_rmax_lambda_values]
    if len(refine_steps_values) == 0:
        raise ValueError("outer_local_refine_steps_values must be non-empty.")
    if len(refine_rmin_values) == 0:
        raise ValueError("outer_local_refine_rmin_lambda_values must be non-empty.")
    if len(refine_rmax_values) == 0:
        raise ValueError("outer_local_refine_rmax_lambda_values must be non-empty.")
    if len(avg_steps_values) == 0:
        raise ValueError("outer_local_vertex_average_steps_values must be non-empty.")
    if len(avg_rmin_values) == 0:
        raise ValueError(
            "outer_local_vertex_average_rmin_lambda_values must be non-empty."
        )
    if len(avg_rmax_values) == 0:
        raise ValueError(
            "outer_local_vertex_average_rmax_lambda_values must be non-empty."
        )
    ratio_version_mode = str(ratio_version).strip().lower()
    if ratio_version_mode not in {"v1", "v2"}:
        raise ValueError("ratio_version must be 'v1' or 'v2'.")

    rows: list[dict[str, float | int | str]] = []
    complexity_rank = 0
    for refine_steps in refine_steps_values:
        if int(refine_steps) <= 0:
            raise ValueError("outer_local_refine_steps_values must be > 0.")
        for refine_rmin in refine_rmin_values:
            for refine_rmax in refine_rmax_values:
                if float(refine_rmax) <= float(refine_rmin):
                    raise ValueError(
                        "outer_local_refine_rmax_lambda must be > "
                        "outer_local_refine_rmin_lambda in strict averaged sweep."
                    )
                for avg_steps in avg_steps_values:
                    if int(avg_steps) < 0:
                        raise ValueError(
                            "outer_local_vertex_average_steps_values must be >= 0."
                        )
                    for avg_rmin in avg_rmin_values:
                        for avg_rmax in avg_rmax_values:
                            if float(avg_rmax) <= float(avg_rmin):
                                raise ValueError(
                                    "outer_local_vertex_average_rmax_lambda must be > "
                                    "outer_local_vertex_average_rmin_lambda in strict averaged "
                                    "sweep."
                                )
                            t0 = perf_counter()
                            audit = run_flat_disk_kh_term_audit(
                                fixture=fixture_path,
                                refine_level=int(refine_level),
                                outer_mode=outer_mode,
                                smoothness_model=smoothness_model,
                                kappa_physical=float(kappa_physical),
                                kappa_t_physical=float(kappa_t_physical),
                                radius_nm=float(radius_nm),
                                length_scale_nm=float(length_scale_nm),
                                drive_physical=float(drive_physical),
                                theta_values=(float(theta_value),),
                                tilt_mass_mode_in=str(tilt_mass_mode_in),
                                rim_local_refine_steps=int(rim_local_refine_steps),
                                rim_local_refine_band_lambda=float(
                                    rim_local_refine_band_lambda
                                ),
                                outer_local_refine_steps=int(refine_steps),
                                outer_local_refine_rmin_lambda=float(refine_rmin),
                                outer_local_refine_rmax_lambda=float(refine_rmax),
                                local_edge_flip_steps=0,
                                local_edge_flip_rmin_lambda=-1.0,
                                local_edge_flip_rmax_lambda=4.0,
                                outer_local_vertex_average_steps=int(avg_steps),
                                outer_local_vertex_average_rmin_lambda=float(avg_rmin),
                                outer_local_vertex_average_rmax_lambda=float(avg_rmax),
                                ratio_version=str(ratio_version_mode),
                            )
                            runtime_seconds = float(perf_counter() - t0)
                            if len(audit.get("rows", [])) == 0:
                                raise AssertionError(
                                    "run_flat_disk_kh_term_audit returned empty rows in strict "
                                    "outerfield averaged sweep."
                                )
                            row0 = audit["rows"][0]
                            if ratio_version_mode == "v2":
                                disk_ratio = float(
                                    row0["internal_disk_ratio_mesh_over_theory_v2"]
                                )
                                near_ratio = float(
                                    row0[
                                        "internal_outer_near_ratio_mesh_over_theory_v2"
                                    ]
                                )
                                far_ratio = float(
                                    row0["internal_outer_far_ratio_mesh_over_theory_v2"]
                                )
                            else:
                                disk_ratio = float(
                                    row0["internal_disk_ratio_mesh_over_theory"]
                                )
                                near_ratio = float(
                                    row0[
                                        "internal_outer_near_ratio_mesh_over_theory_finite"
                                    ]
                                )
                                far_ratio = float(
                                    row0[
                                        "internal_outer_far_ratio_mesh_over_theory_finite"
                                    ]
                                )
                            section_score = float(
                                row0["section_score_internal_bands_finite_outer_l2_log"]
                            )
                            if not np.isfinite(disk_ratio):
                                raise ValueError(
                                    "Non-finite disk ratio in strict averaged sweep for "
                                    f"outer_local_refine_steps={refine_steps}, "
                                    f"outer_local_refine_rmin_lambda={refine_rmin}, "
                                    f"outer_local_refine_rmax_lambda={refine_rmax}, "
                                    f"outer_local_vertex_average_steps={avg_steps}, "
                                    f"outer_local_vertex_average_rmin_lambda={avg_rmin}, "
                                    f"outer_local_vertex_average_rmax_lambda={avg_rmax}."
                                )
                            if not np.isfinite(near_ratio) or not np.isfinite(
                                far_ratio
                            ):
                                raise ValueError(
                                    "Non-finite outer ratios in strict averaged sweep for "
                                    f"outer_local_refine_steps={refine_steps}, "
                                    f"outer_local_refine_rmin_lambda={refine_rmin}, "
                                    f"outer_local_refine_rmax_lambda={refine_rmax}, "
                                    f"outer_local_vertex_average_steps={avg_steps}, "
                                    f"outer_local_vertex_average_rmin_lambda={avg_rmin}, "
                                    f"outer_local_vertex_average_rmax_lambda={avg_rmax}."
                                )
                            if not np.isfinite(section_score):
                                raise ValueError(
                                    "Non-finite section score in strict averaged sweep for "
                                    f"outer_local_refine_steps={refine_steps}, "
                                    f"outer_local_refine_rmin_lambda={refine_rmin}, "
                                    f"outer_local_refine_rmax_lambda={refine_rmax}, "
                                    f"outer_local_vertex_average_steps={avg_steps}, "
                                    f"outer_local_vertex_average_rmin_lambda={avg_rmin}, "
                                    f"outer_local_vertex_average_rmax_lambda={avg_rmax}."
                                )
                            rows.append(
                                {
                                    "outer_local_refine_steps": int(refine_steps),
                                    "outer_local_refine_rmin_lambda": float(
                                        refine_rmin
                                    ),
                                    "outer_local_refine_rmax_lambda": float(
                                        refine_rmax
                                    ),
                                    "outer_local_vertex_average_steps": int(avg_steps),
                                    "outer_local_vertex_average_rmin_lambda": float(
                                        avg_rmin
                                    ),
                                    "outer_local_vertex_average_rmax_lambda": float(
                                        avg_rmax
                                    ),
                                    "internal_disk_ratio_mesh_over_theory": float(
                                        disk_ratio
                                    ),
                                    "internal_outer_near_ratio_mesh_over_theory_finite": float(
                                        near_ratio
                                    ),
                                    "internal_outer_far_ratio_mesh_over_theory_finite": float(
                                        far_ratio
                                    ),
                                    "internal_disk_ratio_mesh_over_theory_v2": float(
                                        disk_ratio
                                    ),
                                    "internal_outer_near_ratio_mesh_over_theory_v2": float(
                                        near_ratio
                                    ),
                                    "internal_outer_far_ratio_mesh_over_theory_v2": float(
                                        far_ratio
                                    ),
                                    "section_score_internal_bands_finite_outer_l2_log": float(
                                        section_score
                                    ),
                                    "runtime_seconds": float(runtime_seconds),
                                    "complexity_rank": int(complexity_rank),
                                }
                            )
                            complexity_rank += 1

    if len(rows) == 0:
        raise AssertionError("Strict outerfield averaged sweep produced no candidates.")

    selected = min(
        rows,
        key=lambda row: (
            float(row["section_score_internal_bands_finite_outer_l2_log"]),
            abs(float(row["internal_disk_ratio_mesh_over_theory"]) - 1.0),
            abs(float(row["internal_outer_near_ratio_mesh_over_theory_finite"]) - 1.0),
            abs(float(row["internal_outer_far_ratio_mesh_over_theory_finite"]) - 1.0),
            float(row["runtime_seconds"]),
            int(row["complexity_rank"]),
        ),
    )
    return {
        "meta": {
            "mode": "strict_outerfield_averaged_sweep",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "parameterization": "kh_physical",
            "splay_modulus_scale_in": 1.0,
            "tilt_mass_mode_in": str(tilt_mass_mode_in),
            "outer_mode": str(outer_mode),
            "smoothness_model": str(smoothness_model),
            "ratio_version": str(ratio_version_mode),
            "theta_value": float(theta_value),
            "refine_level": int(refine_level),
            "rim_local_refine_steps": int(rim_local_refine_steps),
            "rim_local_refine_band_lambda": float(rim_local_refine_band_lambda),
            "outer_local_refine_steps_values": [int(x) for x in refine_steps_values],
            "outer_local_refine_rmin_lambda_values": [
                float(x) for x in refine_rmin_values
            ],
            "outer_local_vertex_average_steps_values": [
                int(x) for x in avg_steps_values
            ],
            "local_edge_flip_steps": 0,
            "baseline_controls": {
                "outer_local_refine_steps": 1,
                "outer_local_refine_rmin_lambda": 1.0,
                "outer_local_refine_rmax_lambda": 8.0,
                "outer_local_vertex_average_steps": 2,
                "outer_local_vertex_average_rmin_lambda": 4.0,
                "outer_local_vertex_average_rmax_lambda": 12.0,
            },
        },
        "rows": rows,
        "selected_best": selected,
    }


def run_flat_disk_kh_disk_refinement_characterization(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    outer_mode: str = "disabled",
    smoothness_model: str = "splay_twist",
    kappa_physical: float = 10.0,
    kappa_t_physical: float = 10.0,
    radius_nm: float = 7.0,
    length_scale_nm: float = 15.0,
    drive_physical: float = (2.0 / 0.7),
    tilt_mass_mode_in: str = "consistent",
    tilt_divergence_mode_in: str = "native",
    theta_value: float = 0.138,
    refine_levels: Sequence[int] = (2, 3),
    rim_local_steps_values: Sequence[int] = (0, 1, 2),
    rim_local_band_lambda_values: Sequence[float] = (2.0, 3.0, 4.0),
    ratio_version: str = "v1",
    theory_outer_mode: str = "infinite",
) -> dict[str, Any]:
    """Characterize strict-KH parity sensitivity to disk/rim-local refinement."""
    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    refine_vals = [int(x) for x in refine_levels]
    rim_steps_vals = [int(x) for x in rim_local_steps_values]
    rim_band_vals = [float(x) for x in rim_local_band_lambda_values]
    if len(refine_vals) == 0:
        raise ValueError("refine_levels must be non-empty.")
    if len(rim_steps_vals) == 0:
        raise ValueError("rim_local_steps_values must be non-empty.")
    if len(rim_band_vals) == 0:
        raise ValueError("rim_local_band_lambda_values must be non-empty.")
    if any(level < 0 for level in refine_vals):
        raise ValueError("refine_levels must be >= 0.")
    if any(step < 0 for step in rim_steps_vals):
        raise ValueError("rim_local_steps_values must be >= 0.")
    if any(band <= 0.0 for band in rim_band_vals):
        raise ValueError("rim_local_band_lambda_values must be > 0.")
    ratio_version_mode = str(ratio_version).strip().lower()
    if ratio_version_mode not in {"v1", "v2"}:
        raise ValueError("ratio_version must be 'v1' or 'v2'.")

    rows: list[dict[str, float | int | str]] = []
    complexity_rank = 0
    for refine_level in refine_vals:
        for rim_steps in rim_steps_vals:
            bands = [0.0] if rim_steps == 0 else rim_band_vals
            for rim_band in bands:
                report = run_flat_disk_kh_term_audit(
                    fixture=fixture_path,
                    refine_level=refine_level,
                    outer_mode=outer_mode,
                    smoothness_model=smoothness_model,
                    kappa_physical=kappa_physical,
                    kappa_t_physical=kappa_t_physical,
                    radius_nm=radius_nm,
                    length_scale_nm=length_scale_nm,
                    drive_physical=drive_physical,
                    theta_values=(theta_value,),
                    tilt_mass_mode_in=tilt_mass_mode_in,
                    tilt_divergence_mode_in=tilt_divergence_mode_in,
                    rim_local_refine_steps=rim_steps,
                    rim_local_refine_band_lambda=rim_band,
                    ratio_version=ratio_version_mode,
                    theory_outer_mode=theory_outer_mode,
                )
                row = dict(report["rows"][0])
                row["refine_level"] = int(refine_level)
                row["rim_local_refine_steps"] = int(rim_steps)
                row["rim_local_refine_band_lambda"] = float(rim_band)
                row["complexity_rank"] = complexity_rank
                rows.append(row)
                complexity_rank += 1

    selected = min(
        rows,
        key=lambda row: (
            float(row["section_score_internal_bands_finite_outer_l2_log"]),
            abs(float(row["internal_disk_ratio_mesh_over_theory"]) - 1.0),
            abs(float(row["internal_outer_near_ratio_mesh_over_theory"]) - 1.0),
            abs(float(row["internal_outer_far_ratio_mesh_over_theory"]) - 1.0),
            float(row["complexity_rank"]),
        ),
    )
    return {
        "meta": {
            "mode": "strict_disk_refinement_characterization",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "parameterization": "kh_physical",
            "outer_mode": str(outer_mode),
            "smoothness_model": str(smoothness_model),
            "theta_value": float(theta_value),
            "ratio_version": str(ratio_version_mode),
            "theory_outer_mode_requested": str(theory_outer_mode),
            "refine_levels": [int(x) for x in refine_vals],
            "rim_local_steps_values": [int(x) for x in rim_steps_vals],
            "rim_local_band_lambda_values": [float(x) for x in rim_band_vals],
        },
        "rows": rows,
        "selected_best": selected,
    }


def _first_theta_row_by_refine(
    report: dict[str, Any], *, refine_level: int
) -> dict[str, Any] | None:
    """Return the first theta row for a given refine level from a sweep report."""
    for run in report.get("runs", []):
        if int(run.get("meta", {}).get("refine_level", -1)) != int(refine_level):
            continue
        rows = run.get("rows", [])
        if len(rows) == 0:
            return None
        return rows[0]
    return None


def _discrete_tilt_candidate_row(
    *, candidate: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    """Build a deterministic summary row for one discrete-tilt candidate."""
    row2 = _first_theta_row_by_refine(report, refine_level=2)
    row3 = _first_theta_row_by_refine(report, refine_level=3)
    if row2 is None or row3 is None:
        raise ValueError("refine sweep report must include refine=2 and refine=3 rows.")

    disk2 = float(row2["internal_disk_ratio_mesh_over_theory_v2"])
    near2 = float(row2["internal_outer_near_ratio_mesh_over_theory_v2"])
    far2 = float(row2["internal_outer_far_ratio_mesh_over_theory_v2"])
    disk3 = float(row3["internal_disk_ratio_mesh_over_theory_v2"])
    near3 = float(row3["internal_outer_near_ratio_mesh_over_theory_v2"])
    far3 = float(row3["internal_outer_far_ratio_mesh_over_theory_v2"])
    err2 = float(report.get("err2_v2", float("nan")))
    err3 = float(report.get("err3_v2", float("nan")))
    score = float(abs(far3 - 1.0) + 0.5 * max(err3, 0.0))
    return {
        **candidate,
        "disk_ratio_refine2_v2": float(disk2),
        "outer_near_ratio_refine2_v2": float(near2),
        "outer_far_ratio_refine2_v2": float(far2),
        "disk_ratio_refine3_v2": float(disk3),
        "outer_near_ratio_refine3_v2": float(near3),
        "outer_far_ratio_refine3_v2": float(far3),
        "err2_v2": float(err2),
        "err3_v2": float(err3),
        "phase1_score": float(score),
        "adaptive_guard_pass": report.get("adaptive_guard_pass"),
        "adaptive_guard_reason": report.get("adaptive_guard_reason"),
    }


def run_flat_disk_kh_discrete_tilt_matrix(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    matrix_fixture: Path | str = (
        ROOT / "tests" / "fixtures" / "flat_disk_kh_discrete_tilt_matrix.yaml"
    ),
) -> dict[str, Any]:
    """Run a fixture-driven discrete-tilt option matrix and rank candidates."""
    matrix_path = Path(matrix_fixture)
    if not matrix_path.is_absolute():
        matrix_path = (ROOT / matrix_path).resolve()
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix fixture not found: {matrix_path}")
    cfg = yaml.safe_load(matrix_path.read_text(encoding="utf-8")) or {}

    base = dict(cfg.get("base_controls", {}))
    sweep = dict(cfg.get("phase1", {}))
    phase2 = dict(cfg.get("phase2", {}))
    err2_max = float(cfg.get("err2_max", 0.10))
    top_k_phase1 = int(cfg.get("top_k_phase1", 3))

    mass_modes = [str(v) for v in sweep.get("tilt_mass_mode_in", ["consistent"])]
    div_modes = [str(v) for v in sweep.get("tilt_divergence_mode_in", ["native"])]
    projections = list(
        sweep.get(
            "projection_controls",
            [{"tilt_projection_cadence": "per_step", "tilt_projection_interval": 1}],
        )
    )

    phase1_rows: list[dict[str, Any]] = []
    for mass_mode, div_mode, projection in itertools.product(
        mass_modes, div_modes, projections
    ):
        candidate = {
            "tilt_mass_mode_in": str(mass_mode),
            "tilt_divergence_mode_in": str(div_mode),
            "tilt_projection_cadence": str(
                projection.get("tilt_projection_cadence", "per_step")
            ),
            "tilt_projection_interval": int(
                projection.get("tilt_projection_interval", 1)
            ),
        }
        report = run_flat_disk_kh_term_audit_refine_sweep(
            fixture=fixture,
            **{**base, **candidate},
        )
        phase1_rows.append(
            _discrete_tilt_candidate_row(candidate=candidate, report=report)
        )
    phase1_rows = sorted(
        phase1_rows,
        key=lambda row: (
            float(row["phase1_score"]),
            float(row["err3_v2"]),
            float(row["err2_v2"]),
            str(row["tilt_mass_mode_in"]),
            str(row["tilt_divergence_mode_in"]),
            str(row["tilt_projection_cadence"]),
            int(row["tilt_projection_interval"]),
        ),
    )
    phase1_kept = [
        row
        for row in phase1_rows
        if np.isfinite(float(row["err2_v2"])) and float(row["err2_v2"]) <= err2_max
    ]
    top_phase1 = (
        phase1_kept[:top_k_phase1] if phase1_kept else phase1_rows[:top_k_phase1]
    )

    phase2_rows: list[dict[str, Any]] = []
    phase2_enabled = bool(phase2.get("enabled", False))
    if phase2_enabled and top_phase1:
        isotropy_values = [str(v) for v in phase2.get("isotropy_pass", ["off"])]
        avg_steps_values = [
            int(v) for v in phase2.get("outer_local_vertex_average_steps", [0])
        ]
        rmax_values = [
            float(v) for v in phase2.get("outer_local_refine_rmax_lambda", [8.0])
        ]
        for seed, isotropy_pass, avg_steps, rmax_val in itertools.product(
            top_phase1, isotropy_values, avg_steps_values, rmax_values
        ):
            candidate = {
                "tilt_mass_mode_in": str(seed["tilt_mass_mode_in"]),
                "tilt_divergence_mode_in": str(seed["tilt_divergence_mode_in"]),
                "tilt_projection_cadence": str(seed["tilt_projection_cadence"]),
                "tilt_projection_interval": int(seed["tilt_projection_interval"]),
                "isotropy_pass": str(isotropy_pass),
                "outer_local_vertex_average_steps": int(avg_steps),
                "outer_local_refine_rmax_lambda": float(rmax_val),
            }
            report = run_flat_disk_kh_term_audit_refine_sweep(
                fixture=fixture,
                **{**base, **candidate},
            )
            row = _discrete_tilt_candidate_row(candidate=candidate, report=report)
            row["phase2_score"] = float(
                abs(float(row["outer_far_ratio_refine3_v2"]) - 1.0)
                + 0.5 * max(float(row["err3_v2"]), 0.0)
            )
            phase2_rows.append(row)
        phase2_rows = sorted(
            phase2_rows,
            key=lambda row: (
                float(row["phase2_score"]),
                float(row["err3_v2"]),
                float(row["err2_v2"]),
            ),
        )

    selected = (
        phase2_rows[0] if phase2_rows else (top_phase1[0] if top_phase1 else None)
    )
    return {
        "meta": {
            "mode": "discrete_tilt_matrix",
            "fixture": str(Path(fixture)),
            "matrix_fixture": str(matrix_path),
            "err2_max": float(err2_max),
            "top_k_phase1": int(top_k_phase1),
            "phase2_enabled": bool(phase2_enabled),
        },
        "phase1_rows": phase1_rows,
        "phase1_top": top_phase1,
        "phase2_rows": phase2_rows,
        "selected": selected,
    }


def main() -> int:
    _ensure_repo_root_on_sys_path()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--refine-level", type=int, default=1)
    ap.add_argument("--refine-levels", type=int, nargs="+", default=None)
    ap.add_argument("--outer-mode", choices=("disabled", "free"), default="disabled")
    ap.add_argument(
        "--smoothness-model",
        choices=("dirichlet", "splay_twist"),
        default="splay_twist",
    )
    ap.add_argument("--kappa-physical", type=float, default=10.0)
    ap.add_argument("--kappa-t-physical", type=float, default=10.0)
    ap.add_argument("--radius-nm", type=float, default=7.0)
    ap.add_argument("--length-scale-nm", type=float, default=15.0)
    ap.add_argument("--drive-physical", type=float, default=(2.0 / 0.7))
    ap.add_argument(
        "--tilt-mass-mode-in",
        choices=("auto", "lumped", "consistent"),
        default="auto",
    )
    ap.add_argument("--rim-local-refine-steps", type=int, default=0)
    ap.add_argument("--rim-local-refine-band-lambda", type=float, default=0.0)
    ap.add_argument("--outer-local-refine-steps", type=int, default=0)
    ap.add_argument("--outer-local-refine-rmin-lambda", type=float, default=0.0)
    ap.add_argument("--outer-local-refine-rmax-lambda", type=float, default=0.0)
    ap.add_argument("--outer-local-vertex-average-steps", type=int, default=0)
    ap.add_argument("--outer-local-vertex-average-rmin-lambda", type=float, default=0.0)
    ap.add_argument("--outer-local-vertex-average-rmax-lambda", type=float, default=0.0)
    ap.add_argument("--radial-projection-diagnostic", action="store_true")
    ap.add_argument(
        "--partition-mode",
        choices=("centroid", "fractional"),
        default="centroid",
    )
    ap.add_argument(
        "--strict-refinement-characterization",
        action="store_true",
        help="Run strict-KH refinement characterization matrix.",
    )
    ap.add_argument(
        "--strict-preset-characterization",
        action="store_true",
        help="Run strict-KH optimize-preset characterization on fixed strict mesh.",
    )
    ap.add_argument(
        "--strict-outertail-characterization",
        action="store_true",
        help="Run strict-KH outer-tail characterization matrix.",
    )
    ap.add_argument(
        "--strict-outerfield-averaged-sweep",
        action="store_true",
        help="Run strict-KH fixed-theta sweep around averaged outer-field controls.",
    )
    ap.add_argument(
        "--optimize-preset",
        default="kh_wide",
    )
    ap.add_argument(
        "--optimize-presets",
        nargs="+",
        default=None,
    )
    ap.add_argument("--outer-local-refine-steps-values", type=int, nargs="+")
    ap.add_argument("--outer-local-refine-rmin-lambda-values", type=float, nargs="+")
    ap.add_argument("--outer-local-refine-rmax-lambda-values", type=float, nargs="+")
    ap.add_argument("--outer-local-vertex-average-steps-values", type=int, nargs="+")
    ap.add_argument(
        "--outer-local-vertex-average-rmin-lambda-values", type=float, nargs="+"
    )
    ap.add_argument(
        "--outer-local-vertex-average-rmax-lambda-values", type=float, nargs="+"
    )
    ap.add_argument("--sweep-theta-value", type=float, default=0.138)
    ap.add_argument(
        "--theta-values", type=float, nargs="+", default=[0.0, 6.366e-4, 0.004]
    )
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if args.strict_outerfield_averaged_sweep:
        outer_rmax_values = (
            tuple(float(x) for x in args.outer_local_refine_rmax_lambda_values)
            if args.outer_local_refine_rmax_lambda_values is not None
            else (8.0, 9.0, 10.0)
        )
        avg_rmin_values = (
            tuple(float(x) for x in args.outer_local_vertex_average_rmin_lambda_values)
            if args.outer_local_vertex_average_rmin_lambda_values is not None
            else (3.5, 4.0, 4.5)
        )
        avg_rmax_values = (
            tuple(float(x) for x in args.outer_local_vertex_average_rmax_lambda_values)
            if args.outer_local_vertex_average_rmax_lambda_values is not None
            else (10.0, 11.0, 12.0)
        )
        report = run_flat_disk_kh_outerfield_averaged_sweep(
            fixture=args.fixture,
            outer_mode=args.outer_mode,
            smoothness_model=args.smoothness_model,
            kappa_physical=args.kappa_physical,
            kappa_t_physical=args.kappa_t_physical,
            radius_nm=args.radius_nm,
            length_scale_nm=args.length_scale_nm,
            drive_physical=args.drive_physical,
            tilt_mass_mode_in=args.tilt_mass_mode_in,
            theta_value=args.sweep_theta_value,
            refine_level=args.refine_level,
            rim_local_refine_steps=args.rim_local_refine_steps,
            rim_local_refine_band_lambda=args.rim_local_refine_band_lambda,
            outer_local_refine_steps_values=(
                tuple(int(x) for x in args.outer_local_refine_steps_values)
                if args.outer_local_refine_steps_values is not None
                else None
            ),
            outer_local_refine_rmin_lambda=args.outer_local_refine_rmin_lambda,
            outer_local_refine_rmin_lambda_values=(
                tuple(float(x) for x in args.outer_local_refine_rmin_lambda_values)
                if args.outer_local_refine_rmin_lambda_values is not None
                else None
            ),
            outer_local_refine_rmax_lambda_values=outer_rmax_values,
            outer_local_vertex_average_steps=args.outer_local_vertex_average_steps,
            outer_local_vertex_average_steps_values=(
                tuple(int(x) for x in args.outer_local_vertex_average_steps_values)
                if args.outer_local_vertex_average_steps_values is not None
                else None
            ),
            outer_local_vertex_average_rmin_lambda_values=avg_rmin_values,
            outer_local_vertex_average_rmax_lambda_values=avg_rmax_values,
        )
    elif args.strict_outertail_characterization:
        presets = (
            tuple(str(x) for x in args.optimize_presets)
            if args.optimize_presets is not None
            else ("kh_strict_outerfield_tight", "kh_strict_outerband_tight")
        )
        outer_steps_values = (
            tuple(int(x) for x in args.outer_local_refine_steps_values)
            if args.outer_local_refine_steps_values is not None
            else (1, 2)
        )
        outer_rmax_values = (
            tuple(float(x) for x in args.outer_local_refine_rmax_lambda_values)
            if args.outer_local_refine_rmax_lambda_values is not None
            else (8.0, 10.0)
        )
        report = run_flat_disk_kh_outertail_characterization(
            fixture=args.fixture,
            outer_mode=args.outer_mode,
            smoothness_model=args.smoothness_model,
            kappa_physical=args.kappa_physical,
            kappa_t_physical=args.kappa_t_physical,
            radius_nm=args.radius_nm,
            length_scale_nm=args.length_scale_nm,
            drive_physical=args.drive_physical,
            tilt_mass_mode_in=args.tilt_mass_mode_in,
            optimize_presets=presets,
            refine_level=args.refine_level,
            rim_local_refine_steps=args.rim_local_refine_steps,
            rim_local_refine_band_lambda=args.rim_local_refine_band_lambda,
            outer_local_refine_steps_values=outer_steps_values,
            outer_local_refine_rmin_lambda=args.outer_local_refine_rmin_lambda,
            outer_local_refine_rmax_lambda_values=outer_rmax_values,
        )
    elif args.strict_preset_characterization:
        presets = (
            tuple(str(x) for x in args.optimize_presets)
            if args.optimize_presets is not None
            else (
                "kh_strict_fast",
                "kh_strict_balanced",
                "kh_strict_energy_tight",
                "kh_strict_section_tight",
                "kh_strict_outerfield_tight",
                "kh_strict_outerfield_averaged",
                "kh_strict_continuity",
                "kh_strict_robust",
            )
        )
        report = run_flat_disk_kh_strict_preset_characterization(
            fixture=args.fixture,
            outer_mode=args.outer_mode,
            smoothness_model=args.smoothness_model,
            kappa_physical=args.kappa_physical,
            kappa_t_physical=args.kappa_t_physical,
            radius_nm=args.radius_nm,
            length_scale_nm=args.length_scale_nm,
            drive_physical=args.drive_physical,
            tilt_mass_mode_in=args.tilt_mass_mode_in,
            optimize_presets=presets,
            refine_level=args.refine_level,
            rim_local_refine_steps=args.rim_local_refine_steps,
            rim_local_refine_band_lambda=args.rim_local_refine_band_lambda,
            outer_local_refine_steps=args.outer_local_refine_steps,
            outer_local_refine_rmin_lambda=args.outer_local_refine_rmin_lambda,
            outer_local_refine_rmax_lambda=args.outer_local_refine_rmax_lambda,
        )
    elif args.strict_refinement_characterization:
        report = run_flat_disk_kh_strict_refinement_characterization(
            fixture=args.fixture,
            outer_mode=args.outer_mode,
            smoothness_model=args.smoothness_model,
            kappa_physical=args.kappa_physical,
            kappa_t_physical=args.kappa_t_physical,
            radius_nm=args.radius_nm,
            length_scale_nm=args.length_scale_nm,
            drive_physical=args.drive_physical,
            tilt_mass_mode_in=args.tilt_mass_mode_in,
            optimize_preset=args.optimize_preset,
            rim_band_lambda=4.0,
        )
    elif args.refine_levels is not None:
        report = run_flat_disk_kh_term_audit_refine_sweep(
            fixture=args.fixture,
            refine_levels=tuple(int(x) for x in args.refine_levels),
            outer_mode=args.outer_mode,
            smoothness_model=args.smoothness_model,
            kappa_physical=args.kappa_physical,
            kappa_t_physical=args.kappa_t_physical,
            radius_nm=args.radius_nm,
            length_scale_nm=args.length_scale_nm,
            drive_physical=args.drive_physical,
            theta_values=args.theta_values,
            tilt_mass_mode_in=args.tilt_mass_mode_in,
            rim_local_refine_steps=args.rim_local_refine_steps,
            rim_local_refine_band_lambda=args.rim_local_refine_band_lambda,
            outer_local_refine_steps=args.outer_local_refine_steps,
            outer_local_refine_rmin_lambda=args.outer_local_refine_rmin_lambda,
            outer_local_refine_rmax_lambda=args.outer_local_refine_rmax_lambda,
            outer_local_vertex_average_steps=args.outer_local_vertex_average_steps,
            outer_local_vertex_average_rmin_lambda=(
                args.outer_local_vertex_average_rmin_lambda
            ),
            outer_local_vertex_average_rmax_lambda=(
                args.outer_local_vertex_average_rmax_lambda
            ),
            radial_projection_diagnostic=args.radial_projection_diagnostic,
            partition_mode=args.partition_mode,
        )
    else:
        report = run_flat_disk_kh_term_audit(
            fixture=args.fixture,
            refine_level=args.refine_level,
            outer_mode=args.outer_mode,
            smoothness_model=args.smoothness_model,
            kappa_physical=args.kappa_physical,
            kappa_t_physical=args.kappa_t_physical,
            radius_nm=args.radius_nm,
            length_scale_nm=args.length_scale_nm,
            drive_physical=args.drive_physical,
            theta_values=args.theta_values,
            tilt_mass_mode_in=args.tilt_mass_mode_in,
            rim_local_refine_steps=args.rim_local_refine_steps,
            rim_local_refine_band_lambda=args.rim_local_refine_band_lambda,
            outer_local_refine_steps=args.outer_local_refine_steps,
            outer_local_refine_rmin_lambda=args.outer_local_refine_rmin_lambda,
            outer_local_refine_rmax_lambda=args.outer_local_refine_rmax_lambda,
            outer_local_vertex_average_steps=args.outer_local_vertex_average_steps,
            outer_local_vertex_average_rmin_lambda=(
                args.outer_local_vertex_average_rmin_lambda
            ),
            outer_local_vertex_average_rmax_lambda=(
                args.outer_local_vertex_average_rmax_lambda
            ),
            radial_projection_diagnostic=args.radial_projection_diagnostic,
            partition_mode=args.partition_mode,
        )

    out = Path(args.output)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
