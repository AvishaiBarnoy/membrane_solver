#!/usr/bin/env python3
"""Strict-KH outer-field vertex-distribution audit at fixed theta_B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
DEFAULT_OUT = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "diagnostics"
    / "flat_disk_kh_outer_vertex_audit.yaml"
)


def _resolve_controls(optimize_preset: str) -> dict[str, float | int]:
    p = str(optimize_preset).lower()
    if p == "kh_strict_outertail_balanced":
        return {
            "refine_level": 2,
            "rim_local_refine_steps": 1,
            "rim_local_refine_band_lambda": 3.0,
            "outer_local_refine_steps": 1,
            "outer_local_refine_rmin_lambda": 1.0,
            "outer_local_refine_rmax_lambda": 10.0,
            "local_edge_flip_steps": 0,
            "local_edge_flip_rmin_lambda": -1.0,
            "local_edge_flip_rmax_lambda": 4.0,
        }
    if p == "kh_strict_outerfield_tight":
        return {
            "refine_level": 2,
            "rim_local_refine_steps": 1,
            "rim_local_refine_band_lambda": 3.0,
            "outer_local_refine_steps": 1,
            "outer_local_refine_rmin_lambda": 1.0,
            "outer_local_refine_rmax_lambda": 8.0,
            "local_edge_flip_steps": 0,
            "local_edge_flip_rmin_lambda": -1.0,
            "local_edge_flip_rmax_lambda": 4.0,
        }
    if p == "kh_strict_outerfield_quality":
        return {
            "refine_level": 2,
            "rim_local_refine_steps": 1,
            "rim_local_refine_band_lambda": 3.0,
            "outer_local_refine_steps": 1,
            "outer_local_refine_rmin_lambda": 1.0,
            "outer_local_refine_rmax_lambda": 8.0,
            "local_edge_flip_steps": 1,
            "local_edge_flip_rmin_lambda": 2.0,
            "local_edge_flip_rmax_lambda": 6.0,
        }
    return {
        "refine_level": 1,
        "rim_local_refine_steps": 1,
        "rim_local_refine_band_lambda": 4.0,
        "outer_local_refine_steps": 0,
        "outer_local_refine_rmin_lambda": 0.0,
        "outer_local_refine_rmax_lambda": 0.0,
        "local_edge_flip_steps": 0,
        "local_edge_flip_rmin_lambda": -1.0,
        "local_edge_flip_rmax_lambda": 4.0,
    }


def _vertex_bands(
    *,
    positions: np.ndarray,
    tri_rows: np.ndarray,
    tri_area: np.ndarray,
    tilts: np.ndarray,
    radius: float,
    lambda_value: float,
) -> list[dict[str, float | int | str]]:
    from tools.diagnostics.flat_disk_kh_metrics import _radial_frames

    r, r_hat, phi_hat = _radial_frames(positions)

    dual = np.zeros(positions.shape[0], dtype=float)
    share = np.asarray(tri_area, dtype=float) / 3.0
    np.add.at(dual, tri_rows[:, 0], share)
    np.add.at(dual, tri_rows[:, 1], share)
    np.add.at(dual, tri_rows[:, 2], share)

    t_rad = np.einsum("ij,ij->i", tilts, r_hat)
    t_phi = np.einsum("ij,ij->i", tilts, phi_hat)

    rim_end = float(radius + lambda_value)
    near_end = float(radius + (4.0 * lambda_value))
    masks = {
        "outer_near": (r > rim_end) & (r <= near_end),
        "outer_far": r > near_end,
    }

    rows: list[dict[str, float | int | str]] = []
    for name, vm in masks.items():
        vm = np.asarray(vm, dtype=bool)
        if int(np.count_nonzero(vm)) == 0:
            raise ValueError(f"Empty vertex band: {name}")
        trad = float(np.median(np.abs(t_rad[vm])))
        tphi = float(np.median(np.abs(t_phi[vm])))
        rows.append(
            {
                "band": str(name),
                "vertex_count": int(np.count_nonzero(vm)),
                "dual_area_total": float(np.sum(dual[vm])),
                "vertex_density_per_dual_area": float(
                    np.count_nonzero(vm) / max(float(np.sum(dual[vm])), 1e-18)
                ),
                "t_phi_over_t_rad_median": float(tphi / max(trad, 1e-18)),
            }
        )
    return rows


def _safe_ratio(num: float, den: float) -> float:
    return float(float(num) / max(float(den), 1e-18))


def _analytic_radial_amplitude(
    *,
    radii: np.ndarray,
    theta: float,
    radius: float,
    lambda_value: float,
) -> np.ndarray:
    from scipy import special

    r = np.asarray(radii, dtype=float)
    x = float(radius) / max(float(lambda_value), 1e-18)
    i1_x = float(special.iv(1, x))
    k1_x = float(special.kv(1, x))
    if abs(i1_x) < 1e-18 or abs(k1_x) < 1e-18:
        raise ValueError("Invalid KH normalization for analytic radial field.")

    scaled_r = np.clip(r / max(float(lambda_value), 1e-18), 0.0, None)
    amplitude = np.zeros_like(scaled_r)
    inner = r <= float(radius)
    amplitude[inner] = (
        float(theta) * np.asarray(special.iv(1, scaled_r[inner]), dtype=float) / i1_x
    )
    amplitude[~inner] = (
        float(theta) * np.asarray(special.kv(1, scaled_r[~inner]), dtype=float) / k1_x
    )
    return amplitude


def _field_profile_errors(
    *,
    positions: np.ndarray,
    tri_rows: np.ndarray,
    tri_area: np.ndarray,
    tilts: np.ndarray,
    theta: float,
    radius: float,
    lambda_value: float,
) -> dict[str, dict[str, float | int]]:
    from tools.diagnostics.flat_disk_kh_metrics import _radial_frames

    r, r_hat, phi_hat = _radial_frames(positions)
    expected = _analytic_radial_amplitude(
        radii=r,
        theta=float(theta),
        radius=float(radius),
        lambda_value=float(lambda_value),
    )
    actual_radial = np.einsum("ij,ij->i", tilts, r_hat)
    actual_azimuthal = np.einsum("ij,ij->i", tilts, phi_hat)

    dual = np.zeros(positions.shape[0], dtype=float)
    share = np.asarray(tri_area, dtype=float) / 3.0
    np.add.at(dual, tri_rows[:, 0], share)
    np.add.at(dual, tri_rows[:, 1], share)
    np.add.at(dual, tri_rows[:, 2], share)

    lam = float(lambda_value)
    rad = float(radius)
    masks = {
        "disk_core": r < max(0.0, rad - lam),
        "rim_band": np.abs(r - rad) <= lam,
        "outer_near": (r > rad + lam) & (r <= rad + 4.0 * lam),
        "outer_tail": (r > rad + 4.0 * lam) & (r <= rad + 8.0 * lam),
    }
    out: dict[str, dict[str, float | int]] = {}
    theta_scale = max(abs(float(theta)), 1e-18)
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            raise ValueError(f"Empty profile-error band: {name}")
        weights = dual[mask]
        expected_band = expected[mask]
        radial_error = actual_radial[mask] - expected_band
        azimuthal = actual_azimuthal[mask]
        expected_norm_sq = float(np.sum(weights * expected_band**2))
        out[name] = {
            "vertex_count": int(np.count_nonzero(mask)),
            "radial_relative_l2": float(
                np.sqrt(
                    float(np.sum(weights * radial_error**2))
                    / max(expected_norm_sq, 1e-30)
                )
            ),
            "azimuthal_relative_l2": float(
                np.sqrt(
                    float(np.sum(weights * azimuthal**2)) / max(expected_norm_sq, 1e-30)
                )
            ),
            "radial_abs_max_over_theta": float(
                np.max(np.abs(radial_error)) / theta_scale
            ),
        }
    return out


def _section_energy_summary(
    *, mesh_bands: dict[str, float], theory_bands: dict[str, float]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}

    def _row(name: str, mesh_v: float, theory_v: float) -> None:
        out[name] = {
            "mesh": float(mesh_v),
            "theory": float(theory_v),
            "ratio_mesh_over_theory": _safe_ratio(float(mesh_v), float(theory_v)),
        }

    m_disk_core = float(mesh_bands["mesh_internal_disk_core"])
    t_disk_core = float(theory_bands["theory_internal_disk_core"])
    m_rim = float(mesh_bands["mesh_internal_rim_band"])
    t_rim = float(theory_bands["theory_internal_rim_band"])
    _row("disk_core", m_disk_core, t_disk_core)
    _row("rim_band", m_rim, t_rim)
    _row("disk_total", m_disk_core + m_rim, t_disk_core + t_rim)
    _row(
        "outer_near",
        float(mesh_bands["mesh_internal_outer_near"]),
        float(theory_bands["theory_internal_outer_near"]),
    )
    _row(
        "outer_far",
        float(mesh_bands["mesh_internal_outer_far"]),
        float(theory_bands["theory_internal_outer_far"]),
    )
    return out


def _frozen_analytic_tilts(
    *,
    positions: np.ndarray,
    theta: float,
    radius: float,
    lambda_value: float,
) -> np.ndarray:
    from tools.diagnostics.flat_disk_kh_metrics import _radial_frames

    r, r_hat, _ = _radial_frames(positions)
    amplitude = _analytic_radial_amplitude(
        radii=r,
        theta=float(theta),
        radius=float(radius),
        lambda_value=float(lambda_value),
    )
    return r_hat * amplitude[:, None]


def run_flat_disk_kh_outer_vertex_audit(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    optimize_preset: str = "kh_strict_outerfield_tight",
    theta: float = 0.138,
    include_frozen_analytic: bool = False,
    relax_max_repeats: int = 5,
    relax_energy_abs_tol: float = 1.0e-8,
    refine_level: int | None = None,
) -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from runtime.refinement import refine_triangle_mesh
    from tools.diagnostics.flat_disk_kh_metrics import (
        _mesh_internal_band_split,
        _radial_frames,
        _theory_term_band_split,
    )
    from tools.diagnostics.flat_disk_one_leaflet_theory import (
        compute_flat_disk_kh_physical_theory,
        physical_to_dimensionless_theory_params,
    )
    from tools.reproduce_flat_disk_one_leaflet import (
        _build_minimizer,
        _configure_benchmark_mesh,
        _flip_edges_locally_in_annulus,
        _load_mesh_from_fixture,
        _refine_mesh_locally_in_outer_annulus,
        _refine_mesh_locally_near_rim,
        _run_theta_relaxation,
    )

    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    controls = _resolve_controls(optimize_preset)
    if refine_level is not None:
        controls["refine_level"] = int(refine_level)
    params = physical_to_dimensionless_theory_params(
        kappa_physical=10.0,
        kappa_t_physical=10.0,
        radius_physical=7.0,
        drive_physical=(2.0 / 0.7),
        length_scale=15.0,
    )
    theory = compute_flat_disk_kh_physical_theory(params)
    radius = float(theory.radius)
    lam = float(theory.lambda_value)

    mesh = _load_mesh_from_fixture(fixture_path)
    for _ in range(int(controls["refine_level"])):
        mesh = refine_triangle_mesh(mesh)
    if int(controls["rim_local_refine_steps"]) > 0:
        mesh = _refine_mesh_locally_near_rim(
            mesh,
            local_steps=int(controls["rim_local_refine_steps"]),
            rim_radius=radius,
            band_half_width=float(controls["rim_local_refine_band_lambda"]) * lam,
        )
    if int(controls["outer_local_refine_steps"]) > 0:
        mesh = _refine_mesh_locally_in_outer_annulus(
            mesh,
            local_steps=int(controls["outer_local_refine_steps"]),
            r_min=radius + float(controls["outer_local_refine_rmin_lambda"]) * lam,
            r_max=radius + float(controls["outer_local_refine_rmax_lambda"]) * lam,
        )
    if int(controls["local_edge_flip_steps"]) > 0:
        mesh = _flip_edges_locally_in_annulus(
            mesh,
            local_steps=int(controls["local_edge_flip_steps"]),
            r_min=max(
                0.0, radius + float(controls["local_edge_flip_rmin_lambda"]) * lam
            ),
            r_max=max(
                0.0, radius + float(controls["local_edge_flip_rmax_lambda"]) * lam
            ),
        )

    _configure_benchmark_mesh(
        mesh,
        theory_params=params,
        parameterization="kh_physical",
        outer_mode="disabled",
        smoothness_model="splay_twist",
        splay_modulus_scale_in=1.0,
        tilt_mass_mode_in="consistent",
        tilt_solver="cg",
        tilt_post_relax_inner_steps=40,
        tilt_post_relax_step_size=0.005,
        tilt_post_relax_passes=1,
    )
    minim = _build_minimizer(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.project_tilts_to_tangent()
    max_repeats = int(relax_max_repeats)
    if max_repeats < 1:
        raise ValueError("relax_max_repeats must be >= 1")
    energy_tol = float(relax_energy_abs_tol)
    if energy_tol < 0.0:
        raise ValueError("relax_energy_abs_tol must be >= 0")
    total_energy = float("nan")
    previous_energy = None
    repeats_applied = 0
    relax_converged = False
    for repeat in range(max_repeats):
        total_energy = float(
            _run_theta_relaxation(
                minim,
                theta_value=float(theta),
                reset_outer=repeat == 0,
                reset_inner=repeat == 0,
            )
        )
        repeats_applied += 1
        if (
            previous_energy is not None
            and abs(total_energy - previous_energy) <= energy_tol
        ):
            relax_converged = True
            break
        previous_energy = total_energy

    positions = mesh.positions_view()
    tri_area, _, _, _, tri_rows = mesh.p1_triangle_shape_gradient_cache(
        positions=positions
    )
    if tri_rows is None or tri_rows.size == 0:
        raise ValueError("Mesh has no triangles after refinement")
    tri_rows_i = np.asarray(tri_rows, dtype=int)
    tri_area_f = np.asarray(tri_area, dtype=float)
    solved_tilts = mesh.tilts_in_view().copy(order="F")
    solved_bands = _vertex_bands(
        positions=positions,
        tri_rows=tri_rows_i,
        tri_area=tri_area_f,
        tilts=mesh.tilts_in_view(),
        radius=radius,
        lambda_value=lam,
    )

    mesh_bands = _mesh_internal_band_split(
        mesh,
        smoothness_model="splay_twist",
        radius=radius,
        lambda_value=lam,
        rim_half_width_lambda=1.0,
        outer_near_width_lambda=4.0,
    )
    theory_bands = _theory_term_band_split(
        theta=float(theta),
        kappa=float(theory.kappa),
        kappa_t=float(theory.kappa_t),
        radius=radius,
        lambda_value=lam,
        rim_half_width_lambda=1.0,
        outer_near_width_lambda=4.0,
    )
    theory_bands_finite_outer = _theory_term_band_split(
        theta=float(theta),
        kappa=float(theory.kappa),
        kappa_t=float(theory.kappa_t),
        radius=radius,
        lambda_value=lam,
        rim_half_width_lambda=1.0,
        outer_near_width_lambda=4.0,
        outer_r_max=float(np.max(np.linalg.norm(positions[:, :2], axis=1))),
    )
    near = _safe_ratio(
        float(mesh_bands["mesh_internal_outer_near"]),
        float(theory_bands["theory_internal_outer_near"]),
    )
    far = _safe_ratio(
        float(mesh_bands["mesh_internal_outer_far"]),
        float(theory_bands["theory_internal_outer_far"]),
    )

    section_energy_by_field: dict[str, dict[str, dict[str, float]]] = {
        "solved": _section_energy_summary(
            mesh_bands=mesh_bands, theory_bands=theory_bands
        )
    }
    section_energy_by_field_finite_outer_reference: dict[
        str, dict[str, dict[str, float]]
    ] = {
        "solved": _section_energy_summary(
            mesh_bands=mesh_bands, theory_bands=theory_bands_finite_outer
        )
    }
    bands_by_field: dict[str, list[dict[str, float | int | str]]] = {
        "solved": solved_bands
    }
    profile_error_by_field = {
        "solved": _field_profile_errors(
            positions=positions,
            tri_rows=tri_rows_i,
            tri_area=tri_area_f,
            tilts=solved_tilts,
            theta=float(theta),
            radius=radius,
            lambda_value=lam,
        )
    }

    _, r_hat, _ = _radial_frames(positions)
    t_rad = np.einsum("ij,ij->i", solved_tilts, r_hat)
    mesh.set_tilts_in_from_array(r_hat * t_rad[:, None])
    mesh.project_tilts_to_tangent()
    radial_mesh_bands = _mesh_internal_band_split(
        mesh,
        smoothness_model="splay_twist",
        radius=radius,
        lambda_value=lam,
        rim_half_width_lambda=1.0,
        outer_near_width_lambda=4.0,
    )
    section_energy_by_field["radial_only"] = _section_energy_summary(
        mesh_bands=radial_mesh_bands,
        theory_bands=theory_bands,
    )
    section_energy_by_field_finite_outer_reference["radial_only"] = (
        _section_energy_summary(
            mesh_bands=radial_mesh_bands,
            theory_bands=theory_bands_finite_outer,
        )
    )
    bands_by_field["radial_only"] = _vertex_bands(
        positions=positions,
        tri_rows=tri_rows_i,
        tri_area=tri_area_f,
        tilts=mesh.tilts_in_view(),
        radius=radius,
        lambda_value=lam,
    )
    profile_error_by_field["radial_only"] = _field_profile_errors(
        positions=positions,
        tri_rows=tri_rows_i,
        tri_area=tri_area_f,
        tilts=mesh.tilts_in_view(),
        theta=float(theta),
        radius=radius,
        lambda_value=lam,
    )

    if bool(include_frozen_analytic):
        mesh.set_tilts_in_from_array(
            _frozen_analytic_tilts(
                positions=positions,
                theta=float(theta),
                radius=radius,
                lambda_value=lam,
            )
        )
        mesh.project_tilts_to_tangent()
        frozen_mesh_bands = _mesh_internal_band_split(
            mesh,
            smoothness_model="splay_twist",
            radius=radius,
            lambda_value=lam,
            rim_half_width_lambda=1.0,
            outer_near_width_lambda=4.0,
        )
        section_energy_by_field["frozen_analytic"] = _section_energy_summary(
            mesh_bands=frozen_mesh_bands,
            theory_bands=theory_bands,
        )
        section_energy_by_field_finite_outer_reference["frozen_analytic"] = (
            _section_energy_summary(
                mesh_bands=frozen_mesh_bands,
                theory_bands=theory_bands_finite_outer,
            )
        )
        bands_by_field["frozen_analytic"] = _vertex_bands(
            positions=positions,
            tri_rows=tri_rows_i,
            tri_area=tri_area_f,
            tilts=mesh.tilts_in_view(),
            radius=radius,
            lambda_value=lam,
        )
        profile_error_by_field["frozen_analytic"] = _field_profile_errors(
            positions=positions,
            tri_rows=tri_rows_i,
            tri_area=tri_area_f,
            tilts=mesh.tilts_in_view(),
            theta=float(theta),
            radius=radius,
            lambda_value=lam,
        )

    mesh.set_tilts_in_from_array(solved_tilts)
    mesh.project_tilts_to_tangent()

    return {
        "meta": {
            "mode": "flat_disk_kh_outer_vertex_audit",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "optimize_preset": str(optimize_preset),
            "theta": float(theta),
            "include_frozen_analytic": bool(include_frozen_analytic),
            "relax_max_repeats": max_repeats,
            "relax_energy_abs_tol": energy_tol,
            "relax_repeats_applied": repeats_applied,
            "relax_converged": relax_converged,
            "continuum_field_model": "vector_field_radial_amplitude",
            "combined_reference_profile": "I1_inside_K1_outside",
            "smoothness_only_reference_profile": "r_over_R_inside_R_over_r_outside",
            "scalar_tex_profile": "I0_inside_K0_outside",
            "outer_reference_primary": "infinite",
            "outer_reference_secondary": "finite_outer_rmax",
            "controls_effective": controls,
        },
        "parity": {
            "mesh_total_energy": total_energy,
            "outer_near_ratio_mesh_over_theory": near,
            "outer_far_ratio_mesh_over_theory": far,
            "outer_tail_balance_score": float(
                np.hypot(np.log(max(near, 1e-18)), np.log(max(far, 1e-18)))
            ),
        },
        "bands": solved_bands,
        "bands_by_field": bands_by_field,
        "profile_error_by_field": profile_error_by_field,
        "section_energy_by_field": section_energy_by_field,
        "section_energy_by_field_finite_outer_reference": (
            section_energy_by_field_finite_outer_reference
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--optimize-preset", default="kh_strict_outerfield_tight")
    ap.add_argument("--theta", type=float, default=0.138)
    ap.add_argument("--include-frozen-analytic", action="store_true")
    ap.add_argument("--relax-max-repeats", type=int, default=5)
    ap.add_argument("--relax-energy-abs-tol", type=float, default=1.0e-8)
    ap.add_argument("--refine-level", type=int)
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    report = run_flat_disk_kh_outer_vertex_audit(
        fixture=args.fixture,
        optimize_preset=args.optimize_preset,
        theta=args.theta,
        include_frozen_analytic=bool(args.include_frozen_analytic),
        relax_max_repeats=int(args.relax_max_repeats),
        relax_energy_abs_tol=float(args.relax_energy_abs_tol),
        refine_level=args.refine_level,
    )
    out = Path(args.output)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
