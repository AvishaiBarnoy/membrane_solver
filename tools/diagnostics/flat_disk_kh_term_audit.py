#!/usr/bin/env python3
"""Per-theta KH physical lane audit for flat one-leaflet disk benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import yaml

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


def _radial_frames(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return radius, radial unit vectors, and azimuthal unit vectors."""
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    phi_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    phi_hat[good, 0] = -positions[good, 1] / r[good]
    phi_hat[good, 1] = positions[good, 0] / r[good]
    return r, r_hat, phi_hat


def _triangle_centroid_radius(
    positions: np.ndarray,
    tri_rows: np.ndarray,
) -> np.ndarray:
    tri_cent = (
        positions[tri_rows[:, 0]]
        + positions[tri_rows[:, 1]]
        + positions[tri_rows[:, 2]]
    ) / 3.0
    return np.linalg.norm(tri_cent[:, :2], axis=1)


def _triangle_inside_fraction(
    positions: np.ndarray,
    tri_rows: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    """Return per-triangle inside fraction by vertex count (<R)."""
    tri_pos = positions[tri_rows]
    tri_r = np.linalg.norm(tri_pos[:, :, :2], axis=2)
    inside_counts = np.sum(tri_r <= float(radius), axis=1)
    return np.asarray(inside_counts, dtype=float) / 3.0


def _mesh_internal_region_split(
    mesh,
    *,
    smoothness_model: str,
    radius: float,
) -> dict[str, float]:
    """Split mesh internal energy into disk (r<R) and outer (r>R) regions."""
    from modules.energy import tilt_smoothness as tilt_smoothness_base

    gp = mesh.global_parameters
    positions = mesh.positions_view()
    tilts_in = mesh.tilts_in_view()

    area, g0, g1, g2, tri_rows = mesh.p1_triangle_shape_gradient_cache(
        positions=positions
    )
    if tri_rows is None or len(tri_rows) == 0:
        return {
            "mesh_internal_disk": 0.0,
            "mesh_internal_outer": 0.0,
            "mesh_internal_total_from_regions": 0.0,
            "mesh_tilt_disk": 0.0,
            "mesh_tilt_outer": 0.0,
            "mesh_smooth_disk": 0.0,
            "mesh_smooth_outer": 0.0,
        }

    disk_frac = _triangle_inside_fraction(positions, tri_rows, radius=float(radius))
    outer_frac = 1.0 - disk_frac

    k_tilt = float(gp.get("tilt_modulus_in") or 0.0)
    tilt_sq = np.einsum("ij,ij->i", tilts_in, tilts_in)
    tri_tilt_sq_sum = tilt_sq[tri_rows].sum(axis=1)
    tilt_tri = 0.5 * k_tilt * area * (tri_tilt_sq_sum / 3.0)

    if str(smoothness_model) == "splay_twist":
        k_splay = gp.get("tilt_splay_modulus_in")
        if k_splay is None:
            k_splay = gp.get("bending_modulus_in")
        if k_splay is None:
            k_splay = gp.get("bending_modulus")
        k_splay_f = float(k_splay or 0.0)
        k_twist_f = float(gp.get("tilt_twist_modulus_in") or 0.0)

        t0 = tilts_in[tri_rows[:, 0]]
        t1 = tilts_in[tri_rows[:, 1]]
        t2 = tilts_in[tri_rows[:, 2]]
        div_tri = (
            np.einsum("ij,ij->i", t0, g0)
            + np.einsum("ij,ij->i", t1, g1)
            + np.einsum("ij,ij->i", t2, g2)
        )

        n = mesh.triangle_normals(positions=positions)
        n_norm = np.linalg.norm(n, axis=1)
        n_hat = np.zeros_like(n)
        good = n_norm > 1e-20
        n_hat[good] = n[good] / n_norm[good, None]

        curl_vec = np.cross(g0, t0) + np.cross(g1, t1) + np.cross(g2, t2)
        curl_n = np.einsum("ij,ij->i", curl_vec, n_hat)
        smooth_tri = (
            0.5
            * area
            * ((k_splay_f * div_tri * div_tri) + (k_twist_f * curl_n * curl_n))
        )
    elif str(smoothness_model) == "dirichlet":
        k_smooth = gp.get("bending_modulus_in")
        if k_smooth is None:
            k_smooth = gp.get("bending_modulus")
        k_smooth_f = float(k_smooth or 0.0)
        weights, smooth_tri_rows = tilt_smoothness_base._get_weights_and_tris(
            mesh,
            positions=positions,
            index_map=mesh.vertex_index_to_row,
        )
        if smooth_tri_rows is None:
            smooth_tri = np.zeros_like(tilt_tri)
            smooth_frac = np.zeros_like(smooth_tri)
        else:
            c0 = weights[:, 0]
            c1 = weights[:, 1]
            c2 = weights[:, 2]
            t0 = tilts_in[smooth_tri_rows[:, 0]]
            t1 = tilts_in[smooth_tri_rows[:, 1]]
            t2 = tilts_in[smooth_tri_rows[:, 2]]
            d12 = t1 - t2
            d20 = t2 - t0
            d01 = t0 - t1
            smooth_tri = (
                0.25
                * k_smooth_f
                * (
                    c0 * np.einsum("ij,ij->i", d12, d12)
                    + c1 * np.einsum("ij,ij->i", d20, d20)
                    + c2 * np.einsum("ij,ij->i", d01, d01)
                )
            )
            smooth_frac = _triangle_inside_fraction(
                positions, smooth_tri_rows, radius=float(radius)
            )

        smooth_disk = float(np.sum(smooth_tri * smooth_frac))
        smooth_outer = float(np.sum(smooth_tri * (1.0 - smooth_frac)))
        tilt_disk = float(np.sum(tilt_tri * disk_frac))
        tilt_outer = float(np.sum(tilt_tri * outer_frac))
        return {
            "mesh_internal_disk": float(tilt_disk + smooth_disk),
            "mesh_internal_outer": float(tilt_outer + smooth_outer),
            "mesh_internal_total_from_regions": float(
                tilt_disk + smooth_disk + tilt_outer + smooth_outer
            ),
            "mesh_tilt_disk": tilt_disk,
            "mesh_tilt_outer": tilt_outer,
            "mesh_smooth_disk": smooth_disk,
            "mesh_smooth_outer": smooth_outer,
        }
    else:
        raise ValueError("smoothness_model must be 'dirichlet' or 'splay_twist'.")

    tilt_disk = float(np.sum(tilt_tri * disk_frac))
    tilt_outer = float(np.sum(tilt_tri * outer_frac))
    smooth_disk = float(np.sum(smooth_tri * disk_frac))
    smooth_outer = float(np.sum(smooth_tri * outer_frac))
    return {
        "mesh_internal_disk": float(tilt_disk + smooth_disk),
        "mesh_internal_outer": float(tilt_outer + smooth_outer),
        "mesh_internal_total_from_regions": float(
            tilt_disk + smooth_disk + tilt_outer + smooth_outer
        ),
        "mesh_tilt_disk": tilt_disk,
        "mesh_tilt_outer": tilt_outer,
        "mesh_smooth_disk": smooth_disk,
        "mesh_smooth_outer": smooth_outer,
    }


def _boundary_realization_metrics(
    mesh,
    *,
    radius: float,
    theta_value: float,
) -> dict[str, float]:
    """Measure realized radial tilt on the rim shell vs imposed theta_B."""
    pos = mesh.positions_view()
    r, r_hat, _ = _radial_frames(pos)
    shell_tol = max(1e-6, 0.02 * float(radius))
    rim_mask = np.abs(r - float(radius)) <= shell_tol
    rows = np.flatnonzero(rim_mask)
    if rows.size == 0:
        return {
            "rim_samples": 0,
            "rim_theta_error_abs_median": float("nan"),
            "rim_theta_error_abs_max": float("nan"),
            "rim_theta_realized_median": float("nan"),
        }
    t_in = mesh.tilts_in_view()
    t_rad = np.einsum("ij,ij->i", t_in[rows], r_hat[rows])
    err = t_rad - float(theta_value)
    return {
        "rim_samples": int(rows.size),
        "rim_theta_error_abs_median": float(np.median(np.abs(err))),
        "rim_theta_error_abs_max": float(np.max(np.abs(err))),
        "rim_theta_realized_median": float(np.median(t_rad)),
    }


def _leakage_metrics(mesh, *, radius: float) -> dict[str, float]:
    """Report azimuthal (t_phi) leakage relative to radial component."""
    pos = mesh.positions_view()
    r, r_hat, phi_hat = _radial_frames(pos)
    t_in = mesh.tilts_in_view()
    t_rad = np.einsum("ij,ij->i", t_in, r_hat)
    t_phi = np.einsum("ij,ij->i", t_in, phi_hat)

    def _ratio(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        num = float(np.median(np.abs(t_phi[mask])))
        den = float(np.median(np.abs(t_rad[mask])))
        return float(num / max(den, 1e-18))

    inner_mask = r < float(radius)
    outer_mask = r > float(radius)
    return {
        "inner_tphi_over_trad_median": _ratio(inner_mask),
        "outer_tphi_over_trad_median": _ratio(outer_mask),
    }


def _resolution_metrics(
    mesh, *, radius: float, lambda_value: float
) -> dict[str, float]:
    """Report rim edge-length scale relative to the decay length lambda."""
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return {
            "rim_edge_count": 0,
            "rim_edge_length_median": float("nan"),
            "rim_edge_length_max": float("nan"),
            "rim_h_over_lambda_median": float("nan"),
        }

    pos = mesh.positions_view()
    all_edges = np.vstack(
        [
            tri_rows[:, [0, 1]],
            tri_rows[:, [1, 2]],
            tri_rows[:, [2, 0]],
        ]
    )
    sorted_edges = np.sort(all_edges, axis=1)
    edges = np.unique(sorted_edges, axis=0)

    p0 = pos[edges[:, 0]]
    p1 = pos[edges[:, 1]]
    mid = 0.5 * (p0 + p1)
    mid_r = np.linalg.norm(mid[:, :2], axis=1)
    lengths = np.linalg.norm(p1 - p0, axis=1)
    rim_mask = (mid_r >= (0.9 * float(radius))) & (mid_r <= (1.1 * float(radius)))
    rim_lengths = lengths[rim_mask]
    if rim_lengths.size == 0:
        return {
            "rim_edge_count": 0,
            "rim_edge_length_median": float("nan"),
            "rim_edge_length_max": float("nan"),
            "rim_h_over_lambda_median": float("nan"),
        }

    h_med = float(np.median(rim_lengths))
    return {
        "rim_edge_count": int(rim_lengths.size),
        "rim_edge_length_median": h_med,
        "rim_edge_length_max": float(np.max(rim_lengths)),
        "rim_h_over_lambda_median": float(h_med / max(float(lambda_value), 1e-18)),
    }


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
    rim_local_refine_steps: int,
    rim_local_refine_band_lambda: float,
) -> dict[str, Any]:
    from runtime.refinement import refine_triangle_mesh
    from tools.diagnostics.flat_disk_one_leaflet_theory import (
        compute_flat_disk_kh_physical_theory,
        physical_to_dimensionless_theory_params,
    )
    from tools.reproduce_flat_disk_one_leaflet import (
        _build_minimizer,
        _configure_benchmark_mesh,
        _load_mesh_from_fixture,
        _refine_mesh_locally_near_rim,
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

    mesh = _load_mesh_from_fixture(fixture)
    for _ in range(int(refine_level)):
        mesh = refine_triangle_mesh(mesh)
    if int(rim_local_refine_steps) > 0:
        mesh = _refine_mesh_locally_near_rim(
            mesh,
            local_steps=int(rim_local_refine_steps),
            rim_radius=float(theory.radius),
            band_half_width=float(rim_local_refine_band_lambda)
            * float(theory.lambda_value),
        )

    mass_mode_raw = str(tilt_mass_mode_in).strip().lower()
    if mass_mode_raw == "auto":
        mass_mode = "consistent"
    elif mass_mode_raw in {"lumped", "consistent"}:
        mass_mode = mass_mode_raw
    else:
        raise ValueError("tilt_mass_mode_in must be 'auto', 'lumped', or 'consistent'.")

    _configure_benchmark_mesh(
        mesh,
        theory_params=params,
        parameterization="kh_physical",
        outer_mode=outer_mode,
        smoothness_model=smoothness_model,
        splay_modulus_scale_in=1.0,
        tilt_mass_mode_in=str(mass_mode),
    )
    minim = _build_minimizer(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.project_tilts_to_tangent()

    rows: list[dict[str, float]] = []
    for theta in np.asarray(theta_values, dtype=float).tolist():
        theta_f = float(theta)
        mesh_total = float(
            _run_theta_relaxation(minim, theta_value=theta_f, reset_outer=True)
        )
        breakdown = minim.compute_energy_breakdown()
        mesh_contact = float(breakdown.get("tilt_thetaB_contact_in", 0.0))
        mesh_internal = float(mesh_total - mesh_contact)
        mesh_region = _mesh_internal_region_split(
            mesh,
            smoothness_model=smoothness_model,
            radius=float(theory.radius),
        )

        th_in = float(c_in * theta_f * theta_f)
        th_out = float(c_out * theta_f * theta_f)
        th_contact = float(-b_contact * theta_f)
        th_total = float(th_in + th_out + th_contact)
        th_internal = float(th_in + th_out)

        boundary = _boundary_realization_metrics(
            mesh,
            radius=float(theory.radius),
            theta_value=theta_f,
        )
        leakage = _leakage_metrics(mesh, radius=float(theory.radius))

        rows.append(
            {
                "theta": theta_f,
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
                "contact_ratio_mesh_over_theory": (
                    float(mesh_contact / th_contact)
                    if abs(th_contact) > 1e-18
                    else float("nan")
                ),
                "internal_disk_ratio_mesh_over_theory": (
                    float(mesh_region["mesh_internal_disk"] / th_in)
                    if abs(th_in) > 1e-18
                    else float("nan")
                ),
                "internal_outer_ratio_mesh_over_theory": (
                    float(mesh_region["mesh_internal_outer"] / th_out)
                    if abs(th_out) > 1e-18
                    else float("nan")
                ),
                **boundary,
                **leakage,
            }
        )

    resolution = _resolution_metrics(
        mesh,
        radius=float(theory.radius),
        lambda_value=float(theory.lambda_value),
    )

    return {
        "meta": {
            "fixture": str(fixture.relative_to(ROOT)),
            "refine_level": int(refine_level),
            "outer_mode": str(outer_mode),
            "smoothness_model": str(smoothness_model),
            "parameterization": "kh_physical",
            "theory_model": "kh_physical_strict_kh",
            "kappa_physical": float(kappa_physical),
            "kappa_t_physical": float(kappa_t_physical),
            "radius_nm": float(radius_nm),
            "length_scale_nm": float(length_scale_nm),
            "drive_physical": float(drive_physical),
            "tilt_mass_mode_in": str(mass_mode),
            "rim_local_refine_steps": int(rim_local_refine_steps),
            "rim_local_refine_band_lambda": float(rim_local_refine_band_lambda),
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
    rim_local_refine_steps: int = 0,
    rim_local_refine_band_lambda: float = 0.0,
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
        rim_local_refine_steps=int(rim_local_refine_steps),
        rim_local_refine_band_lambda=float(rim_local_refine_band_lambda),
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
    rim_local_refine_steps: int = 0,
    rim_local_refine_band_lambda: float = 0.0,
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
            rim_local_refine_steps=int(rim_local_refine_steps),
            rim_local_refine_band_lambda=float(rim_local_refine_band_lambda),
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
            "rim_local_refine_steps": int(rim_local_refine_steps),
            "rim_local_refine_band_lambda": float(rim_local_refine_band_lambda),
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
        "kh_strict_continuity",
        "kh_strict_robust",
    ),
    refine_level: int = 1,
    rim_local_refine_steps: int = 1,
    rim_local_refine_band_lambda: float = 4.0,
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
        },
        "rows": rows,
        "selected_best": selected,
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
        "--optimize-preset",
        default="kh_wide",
    )
    ap.add_argument(
        "--optimize-presets",
        nargs="+",
        default=None,
    )
    ap.add_argument(
        "--theta-values", type=float, nargs="+", default=[0.0, 6.366e-4, 0.004]
    )
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if args.strict_preset_characterization:
        presets = (
            tuple(str(x) for x in args.optimize_presets)
            if args.optimize_presets is not None
            else ("kh_strict_fast", "kh_strict_continuity", "kh_strict_robust")
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
        )

    out = Path(args.output)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
