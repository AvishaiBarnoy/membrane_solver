#!/usr/bin/env python3
"""Curved 1-disk theory benchmark on the coupled tensionless free-disk lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import i1, k1

from geometry.curvature import compute_curvature_data, compute_curvature_fields
from runtime.refinement import refine_triangle_mesh
from tools.diagnostics.curved_disk_theory import (
    CurvedDiskTheoryParams,
    CurvedDiskTheoryResult,
    compute_curved_disk_theory,
    tex_reference_params,
)
from tools.diagnostics.free_disk_energy_split import (
    _bending_tilt_energy,
    _energy_breakdown,
    _inner_leaflet_vertex_split,
    _split_masks,
    _tilt_energy,
)
from tools.diagnostics.free_disk_profile_fits import _fit_i1, _fit_k1, _fit_log
from tools.diagnostics.free_disk_profile_protocol import (
    DEFAULT_FREE_DISK_CURVED_BILAYER_SOURCE,
    DEFAULT_FREE_DISK_FIXTURE,
    _configure_shape_relax,
    configure_free_disk_curved_bilayer_stage2,
    load_free_disk_curved_bilayer_mesh,
    load_free_disk_theory_mesh,
    measure_free_disk_curved_bilayer_near_rim,
    optimize_free_disk_theta_b,
)

REFINE_STEPS = 1
THETA_SCANS = 4
SHAPE_STEPS = 60
THETA_OFFSETS = (-0.02, 0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14)
INNER_I1_WINDOW = (0.25, 0.75)
OUTER_K1_WINDOW = (2.0, 10.0)
OUTER_LOG_WINDOW = (3.0, 10.0)
K1_SENS_WINDOWS = ((2.0, 8.0), (2.0, 10.0), (2.0, 16.0))
LOG_SENS_WINDOWS = ((3.0, 8.0), (3.0, 10.0), (3.0, 16.0))


def _refine_once(mesh, *, steps: int = REFINE_STEPS):
    """Return ``mesh`` refined ``steps`` times."""
    for _ in range(int(steps)):
        mesh = refine_triangle_mesh(mesh)
    return mesh


def _shell_profile(mesh) -> list[dict[str, float]]:
    """Return ring-median profile rows keyed by rounded radius."""
    from tools.diagnostics.utils import positions_radii

    positions = mesh.positions_view()
    radii = positions_radii(mesh)
    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    r_hat = np.zeros_like(positions)
    good = radii > 1.0e-12
    r_hat[good, 0] = positions[good, 0] / radii[good]
    r_hat[good, 1] = positions[good, 1] / radii[good]
    theta_in = np.einsum("ij,ij->i", tilts_in, r_hat)
    theta_out = np.einsum("ij,ij->i", tilts_out, r_hat)

    fields = compute_curvature_fields(mesh, positions, mesh.vertex_index_to_row)
    mean_curvature = fields.mean_curvature

    rows: list[dict[str, float]] = []
    for radius_key in sorted({round(float(r), 6) for r in radii if r > 1.0e-12}):
        mask = np.isclose(radii, float(radius_key), atol=1.0e-6)
        if not np.any(mask):
            continue
        rows.append(
            {
                "radius": float(np.median(radii[mask])),
                "theta_in": float(np.median(theta_in[mask])),
                "theta_out": float(np.median(theta_out[mask])),
                "theta_shared": float(
                    0.5 * (np.median(theta_in[mask]) + np.median(theta_out[mask]))
                ),
                "z": float(np.median(positions[mask, 2])),
                "J": float(np.median(mean_curvature[mask])),
                "count": int(np.sum(mask)),
            }
        )
    return rows


def _window_rows(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    window: tuple[float, float],
    last_free_shell_radius: float,
) -> list[dict[str, float]]:
    """Return shell rows whose radius lies inside a benchmark window."""
    r_lo = float(window[0]) * float(radius)
    r_hi = min(float(window[1]) * float(radius), float(last_free_shell_radius) - 1.0e-6)
    return [row for row in shell_rows if r_lo <= float(row["radius"]) <= r_hi]


def _relative_rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Return RMSE normalized by the max signal magnitude."""
    scale = max(float(np.max(np.abs(y))), 1.0e-12)
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / scale)


def _fit_inner_i1(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    lambda_theory: float,
    last_free_shell_radius: float,
) -> dict[str, object]:
    """Fit the inner disk median tilt to the ``I1`` theory form."""
    rows = _window_rows(
        shell_rows,
        radius=radius,
        window=INNER_I1_WINDOW,
        last_free_shell_radius=last_free_shell_radius,
    )
    r = np.asarray([float(row["radius"]) for row in rows], dtype=float)
    y = np.asarray([float(row["theta_in"]) for row in rows], dtype=float)
    amp, lam_fit, _ = _fit_i1(r, y, float(radius))
    yhat = amp * i1(lam_fit * r) / i1(lam_fit * float(radius))
    return {
        "window": [float(INNER_I1_WINDOW[0]), float(INNER_I1_WINDOW[1])],
        "count": int(len(r)),
        "radii": [float(v) for v in r],
        "amplitude_fit": float(amp),
        "lambda_fit": float(lam_fit),
        "lambda_ratio": float(lam_fit / lambda_theory),
        "rel_rmse": _relative_rmse(y, yhat),
    }


def _fit_outer_k1(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    lambda_theory: float,
    last_free_shell_radius: float,
    window: tuple[float, float],
) -> dict[str, object]:
    """Fit the shared outer tilt to the ``K1`` theory form."""
    rows = _window_rows(
        shell_rows,
        radius=radius,
        window=window,
        last_free_shell_radius=last_free_shell_radius,
    )
    r = np.asarray([float(row["radius"]) for row in rows], dtype=float)
    y_in = np.asarray([float(row["theta_in"]) for row in rows], dtype=float)
    y_out = np.asarray([float(row["theta_out"]) for row in rows], dtype=float)
    y = 0.5 * (y_in + y_out)
    amp, lam_fit, _ = _fit_k1(r, y, float(radius))
    yhat = amp * k1(lam_fit * r) / k1(lam_fit * float(radius))
    signal = np.maximum(np.abs(y), 1.0e-12)
    leaflet_mismatch = np.median(np.abs(y_in - y_out) / signal)
    return {
        "window": [float(window[0]), float(window[1])],
        "count": int(len(r)),
        "radii": [float(v) for v in r],
        "amplitude_fit": float(amp),
        "lambda_fit": float(lam_fit),
        "lambda_ratio": float(lam_fit / lambda_theory),
        "rel_rmse": _relative_rmse(y, yhat),
        "leaflet_mismatch_median": float(leaflet_mismatch),
    }


def _fit_outer_log_height(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    slope_theory: float,
    last_free_shell_radius: float,
    window: tuple[float, float],
) -> dict[str, object]:
    """Fit the outer shell-median height to the tensionless logarithmic law."""
    rows = _window_rows(
        shell_rows,
        radius=radius,
        window=window,
        last_free_shell_radius=last_free_shell_radius,
    )
    r = np.asarray([float(row["radius"]) for row in rows], dtype=float)
    z = np.asarray([float(row["z"]) for row in rows], dtype=float)
    z0, slope_fit, _ = _fit_log(r, z, float(radius))
    zhat = z0 + slope_fit * np.log(r / float(radius))
    return {
        "window": [float(window[0]), float(window[1])],
        "count": int(len(r)),
        "radii": [float(v) for v in r],
        "z0_fit": float(z0),
        "slope_fit": float(slope_fit),
        "slope_ratio": float(slope_fit / slope_theory),
        "rel_rmse": _relative_rmse(z, zhat),
    }


def _outer_curvature_summary(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    last_free_shell_radius: float,
) -> dict[str, float]:
    """Return mean and tail curvature summary on the free outer membrane."""
    rows = [
        row
        for row in shell_rows
        if float(row["radius"]) > float(radius) + 1.0e-6
        and float(row["radius"]) < float(last_free_shell_radius) + 1.0e-6
    ]
    abs_j = np.asarray([abs(float(row["J"])) for row in rows], dtype=float)
    return {
        "count": int(abs_j.size),
        "mean_abs_J": float(np.mean(abs_j)) if abs_j.size else 0.0,
        "p95_abs_J": float(np.percentile(abs_j, 95.0)) if abs_j.size else 0.0,
    }


def _compute_numeric_energy_split(mesh) -> dict[str, float]:
    """Return the disk/outer/contact split for the final curved benchmark mesh."""
    positions = mesh.positions_view()
    tri_rows_full, _ = mesh.triangle_row_cache()
    if tri_rows_full is None or tri_rows_full.size == 0:
        raise ValueError("No triangles found in benchmark mesh.")

    _, _, weights_full, tri_rows_curv = compute_curvature_data(
        mesh, positions, mesh.vertex_index_to_row
    )
    if tri_rows_curv.shape[0] != tri_rows_full.shape[0]:
        raise ValueError("Triangle rows mismatch between caches.")

    inner_split = _inner_leaflet_vertex_split(
        mesh=mesh,
        positions=positions,
        tri_rows_full=tri_rows_full,
        weights_full=weights_full,
    )
    tri_mask_disk, tri_mask_outer = _split_masks(mesh, tri_rows_full)
    del tri_mask_disk  # Disk contribution is already carried by the inner split.

    tilt_out_outer = _tilt_energy(
        positions=positions,
        tri_rows=tri_rows_full[tri_mask_outer],
        tilts=mesh.tilts_out_view(),
        k_tilt=float(mesh.global_parameters.get("tilt_modulus_out") or 0.0),
    )
    bend_out_outer = _bending_tilt_energy(
        mesh=mesh,
        global_params=mesh.global_parameters,
        positions=positions,
        tri_rows_full=tri_rows_full,
        weights_full=weights_full,
        tri_mask=tri_mask_outer,
        tilts=mesh.tilts_out_view(),
        kappa_key="bending_modulus_out",
        cache_tag="out",
    )

    breakdown = _energy_breakdown(mesh)
    contact = float(breakdown.get("tilt_thetaB_contact_in") or 0.0)
    disk_total = float(inner_split["tilt_in_disk"]) + float(
        inner_split["bending_tilt_in_disk"]
    )
    outer_total = (
        float(inner_split["tilt_in_outer"])
        + float(inner_split["bending_tilt_in_outer"])
        + float(tilt_out_outer)
        + float(bend_out_outer)
    )
    return {
        "total_numeric": float(sum(float(v) for v in breakdown.values())),
        "inner_elastic_numeric": float(disk_total),
        "outer_elastic_numeric": float(outer_total),
        "contact_numeric": float(contact),
    }


def _run_curved_theta_candidate(
    theta_b: float,
    *,
    curved_path: str | Path | None = None,
) -> dict[str, object]:
    """Run one refined curved candidate and return mesh, near-rim stats, and energy."""
    mesh = _refine_once(load_free_disk_curved_bilayer_mesh(curved_path))
    configure_free_disk_curved_bilayer_stage2(mesh, theta_b=float(theta_b), z_bump=None)
    minim = _configure_shape_relax(mesh, theta_b=float(theta_b))
    minim.minimize(n_steps=SHAPE_STEPS)
    breakdown = minim.compute_energy_breakdown()
    row = measure_free_disk_curved_bilayer_near_rim(mesh, theta_b=float(theta_b))
    row["total_energy"] = float(sum(float(v) for v in breakdown.values()))
    return {"mesh": mesh, "near_rim": row, "breakdown": breakdown}


def _run_canonical_schedule(
    *,
    theta_path: str | Path | None = None,
    curved_path: str | Path | None = None,
) -> dict[str, object]:
    """Run the benchmark's exact three-stage canonical schedule."""
    theta_mesh = _refine_once(load_free_disk_theory_mesh(theta_path))
    theta_seed = float(optimize_free_disk_theta_b(theta_mesh, scans=THETA_SCANS))
    theta_values = sorted(
        {
            round(max(0.0, float(theta_seed) + float(delta)), 8)
            for delta in THETA_OFFSETS
        }
    )

    sweep_rows: list[dict[str, float]] = []
    best_theta_b = None
    best_total_energy = None
    for theta_b in theta_values:
        result = _run_curved_theta_candidate(theta_b, curved_path=curved_path)
        row = dict(result["near_rim"])
        row["total_energy"] = float(row["total_energy"])
        sweep_rows.append(row)
        if best_total_energy is None or float(row["total_energy"]) < best_total_energy:
            best_total_energy = float(row["total_energy"])
            best_theta_b = float(theta_b)

    if best_theta_b is None:
        raise AssertionError("Curved local theta scan produced no valid candidate.")

    final = _run_curved_theta_candidate(best_theta_b, curved_path=curved_path)
    return {
        "theta_seed": float(theta_seed),
        "theta_values": [float(v) for v in theta_values],
        "theta_sweep_rows": sweep_rows,
        "theta_B_selected": float(best_theta_b),
        "theta_selected_energy": float(best_total_energy),
        "mesh": final["mesh"],
        "near_rim": dict(final["near_rim"]),
    }


def _outer_window_sensitivity(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    lambda_theory: float,
    slope_theory: float,
    last_free_shell_radius: float,
) -> dict[str, object]:
    """Return the explicit outer-window sensitivity audit required for lock."""
    k1_fits = [
        _fit_outer_k1(
            shell_rows,
            radius=radius,
            lambda_theory=lambda_theory,
            last_free_shell_radius=last_free_shell_radius,
            window=window,
        )
        for window in K1_SENS_WINDOWS
    ]
    log_fits = [
        _fit_outer_log_height(
            shell_rows,
            radius=radius,
            slope_theory=slope_theory,
            last_free_shell_radius=last_free_shell_radius,
            window=window,
        )
        for window in LOG_SENS_WINDOWS
    ]

    primary_lambda = float(k1_fits[1]["lambda_fit"])
    primary_slope = float(log_fits[1]["slope_fit"])
    lambda_values = np.asarray(
        [float(row["lambda_fit"]) for row in k1_fits], dtype=float
    )
    slope_values = np.asarray(
        [float(row["slope_fit"]) for row in log_fits], dtype=float
    )
    lambda_spread = float(
        (np.max(lambda_values) - np.min(lambda_values))
        / max(abs(primary_lambda), 1.0e-12)
    )
    slope_spread = float(
        (np.max(slope_values) - np.min(slope_values)) / max(abs(primary_slope), 1.0e-12)
    )
    return {
        "outer_k1_windows": k1_fits,
        "outer_log_windows": log_fits,
        "lambda_fit_spread": lambda_spread,
        "log_slope_spread": slope_spread,
    }


def _theory_summary(theory: CurvedDiskTheoryResult) -> dict[str, float]:
    """Return the benchmark theory targets as a flat serializable mapping."""
    return {
        "radius": float(theory.params.radius),
        "drive": float(theory.params.drive),
        "kappa": float(theory.params.kappa),
        "kappa_t": float(theory.params.kappa_t),
        "surface_tension": float(theory.params.surface_tension),
        "lambda_theory": float(theory.lambda_value),
        "theta_B_opt": float(theory.theta_star),
        "phi_star": float(theory.phi_star),
        "F_in_el": float(theory.elastic_inner),
        "F_out_el": float(theory.elastic_outer),
        "F_cont": float(theory.contact),
        "F_tot": float(theory.total),
    }


def run_curved_1disk_theory_benchmark(
    *,
    theta_path: str | Path | None = None,
    curved_path: str | Path | None = None,
    theory_params: CurvedDiskTheoryParams | None = None,
) -> dict[str, object]:
    """Run the full curved 1-disk benchmark and return a report dictionary."""
    params = theory_params or tex_reference_params()
    theory = compute_curved_disk_theory(params)
    schedule = _run_canonical_schedule(theta_path=theta_path, curved_path=curved_path)
    mesh = schedule["mesh"]
    shell_rows = _shell_profile(mesh)

    radius = float(params.radius)
    radii = [float(row["radius"]) for row in shell_rows]
    max_radius = float(max(radii))
    free_outer_radii = sorted(
        rr for rr in radii if rr > radius + 1.0e-6 and rr < max_radius - 1.0e-6
    )
    if not free_outer_radii:
        raise AssertionError("Benchmark mesh has no free outer shells.")
    last_free_shell_radius = float(free_outer_radii[-1])

    near_rim_raw = dict(schedule["near_rim"])
    half_theta = 0.5 * float(schedule["theta_B_selected"])
    near_rim = {
        **near_rim_raw,
        "phi_over_theta_B": float(
            abs(near_rim_raw["phi"]) / max(float(schedule["theta_B_selected"]), 1.0e-12)
        ),
        "theta_in_over_half_theta_B": float(
            near_rim_raw["theta_outer_in"] / max(abs(half_theta), 1.0e-12)
        ),
        "theta_out_over_half_theta_B": float(
            near_rim_raw["theta_outer_out"] / max(abs(half_theta), 1.0e-12)
        ),
    }

    fits = {
        "inner_i1": _fit_inner_i1(
            shell_rows,
            radius=radius,
            lambda_theory=float(theory.lambda_value),
            last_free_shell_radius=last_free_shell_radius,
        ),
        "outer_k1": _fit_outer_k1(
            shell_rows,
            radius=radius,
            lambda_theory=float(theory.lambda_value),
            last_free_shell_radius=last_free_shell_radius,
            window=OUTER_K1_WINDOW,
        ),
        "outer_height_log": _fit_outer_log_height(
            shell_rows,
            radius=radius,
            slope_theory=float(theory.phi_star) * radius,
            last_free_shell_radius=last_free_shell_radius,
            window=OUTER_LOG_WINDOW,
        ),
    }
    outer_sensitivity = _outer_window_sensitivity(
        shell_rows,
        radius=radius,
        lambda_theory=float(theory.lambda_value),
        slope_theory=float(theory.phi_star) * radius,
        last_free_shell_radius=last_free_shell_radius,
    )
    curvature = _outer_curvature_summary(
        shell_rows, radius=radius, last_free_shell_radius=last_free_shell_radius
    )
    energies = _compute_numeric_energy_split(mesh)

    failures: list[str] = []
    theta_b_num = float(schedule["theta_B_selected"])
    if not np.isclose(theta_b_num, float(theory.theta_star), rtol=0.10, atol=0.0):
        failures.append("theta_B_opt")
    if not np.isclose(float(near_rim["phi_over_theta_B"]), 0.5, rtol=0.10, atol=0.0):
        failures.append("phi_star_ratio")
    if not np.isclose(
        float(near_rim["theta_in_over_half_theta_B"]), 1.0, rtol=0.15, atol=0.0
    ):
        failures.append("theta_in_half_split")
    if not np.isclose(
        float(near_rim["theta_out_over_half_theta_B"]), 1.0, rtol=0.15, atol=0.0
    ):
        failures.append("theta_out_half_split")
    if (
        not np.isclose(
            float(fits["inner_i1"]["lambda_fit"]), float(theory.lambda_value), rtol=0.10
        )
        or float(fits["inner_i1"]["rel_rmse"]) > 0.05
    ):
        failures.append("inner_i1_fit")
    if (
        not np.isclose(
            float(fits["outer_k1"]["lambda_fit"]), float(theory.lambda_value), rtol=0.10
        )
        or float(fits["outer_k1"]["rel_rmse"]) > 0.05
        or float(fits["outer_k1"]["leaflet_mismatch_median"]) > 0.05
    ):
        failures.append("outer_k1_fit")
    if (
        not np.isclose(
            float(fits["outer_height_log"]["slope_fit"]),
            float(theory.phi_star) * radius,
            rtol=0.15,
        )
        or float(fits["outer_height_log"]["rel_rmse"]) > 0.20
    ):
        failures.append("outer_height_log_fit")
    if float(curvature["mean_abs_J"]) > 0.05 or float(curvature["p95_abs_J"]) > 0.15:
        failures.append("outer_curvature")
    if not np.isclose(
        float(energies["total_numeric"]), float(theory.total), rtol=0.10, atol=0.0
    ):
        failures.append("total_energy")
    if not np.isclose(
        float(energies["inner_elastic_numeric"]),
        float(theory.elastic_inner),
        rtol=0.15,
        atol=0.0,
    ):
        failures.append("inner_elastic")
    if not np.isclose(
        float(energies["outer_elastic_numeric"]),
        float(theory.elastic_outer),
        rtol=0.15,
        atol=0.0,
    ):
        failures.append("outer_elastic")
    if not np.isclose(
        float(energies["contact_numeric"]), float(theory.contact), rtol=0.15, atol=0.0
    ):
        failures.append("contact_energy")
    if float(outer_sensitivity["lambda_fit_spread"]) > 0.10:
        failures.append("outer_k1_window_sensitivity")
    if float(outer_sensitivity["log_slope_spread"]) > 0.10:
        failures.append("outer_log_window_sensitivity")

    return {
        "canonical_schedule": {
            "theta_scans": THETA_SCANS,
            "theta_offsets": [float(v) for v in THETA_OFFSETS],
            "shape_steps": SHAPE_STEPS,
            "refine_steps": REFINE_STEPS,
        },
        "theory": _theory_summary(theory),
        "theta_seed": float(schedule["theta_seed"]),
        "theta_values": [float(v) for v in schedule["theta_values"]],
        "theta_sweep_rows": schedule["theta_sweep_rows"],
        "theta_B_selected": float(schedule["theta_B_selected"]),
        "near_rim": near_rim,
        "fits": fits,
        "outer_curvature": curvature,
        "energies": energies,
        "outer_window_sensitivity": outer_sensitivity,
        "last_free_shell_radius": float(last_free_shell_radius),
        "shell_rows": shell_rows,
        "benchmark_lock_failures": failures,
        "benchmark_lock_passed": not failures,
    }


def main() -> None:
    """Run the benchmark and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theta-path",
        default=str(DEFAULT_FREE_DISK_FIXTURE),
        help="Path to the theory/theta fixture YAML.",
    )
    parser.add_argument(
        "--curved-path",
        default=str(DEFAULT_FREE_DISK_CURVED_BILAYER_SOURCE),
        help="Path to the curved free-disk source YAML.",
    )
    args = parser.parse_args()

    report = run_curved_1disk_theory_benchmark(
        theta_path=args.theta_path,
        curved_path=args.curved_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
