#!/usr/bin/env python3
"""Forced-theta diagnostic for the curved 1-disk coupled response."""

from __future__ import annotations

import argparse
import json
from typing import Iterable

import numpy as np
from scipy.special import k1

from geometry.curvature import compute_curvature_fields
from tools.diagnostics.curved_1disk_theory_benchmark import (
    OUTER_K1_WINDOW,
    OUTER_LOG_WINDOW,
    _compute_numeric_energy_split,
    _fit_outer_k1,
    _fit_outer_log_height,
    _run_curved_theta_candidate,
    _shell_profile,
)
from tools.diagnostics.curved_disk_theory import (
    compute_curved_disk_theory,
    tex_reference_params,
)

FORCED_THETA_VALUES = (0.06, 0.12, 0.1845693593)
SCATTER_JUDGMENT_THRESHOLD = 0.10


def _positions_radii(mesh) -> np.ndarray:
    """Return radial distance in the xy plane for every vertex."""
    return np.linalg.norm(mesh.positions_view()[:, :2], axis=1)


def _window_rows(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    window: tuple[float, float],
    last_free_shell_radius: float,
) -> list[dict[str, float]]:
    """Return shell rows inside a benchmark radial window."""
    r_lo = float(window[0]) * float(radius)
    r_hi = min(float(window[1]) * float(radius), float(last_free_shell_radius) - 1.0e-6)
    return [row for row in shell_rows if r_lo <= float(row["radius"]) <= r_hi]


def _relative_rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Return RMSE normalized by the max signal magnitude."""
    scale = max(float(np.max(np.abs(y))), 1.0e-12)
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / scale)


def _shell_vertex_samples(mesh) -> dict[float, dict[str, np.ndarray | float | int]]:
    """Return per-shell raw vertex samples for scatter diagnostics."""
    positions = mesh.positions_view()
    radii = _positions_radii(mesh)
    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    r_hat = np.zeros_like(positions)
    good = radii > 1.0e-12
    r_hat[good, 0] = positions[good, 0] / radii[good]
    r_hat[good, 1] = positions[good, 1] / radii[good]
    theta_in = np.einsum("ij,ij->i", tilts_in, r_hat)
    theta_out = np.einsum("ij,ij->i", tilts_out, r_hat)
    theta_shared = 0.5 * (theta_in + theta_out)
    fields = compute_curvature_fields(mesh, positions, mesh.vertex_index_to_row)
    mean_curvature = fields.mean_curvature

    samples: dict[float, dict[str, np.ndarray | float | int]] = {}
    for radius_key in sorted({round(float(r), 6) for r in radii if r > 1.0e-12}):
        mask = np.isclose(radii, float(radius_key), atol=1.0e-6)
        if not np.any(mask):
            continue
        rr = float(np.median(radii[mask]))
        samples[rr] = {
            "radius": rr,
            "count": int(np.sum(mask)),
            "theta_in": np.asarray(theta_in[mask], dtype=float),
            "theta_out": np.asarray(theta_out[mask], dtype=float),
            "theta_shared": np.asarray(theta_shared[mask], dtype=float),
            "z": np.asarray(positions[mask, 2], dtype=float),
            "J": np.asarray(mean_curvature[mask], dtype=float),
        }
    return samples


def _shell_mad(values: np.ndarray) -> float:
    """Return the median absolute deviation."""
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _scatter_summary(
    samples: dict[float, dict[str, np.ndarray | float | int]],
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    last_free_shell_radius: float,
) -> dict[str, object]:
    """Return azimuthal scatter summaries over the outer region."""
    outer_rows = [
        row
        for row in shell_rows
        if float(row["radius"]) > float(radius) + 1.0e-6
        and float(row["radius"]) < float(last_free_shell_radius) + 1.0e-6
    ]
    per_shell: list[dict[str, float]] = []
    for row in outer_rows:
        rr = float(row["radius"])
        shell = samples[rr]
        theta_shared = np.asarray(shell["theta_shared"], dtype=float)
        z_vals = np.asarray(shell["z"], dtype=float)
        j_vals = np.asarray(shell["J"], dtype=float)
        theta_med = float(np.median(theta_shared))
        z_med = float(np.median(z_vals))
        j_med = float(np.median(j_vals))
        per_shell.append(
            {
                "radius": rr,
                "count": int(shell["count"]),
                "theta_shared_median": theta_med,
                "theta_shared_rel_mad": _shell_mad(theta_shared)
                / max(abs(theta_med), 1.0e-12),
                "z_median": z_med,
                "z_rel_mad": _shell_mad(z_vals) / max(abs(z_med), 1.0e-12),
                "J_median": j_med,
                "J_rel_mad": _shell_mad(j_vals) / max(abs(j_med), 1.0e-12),
            }
        )

    def _worst_in_window(window: tuple[float, float], field: str) -> float:
        rows = _window_rows(
            per_shell,
            radius=radius,
            window=window,
            last_free_shell_radius=last_free_shell_radius,
        )
        if not rows:
            return 0.0
        return float(max(float(row[field]) for row in rows))

    worst_k1 = _worst_in_window(OUTER_K1_WINDOW, "theta_shared_rel_mad")
    worst_log = _worst_in_window(OUTER_LOG_WINDOW, "z_rel_mad")
    judgment = (
        "axisymmetric enough"
        if worst_k1 <= SCATTER_JUDGMENT_THRESHOLD
        and worst_log <= SCATTER_JUDGMENT_THRESHOLD
        else "noticeable azimuthal scatter"
    )
    return {
        "per_shell": per_shell,
        "worst_theta_shared_rel_scatter_outer_k1_window": worst_k1,
        "worst_z_rel_scatter_outer_log_window": worst_log,
        "judgment": judgment,
    }


def _outer_leaflet_mismatch_summary(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    last_free_shell_radius: float,
) -> dict[str, object]:
    """Return distal/proximal mismatch across the outer fit window."""
    rows = _window_rows(
        shell_rows,
        radius=radius,
        window=OUTER_K1_WINDOW,
        last_free_shell_radius=last_free_shell_radius,
    )
    shell_table: list[dict[str, float]] = []
    rels: list[float] = []
    for row in rows:
        signal = max(abs(float(row["theta_shared"])), 1.0e-12)
        rel = abs(float(row["theta_in"]) - float(row["theta_out"])) / signal
        rels.append(rel)
        shell_table.append(
            {
                "radius": float(row["radius"]),
                "theta_in": float(row["theta_in"]),
                "theta_out": float(row["theta_out"]),
                "theta_shared": float(row["theta_shared"]),
                "relative_mismatch": float(rel),
            }
        )
    rel_arr = np.asarray(rels, dtype=float)
    return {
        "shell_count": int(len(rows)),
        "shell_table": shell_table,
        "median_relative_mismatch": float(np.median(rel_arr)) if rel_arr.size else 0.0,
        "max_relative_mismatch": float(np.max(rel_arr)) if rel_arr.size else 0.0,
    }


def _outer_k1_summary(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    lambda_theory: float,
    theta_b: float,
    last_free_shell_radius: float,
) -> dict[str, object]:
    """Return outer tilt comparison to the theoretical K1 law."""
    fit = _fit_outer_k1(
        shell_rows,
        radius=radius,
        lambda_theory=lambda_theory,
        last_free_shell_radius=last_free_shell_radius,
        window=OUTER_K1_WINDOW,
    )
    rows = _window_rows(
        shell_rows,
        radius=radius,
        window=OUTER_K1_WINDOW,
        last_free_shell_radius=last_free_shell_radius,
    )
    theory_denom = float(k1(float(lambda_theory) * float(radius)))
    shell_table = [
        {
            "radius": float(row["radius"]),
            "theta_in": float(row["theta_in"]),
            "theta_out": float(row["theta_out"]),
            "theta_shared": float(row["theta_shared"]),
            "theta_K1_theory": float(
                0.5
                * float(theta_b)
                * float(k1(float(lambda_theory) * float(row["radius"])))
                / theory_denom
            ),
        }
        for row in rows
    ]
    return {**fit, "shell_count": int(len(rows)), "shell_table": shell_table}


def _outer_height_summary(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    theta_b: float,
    last_free_shell_radius: float,
) -> dict[str, object]:
    """Return outer height comparison to the logarithmic theory law."""
    slope_theory = 0.5 * float(theta_b) * float(radius)
    fit = _fit_outer_log_height(
        shell_rows,
        radius=radius,
        slope_theory=slope_theory,
        last_free_shell_radius=last_free_shell_radius,
        window=OUTER_LOG_WINDOW,
    )
    rows = _window_rows(
        shell_rows,
        radius=radius,
        window=OUTER_LOG_WINDOW,
        last_free_shell_radius=last_free_shell_radius,
    )
    z0 = float(fit["z0_fit"])
    shell_table = []
    residuals = []
    for row in rows:
        rr = float(row["radius"])
        z_theory = z0 + slope_theory * float(np.log(rr / float(radius)))
        residual = float(row["z"]) - z_theory
        residuals.append(residual)
        shell_table.append(
            {
                "radius": rr,
                "z": float(row["z"]),
                "z_log_theory": z_theory,
                "residual": residual,
            }
        )
    res_arr = np.asarray(residuals, dtype=float)
    return {
        **fit,
        "shell_count": int(len(rows)),
        "theory_slope": slope_theory,
        "shell_table": shell_table,
        "max_abs_residual": float(np.max(np.abs(res_arr))) if res_arr.size else 0.0,
        "median_abs_residual": float(np.median(np.abs(res_arr)))
        if res_arr.size
        else 0.0,
    }


def _outer_slope_summary(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    theta_b: float,
    last_free_shell_radius: float,
) -> dict[str, object]:
    """Return slope comparison against phi_* R / r on the outer log window."""
    rows = _window_rows(
        shell_rows,
        radius=radius,
        window=OUTER_LOG_WINDOW,
        last_free_shell_radius=last_free_shell_radius,
    )
    if len(rows) < 3:
        return {
            "window": [float(OUTER_LOG_WINDOW[0]), float(OUTER_LOG_WINDOW[1])],
            "shell_count": 0,
            "shell_table": [],
            "rel_rmse": 0.0,
            "max_abs_residual": 0.0,
        }
    radii = np.asarray([float(row["radius"]) for row in rows], dtype=float)
    z_vals = np.asarray([float(row["z"]) for row in rows], dtype=float)
    r_mid = radii[1:-1]
    phi_num = (z_vals[2:] - z_vals[:-2]) / (radii[2:] - radii[:-2])
    phi_theory = 0.5 * float(theta_b) * float(radius) / r_mid
    residuals = phi_num - phi_theory
    shell_table = [
        {
            "radius": float(rr),
            "phi_numeric": float(pn),
            "phi_theory": float(pt),
            "residual": float(res),
        }
        for rr, pn, pt, res in zip(r_mid, phi_num, phi_theory, residuals)
    ]
    return {
        "window": [float(OUTER_LOG_WINDOW[0]), float(OUTER_LOG_WINDOW[1])],
        "shell_count": int(len(shell_table)),
        "shell_table": shell_table,
        "rel_rmse": _relative_rmse(phi_num, phi_theory),
        "max_abs_residual": float(np.max(np.abs(residuals))),
    }


def _outer_curvature_diagnostic(
    shell_rows: list[dict[str, float]],
    *,
    radius: float,
    last_free_shell_radius: float,
) -> dict[str, object]:
    """Return shellwise outer curvature plus summary statistics."""
    rows = [
        row
        for row in shell_rows
        if float(row["radius"]) > float(radius) + 1.0e-6
        and float(row["radius"]) < float(last_free_shell_radius) + 1.0e-6
    ]
    abs_j = np.asarray([abs(float(row["J"])) for row in rows], dtype=float)
    return {
        "shell_count": int(len(rows)),
        "shell_table": [
            {"radius": float(row["radius"]), "J": float(row["J"])} for row in rows
        ],
        "mean_abs_J": float(np.mean(abs_j)) if abs_j.size else 0.0,
        "p95_abs_J": float(np.percentile(abs_j, 95.0)) if abs_j.size else 0.0,
    }


def _classify_outer_elastic_scaling(
    cases: Iterable[dict[str, object]],
) -> tuple[str, dict[str, float]]:
    """Classify outer-elastic scaling with theta_B."""
    rows = sorted(cases, key=lambda row: float(row["theta_B"]))
    theta_vals = np.asarray([float(row["theta_B"]) for row in rows], dtype=float)
    normalized = np.asarray(
        [float(row["outer_elastic"]["outer_elastic_over_theta_B_sq"]) for row in rows],
        dtype=float,
    )
    median_norm = float(np.median(normalized))
    spread = float(
        (np.max(normalized) - np.min(normalized)) / max(abs(median_norm), 1.0e-12)
    )
    first = float(normalized[0])
    last = float(normalized[-1])
    if spread <= 0.15:
        kind = "approximately quadratic"
    elif last < first * (1.0 - 0.15):
        kind = "subquadratic"
    else:
        kind = "superquadratic"
    return kind, {
        "normalized_spread": spread,
        "theta_values": [float(v) for v in theta_vals],
        "normalized_values": [float(v) for v in normalized],
    }


def _diagnose(cases: list[dict[str, object]], *, radius: float) -> dict[str, object]:
    """Return the final diagnostic call and recommendation."""
    near_rim_good = all(
        abs(float(case["near_rim"]["phi_over_theta_B"]) - 0.5) <= 0.05
        and abs(float(case["near_rim"]["theta_in_over_half_theta_B"]) - 1.0) <= 0.15
        and abs(float(case["near_rim"]["theta_out_over_half_theta_B"]) - 1.0) <= 0.15
        for case in cases
    )
    outer_tilt_bad = any(
        float(case["outer_k1"]["rel_rmse"]) > 0.10
        or float(case["outer_leaflet_mismatch"]["median_relative_mismatch"]) > 0.05
        for case in cases
    )
    outer_shape_bad = any(
        float(case["outer_height_log"]["slope_fit"]) == pytest_approx_zero_placeholder()
        or float(case["outer_height_log"]["rel_rmse"]) > 0.20
        or float(case["outer_slope"]["rel_rmse"]) > 0.20
        for case in cases
    )
    mean_js = [float(case["outer_curvature"]["mean_abs_J"]) for case in cases]
    curvature_grows = mean_js[-1] > max(1.5 * mean_js[0], mean_js[0] + 0.02)

    if outer_tilt_bad and outer_shape_bad and curvature_grows:
        dominant = "coupling"
    elif outer_shape_bad and not outer_tilt_bad:
        dominant = "outer shape response"
    elif outer_tilt_bad and not outer_shape_bad:
        dominant = "outer tilt response"
    else:
        dominant = "coupling"

    totals = [float(case["total_energy"]) for case in cases]
    zspans = [float(case["z_span"]) for case in cases]
    if totals[0] < totals[1] < totals[2] and zspans[0] < zspans[1] < zspans[2]:
        branch_reason = (
            "the realized coupled energy genuinely favors a different branch"
        )
    else:
        branch_reason = "the geometry solver/path is too stiff or constrained"

    if dominant == "coupling":
        recommendation = (
            "Next stream should isolate the fixed-theta outer shape solve and shape-side "
            "constraints/energy contributions on the curved free-disk lane, using forced theta_B near theory."
        )
    elif dominant == "outer shape response":
        recommendation = "Next stream should isolate the outer geometry solve under fixed tilts and audit why the logarithmic trumpet shape is not realized."
    else:
        recommendation = "Next stream should isolate the outer leaflet transport/tilt discretization on the curved free-disk lane under fixed theta_B."

    return {
        "dominant_failure": dominant,
        "near_rim_half_split_stays_good": bool(near_rim_good),
        "branch_preference_interpretation": branch_reason,
        "recommended_next_stream": recommendation,
    }


def pytest_approx_zero_placeholder() -> float:
    """Return the exact zero threshold used in this diagnostic."""
    return 1.0e-12


def _run_forced_case(
    theta_b: float, *, radius: float, lambda_theory: float
) -> dict[str, object]:
    """Run one forced theta_B case and return its diagnostic payload."""
    result = _run_curved_theta_candidate(theta_b)
    mesh = result["mesh"]
    shell_rows = _shell_profile(mesh)
    radii = [float(row["radius"]) for row in shell_rows]
    max_radius = float(max(radii))
    free_outer_radii = sorted(
        rr for rr in radii if rr > float(radius) + 1.0e-6 and rr < max_radius - 1.0e-6
    )
    if not free_outer_radii:
        raise AssertionError("Forced-theta diagnostic found no free outer shells.")
    last_free_shell_radius = float(free_outer_radii[-1])
    samples = _shell_vertex_samples(mesh)
    near_rim_raw = dict(result["near_rim"])
    half_theta = 0.5 * float(theta_b)
    near_rim = {
        **near_rim_raw,
        "phi_over_theta_B": float(
            abs(near_rim_raw["phi"]) / max(float(theta_b), 1.0e-12)
        ),
        "theta_in_over_half_theta_B": float(
            near_rim_raw["theta_outer_in"] / max(abs(half_theta), 1.0e-12)
        ),
        "theta_out_over_half_theta_B": float(
            near_rim_raw["theta_outer_out"] / max(abs(half_theta), 1.0e-12)
        ),
    }
    outer_k1 = _outer_k1_summary(
        shell_rows,
        radius=radius,
        lambda_theory=lambda_theory,
        theta_b=theta_b,
        last_free_shell_radius=last_free_shell_radius,
    )
    outer_height = _outer_height_summary(
        shell_rows,
        radius=radius,
        theta_b=theta_b,
        last_free_shell_radius=last_free_shell_radius,
    )
    outer_slope = _outer_slope_summary(
        shell_rows,
        radius=radius,
        theta_b=theta_b,
        last_free_shell_radius=last_free_shell_radius,
    )
    outer_curvature = _outer_curvature_diagnostic(
        shell_rows, radius=radius, last_free_shell_radius=last_free_shell_radius
    )
    mismatch = _outer_leaflet_mismatch_summary(
        shell_rows, radius=radius, last_free_shell_radius=last_free_shell_radius
    )
    energies = _compute_numeric_energy_split(mesh)
    outer_elastic = {
        "outer_elastic_numeric": float(energies["outer_elastic_numeric"]),
        "outer_elastic_over_theta_B_sq": float(
            float(energies["outer_elastic_numeric"]) / max(float(theta_b) ** 2, 1.0e-12)
        ),
    }
    axisymmetry = _scatter_summary(
        samples,
        shell_rows,
        radius=radius,
        last_free_shell_radius=last_free_shell_radius,
    )
    return {
        "theta_B": float(theta_b),
        "phi_star_forced": float(0.5 * theta_b),
        "total_energy": float(result["near_rim"]["total_energy"]),
        "z_span": float(result["near_rim"]["z_span"]),
        "last_free_shell_radius": float(last_free_shell_radius),
        "near_rim": near_rim,
        "outer_k1": outer_k1,
        "outer_height_log": outer_height,
        "outer_slope": outer_slope,
        "outer_curvature": outer_curvature,
        "outer_leaflet_mismatch": mismatch,
        "outer_elastic": outer_elastic,
        "axisymmetry": axisymmetry,
    }


def run_curved_1disk_forced_theta_diagnostic() -> dict[str, object]:
    """Run the forced-theta curved diagnostic and return a JSON-serializable report."""
    theory = compute_curved_disk_theory(tex_reference_params())
    radius = float(theory.params.radius)
    lambda_theory = float(theory.lambda_value)
    cases = [
        _run_forced_case(
            theta_b=float(theta), radius=radius, lambda_theory=lambda_theory
        )
        for theta in FORCED_THETA_VALUES
    ]
    scaling_kind, scaling_stats = _classify_outer_elastic_scaling(cases)
    for case in cases:
        case["outer_elastic"]["scaling_classification"] = scaling_kind
    diagnosis = _diagnose(cases, radius=radius)
    return {
        "forced_theta_values": [float(v) for v in FORCED_THETA_VALUES],
        "theory": {
            "R": radius,
            "lambda_theory": lambda_theory,
        },
        "cases": cases,
        "outer_elastic_scaling": {
            "classification": scaling_kind,
            **scaling_stats,
        },
        "diagnosis": diagnosis,
    }


def main() -> None:
    """Run the forced-theta diagnostic and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = run_curved_1disk_forced_theta_diagnostic()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
