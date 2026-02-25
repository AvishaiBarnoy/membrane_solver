#!/usr/bin/env python3
"""KH strict error-source audit with deterministic effect ranking."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import yaml

from tools.diagnostics.flat_disk_kh_term_audit import (
    DEFAULT_FIXTURE,
    run_flat_disk_kh_term_audit,
)
from tools.reproduce_flat_disk_one_leaflet import run_flat_disk_one_leaflet_benchmark

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "diagnostics"
    / "flat_disk_kh_error_source_audit.yaml"
)


def _mean_abs(values: list[float]) -> float:
    if len(values) == 0:
        return 0.0
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr)))


def _rank_effects(effect_sizes: dict[str, float]) -> dict[str, Any]:
    items = sorted(effect_sizes.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    total = float(sum(max(float(v), 0.0) for _, v in items))
    dominant = items[0][0] if items else "none"
    confidence = float(items[0][1] / total) if total > 0.0 and items else 0.0
    return {
        "dominant_source": str(dominant),
        "confidence": float(confidence),
        "effect_sizes": {k: float(v) for k, v in items},
        "ranking": [k for k, _ in items],
    }


def run_flat_disk_kh_error_source_audit(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    primary_preset: str = "kh_strict_outerfield_tight",
    reference_preset: str = "kh_strict_outerband_tight",
    refine_levels: Sequence[int] = (2, 3),
    mass_modes: Sequence[str] = ("consistent", "lumped"),
    partition_modes: Sequence[str] = ("centroid", "fractional"),
) -> dict[str, Any]:
    """Audit outer-band mismatch source ranking for strict KH presets."""
    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    levels = [int(x) for x in refine_levels]
    if len(levels) == 0:
        raise ValueError("refine_levels must be non-empty.")
    masses = [str(x).strip().lower() for x in mass_modes]
    if any(x not in {"consistent", "lumped"} for x in masses):
        raise ValueError("mass_modes must contain only consistent|lumped.")
    partitions = [str(x).strip().lower() for x in partition_modes]
    if any(x not in {"centroid", "fractional"} for x in partitions):
        raise ValueError("partition_modes must contain only centroid|fractional.")

    runs: list[dict[str, Any]] = []
    for preset in (str(primary_preset), str(reference_preset)):
        for refine in levels:
            for mass_mode in masses:
                bench = run_flat_disk_one_leaflet_benchmark(
                    fixture=fixture_path,
                    refine_level=int(refine),
                    outer_mode="disabled",
                    smoothness_model="splay_twist",
                    theta_mode="optimize",
                    optimize_preset=str(preset),
                    parameterization="kh_physical",
                    tilt_mass_mode_in=str(mass_mode),
                    splay_modulus_scale_in=1.0,
                )
                theta_star = float(bench["mesh"]["theta_star"])
                for partition_mode in partitions:
                    audit = run_flat_disk_kh_term_audit(
                        fixture=fixture_path,
                        refine_level=int(refine),
                        outer_mode="disabled",
                        smoothness_model="splay_twist",
                        theta_values=(float(theta_star),),
                        tilt_mass_mode_in=str(mass_mode),
                        radial_projection_diagnostic=True,
                        partition_mode=str(partition_mode),
                    )
                    row = audit["rows"][0]
                    near = float(
                        row["internal_outer_near_ratio_mesh_over_theory_finite"]
                    )
                    far = float(row["internal_outer_far_ratio_mesh_over_theory_finite"])
                    disk = float(row["internal_disk_ratio_mesh_over_theory"])
                    runs.append(
                        {
                            "preset": str(preset),
                            "refine_level": int(refine),
                            "tilt_mass_mode_in": str(mass_mode),
                            "partition_mode": str(partition_mode),
                            "theta_star": float(theta_star),
                            "disk_ratio": disk,
                            "outer_near_ratio": near,
                            "outer_far_ratio": far,
                            "section_score_internal_bands_finite_outer_l2_log": float(
                                row["section_score_internal_bands_finite_outer_l2_log"]
                            ),
                            "operator_effect_proxy": float(
                                abs(
                                    float(
                                        row[
                                            "proj_radial_internal_outer_near_abs_error_delta_vs_unprojected"
                                        ]
                                    )
                                )
                                + abs(
                                    float(
                                        row[
                                            "proj_radial_internal_outer_far_abs_error_delta_vs_unprojected"
                                        ]
                                    )
                                )
                            ),
                        }
                    )

    by = {
        (r["preset"], r["refine_level"], r["tilt_mass_mode_in"], r["partition_mode"]): r
        for r in runs
    }
    partition_effects: list[float] = []
    mass_effects: list[float] = []
    refine_effects: list[float] = []
    for preset in (str(primary_preset), str(reference_preset)):
        for refine in levels:
            for mass_mode in masses:
                centroid_row = by.get((preset, refine, mass_mode, "centroid"))
                fractional_row = by.get((preset, refine, mass_mode, "fractional"))
                if centroid_row is not None and fractional_row is not None:
                    partition_effects.append(
                        float(
                            fractional_row[
                                "section_score_internal_bands_finite_outer_l2_log"
                            ]
                        )
                        - float(
                            centroid_row[
                                "section_score_internal_bands_finite_outer_l2_log"
                            ]
                        )
                    )
            for partition_mode in partitions:
                consistent_row = by.get((preset, refine, "consistent", partition_mode))
                lumped_row = by.get((preset, refine, "lumped", partition_mode))
                if consistent_row is not None and lumped_row is not None:
                    mass_effects.append(
                        float(
                            consistent_row[
                                "section_score_internal_bands_finite_outer_l2_log"
                            ]
                        )
                        - float(
                            lumped_row[
                                "section_score_internal_bands_finite_outer_l2_log"
                            ]
                        )
                    )
        for partition_mode in partitions:
            c2 = by.get((preset, 2, "consistent", partition_mode))
            c3 = by.get((preset, 3, "consistent", partition_mode))
            if c2 is not None and c3 is not None:
                refine_effects.append(
                    float(c3["section_score_internal_bands_finite_outer_l2_log"])
                    - float(c2["section_score_internal_bands_finite_outer_l2_log"])
                )

    effect_sizes = {
        "partition_effect": _mean_abs(partition_effects),
        "mass_effect": _mean_abs(mass_effects),
        "resolution_effect": _mean_abs(refine_effects),
        "operator_effect": _mean_abs([float(r["operator_effect_proxy"]) for r in runs]),
    }
    return {
        "meta": {
            "mode": "kh_error_source_audit",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "primary_preset": str(primary_preset),
            "reference_preset": str(reference_preset),
            "refine_levels": levels,
            "mass_modes": masses,
            "partition_modes": partitions,
            "primary_partition_mode": (
                "fractional" if "fractional" in partitions else partitions[0]
            ),
            "unmeasured_effects": ["solver_effect", "nearcut_effect"],
        },
        "runs": runs,
        "attribution": _rank_effects(effect_sizes),
    }


def run_flat_disk_kh_fractional_refinement_trend(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    optimize_preset: str = "kh_strict_outerfield_best",
    refine_levels: Sequence[int] = (1, 2, 3),
    mass_mode: str = "consistent",
) -> dict[str, Any]:
    """Report strict KH section-score trend across refinement levels."""
    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    levels = [int(x) for x in refine_levels]
    if len(levels) == 0:
        raise ValueError("refine_levels must be non-empty.")
    mass_mode_norm = str(mass_mode).strip().lower()
    if mass_mode_norm not in {"consistent", "lumped"}:
        raise ValueError("mass_mode must be consistent|lumped.")

    rows: list[dict[str, Any]] = []
    for level in levels:
        report = run_flat_disk_kh_error_source_audit(
            fixture=fixture_path,
            primary_preset=str(optimize_preset),
            reference_preset="kh_strict_outerband_tight",
            refine_levels=(int(level),),
            mass_modes=(mass_mode_norm,),
            partition_modes=("centroid", "fractional"),
        )
        run_rows = [
            row
            for row in report["runs"]
            if str(row["preset"]) == str(optimize_preset)
            and int(row["refine_level"]) == int(level)
            and str(row["tilt_mass_mode_in"]) == mass_mode_norm
            and str(row["partition_mode"]) == "fractional"
        ]
        if len(run_rows) == 0:
            raise ValueError(
                f"No fractional run row found for preset={optimize_preset} refine={level}"
            )
        row = run_rows[0]
        score = float(row["section_score_internal_bands_finite_outer_l2_log"])
        if not np.isfinite(score):
            raise ValueError(
                f"Non-finite section score for preset={optimize_preset} refine={level}"
            )
        rows.append(
            {
                "refine_level": int(level),
                "section_score_internal_bands_finite_outer_l2_log": float(score),
                "outer_near_ratio": float(row["outer_near_ratio"]),
                "outer_far_ratio": float(row["outer_far_ratio"]),
                "disk_ratio": float(row["disk_ratio"]),
            }
        )

    scores = [
        float(r["section_score_internal_bands_finite_outer_l2_log"]) for r in rows
    ]
    monotone_non_worsening = all(
        (scores[i + 1] <= (scores[i] + 1e-12)) for i in range(len(scores) - 1)
    )
    return {
        "meta": {
            "mode": "kh_fractional_refinement_trend",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "optimize_preset": str(optimize_preset),
            "mass_mode": mass_mode_norm,
            "primary_partition_mode": "fractional",
            "refine_levels": levels,
        },
        "trend": {
            "rows": rows,
            "monotone_non_worsening": bool(monotone_non_worsening),
        },
    }


def run_flat_disk_kh_error_source_candidate_bakeoff(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    optimize_presets: Sequence[str] = (
        "kh_strict_outerfield_tight",
        "kh_strict_outerband_tight",
        "kh_strict_outerfield_averaged",
    ),
    refine_level: int = 2,
    mass_modes: Sequence[str] = ("consistent",),
    partition_modes: Sequence[str] = ("centroid", "fractional"),
) -> dict[str, Any]:
    """Run bounded strict-KH candidate bakeoff and pick deterministic best row."""
    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    presets = [str(x) for x in optimize_presets]
    if len(presets) == 0:
        raise ValueError("optimize_presets must be non-empty.")

    rows: list[dict[str, Any]] = []
    for preset in presets:
        t0 = perf_counter()
        bench = run_flat_disk_one_leaflet_benchmark(
            fixture=fixture_path,
            refine_level=int(refine_level),
            outer_mode="disabled",
            smoothness_model="splay_twist",
            theta_mode="optimize",
            optimize_preset=str(preset),
            parameterization="kh_physical",
            tilt_mass_mode_in="consistent",
            splay_modulus_scale_in=1.0,
        )
        runtime_seconds = max(float(perf_counter() - t0), 0.0)
        theta_factor = float(bench["parity"]["theta_factor"])
        energy_factor = float(bench["parity"]["energy_factor"])
        parity_score = float(
            np.hypot(
                np.log(max(theta_factor, 1e-18)),
                np.log(max(energy_factor, 1e-18)),
            )
        )
        audit = run_flat_disk_kh_error_source_audit(
            fixture=fixture_path,
            primary_preset=str(preset),
            reference_preset="kh_strict_outerband_tight",
            refine_levels=(int(refine_level),),
            mass_modes=tuple(mass_modes),
            partition_modes=tuple(partition_modes),
        )
        run_rows = list(audit.get("runs", []))
        if len(run_rows) == 0:
            raise ValueError(f"candidate {preset} produced empty runs.")
        near_logs = [
            abs(np.log(max(float(r["outer_near_ratio"]), 1e-18))) for r in run_rows
        ]
        far_logs = [
            abs(np.log(max(float(r["outer_far_ratio"]), 1e-18))) for r in run_rows
        ]
        outer_section_score = float(
            np.hypot(float(np.mean(near_logs)), float(np.mean(far_logs)))
        )
        row = {
            "optimize_preset": str(preset),
            "refine_level": int(refine_level),
            "theta_factor": theta_factor,
            "energy_factor": energy_factor,
            "balanced_parity_score": parity_score,
            "outer_section_score": outer_section_score,
            "runtime_seconds": float(runtime_seconds),
            "dominant_source": str(audit["attribution"]["dominant_source"]),
            "partition_effect": float(
                audit["attribution"]["effect_sizes"].get("partition_effect", 0.0)
            ),
            "mass_effect": float(
                audit["attribution"]["effect_sizes"].get("mass_effect", 0.0)
            ),
            "resolution_effect": float(
                audit["attribution"]["effect_sizes"].get("resolution_effect", 0.0)
            ),
            "operator_effect": float(
                audit["attribution"]["effect_sizes"].get("operator_effect", 0.0)
            ),
        }
        for key in (
            "theta_factor",
            "energy_factor",
            "balanced_parity_score",
            "outer_section_score",
            "runtime_seconds",
        ):
            if not np.isfinite(float(row[key])):
                raise ValueError(f"candidate {preset} produced non-finite {key}.")
        rows.append(row)

    selected = min(
        rows,
        key=lambda row: (
            float(row["outer_section_score"]),
            float(row["balanced_parity_score"]),
            float(row["runtime_seconds"]),
            str(row["optimize_preset"]),
        ),
    )
    return {
        "meta": {
            "mode": "kh_error_source_candidate_bakeoff",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "refine_level": int(refine_level),
            "mass_modes": [str(x) for x in mass_modes],
            "partition_modes": [str(x) for x in partition_modes],
        },
        "candidates": rows,
        "selected_best": selected,
    }


def run_flat_disk_kh_discrete_quality_sweep(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    optimize_preset: str = "kh_strict_outerfield_best",
    refine_level: int = 2,
    outer_local_refine_rmax_lambda_values: Sequence[float] = (8.0, 9.0, 10.0),
    local_edge_flip_steps_values: Sequence[int] = (0, 1),
    outer_local_vertex_average_steps_values: Sequence[int] = (2, 3),
) -> dict[str, Any]:
    """Run bounded mesh-quality sweep and report total+section parity per change."""
    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    rmax_values = [float(x) for x in outer_local_refine_rmax_lambda_values]
    flip_values = [int(x) for x in local_edge_flip_steps_values]
    avg_values = [int(x) for x in outer_local_vertex_average_steps_values]
    if len(rmax_values) == 0 or len(flip_values) == 0 or len(avg_values) == 0:
        raise ValueError("quality sweep value lists must be non-empty.")

    rows: list[dict[str, Any]] = []
    for rmax in rmax_values:
        for flip_steps in flip_values:
            for avg_steps in avg_values:
                bench = run_flat_disk_one_leaflet_benchmark(
                    fixture=fixture_path,
                    refine_level=int(refine_level),
                    outer_mode="disabled",
                    smoothness_model="splay_twist",
                    theta_mode="optimize",
                    optimize_preset=str(optimize_preset),
                    parameterization="kh_physical",
                    tilt_mass_mode_in="consistent",
                    outer_local_refine_steps=1,
                    outer_local_refine_rmin_lambda=1.0,
                    outer_local_refine_rmax_lambda=float(rmax),
                    local_edge_flip_steps=int(flip_steps),
                    local_edge_flip_rmin_lambda=2.0,
                    local_edge_flip_rmax_lambda=6.0,
                    outer_local_vertex_average_steps=int(avg_steps),
                    outer_local_vertex_average_rmin_lambda=4.0,
                    outer_local_vertex_average_rmax_lambda=12.0,
                )
                theta = float(bench["mesh"]["theta_star"])
                audit = run_flat_disk_kh_term_audit(
                    fixture=fixture_path,
                    refine_level=int(bench["meta"]["refine_level"]),
                    outer_mode="disabled",
                    smoothness_model="splay_twist",
                    kappa_physical=10.0,
                    kappa_t_physical=10.0,
                    radius_nm=7.0,
                    length_scale_nm=15.0,
                    drive_physical=(2.0 / 0.7),
                    theta_values=(theta,),
                    tilt_mass_mode_in="consistent",
                    rim_local_refine_steps=int(bench["meta"]["rim_local_refine_steps"]),
                    rim_local_refine_band_lambda=float(
                        bench["meta"]["rim_local_refine_band_lambda"]
                    ),
                    outer_local_refine_steps=int(
                        bench["meta"]["outer_local_refine_steps"]
                    ),
                    outer_local_refine_rmin_lambda=float(
                        bench["meta"]["outer_local_refine_rmin_lambda"]
                    ),
                    outer_local_refine_rmax_lambda=float(
                        bench["meta"]["outer_local_refine_rmax_lambda"]
                    ),
                    local_edge_flip_steps=int(bench["meta"]["local_edge_flip_steps"]),
                    local_edge_flip_rmin_lambda=float(
                        bench["meta"]["local_edge_flip_rmin_lambda"]
                    ),
                    local_edge_flip_rmax_lambda=float(
                        bench["meta"]["local_edge_flip_rmax_lambda"]
                    ),
                    outer_local_vertex_average_steps=int(
                        bench["meta"]["outer_local_vertex_average_steps"]
                    ),
                    outer_local_vertex_average_rmin_lambda=float(
                        bench["meta"]["outer_local_vertex_average_rmin_lambda"]
                    ),
                    outer_local_vertex_average_rmax_lambda=float(
                        bench["meta"]["outer_local_vertex_average_rmax_lambda"]
                    ),
                    partition_mode="fractional",
                )
                row = audit["rows"][0]
                out_row = {
                    "optimize_preset": str(optimize_preset),
                    "requested_refine_level": int(refine_level),
                    "realized_refine_level": int(bench["meta"]["refine_level"]),
                    "outer_local_refine_rmax_lambda": float(rmax),
                    "local_edge_flip_steps": int(flip_steps),
                    "outer_local_vertex_average_steps": int(avg_steps),
                    "theta_star": float(theta),
                    "theta_factor": float(bench["parity"]["theta_factor"]),
                    "energy_factor": float(bench["parity"]["energy_factor"]),
                    "section_score_internal_bands_finite_outer_l2_log": float(
                        row["section_score_internal_bands_finite_outer_l2_log"]
                    ),
                    "disk_ratio": float(row["internal_disk_ratio_mesh_over_theory"]),
                    "outer_near_ratio": float(
                        row["internal_outer_near_ratio_mesh_over_theory_finite"]
                    ),
                    "outer_far_ratio": float(
                        row["internal_outer_far_ratio_mesh_over_theory_finite"]
                    ),
                }
                for key in (
                    "theta_factor",
                    "energy_factor",
                    "section_score_internal_bands_finite_outer_l2_log",
                    "disk_ratio",
                    "outer_near_ratio",
                    "outer_far_ratio",
                ):
                    if not np.isfinite(float(out_row[key])):
                        raise ValueError(f"Non-finite quality metric {key}: {out_row}")
                rows.append(out_row)

    baseline = next(
        (
            row
            for row in rows
            if float(row["outer_local_refine_rmax_lambda"]) == 8.0
            and int(row["local_edge_flip_steps"]) == 0
            and int(row["outer_local_vertex_average_steps"]) == 2
        ),
        rows[0],
    )
    for row in rows:
        row["delta_theta_factor_vs_baseline"] = float(row["theta_factor"]) - float(
            baseline["theta_factor"]
        )
        row["delta_energy_factor_vs_baseline"] = float(row["energy_factor"]) - float(
            baseline["energy_factor"]
        )
        row["delta_section_score_vs_baseline"] = float(
            row["section_score_internal_bands_finite_outer_l2_log"]
        ) - float(baseline["section_score_internal_bands_finite_outer_l2_log"])
        row["delta_disk_ratio_vs_baseline"] = float(row["disk_ratio"]) - float(
            baseline["disk_ratio"]
        )
        row["delta_outer_near_ratio_vs_baseline"] = float(
            row["outer_near_ratio"]
        ) - float(baseline["outer_near_ratio"])
        row["delta_outer_far_ratio_vs_baseline"] = float(
            row["outer_far_ratio"]
        ) - float(baseline["outer_far_ratio"])

    selected = min(
        rows,
        key=lambda row: (
            float(row["section_score_internal_bands_finite_outer_l2_log"]),
            float(
                np.hypot(
                    np.log(max(float(row["theta_factor"]), 1e-18)),
                    np.log(max(float(row["energy_factor"]), 1e-18)),
                )
            ),
            abs(float(row["delta_theta_factor_vs_baseline"])),
            abs(float(row["delta_energy_factor_vs_baseline"])),
        ),
    )
    return {
        "meta": {
            "mode": "kh_discrete_quality_sweep",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "optimize_preset": str(optimize_preset),
            "refine_level": int(refine_level),
            "baseline_outer_local_refine_rmax_lambda": float(
                baseline["outer_local_refine_rmax_lambda"]
            ),
            "baseline_local_edge_flip_steps": int(baseline["local_edge_flip_steps"]),
            "baseline_outer_local_vertex_average_steps": int(
                baseline["outer_local_vertex_average_steps"]
            ),
        },
        "rows": rows,
        "selected_best": selected,
    }


def run_flat_disk_kh_discrete_quality_safe_sweep(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    optimize_preset: str = "kh_strict_outerfield_unpinned",
    refine_level: int = 2,
    outer_local_refine_rmax_lambda_values: Sequence[float] = (7.0, 8.0, 9.0),
    outer_local_vertex_average_steps_values: Sequence[int] = (1, 2, 3),
    outer_far_floor: float = 0.85,
    outer_far_ceiling: float = 1.20,
) -> dict[str, Any]:
    """Run bounded quality sweep with outer-far eligibility filtering."""
    floor = float(outer_far_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("outer_far_floor must be finite and > 0.")
    ceiling = float(outer_far_ceiling)
    if ceiling <= 0.0:
        raise ValueError("outer_far_ceiling must be > 0.")
    if ceiling < floor:
        raise ValueError("outer_far_ceiling must be >= outer_far_floor.")
    report = run_flat_disk_kh_discrete_quality_sweep(
        fixture=fixture,
        optimize_preset=optimize_preset,
        refine_level=refine_level,
        outer_local_refine_rmax_lambda_values=outer_local_refine_rmax_lambda_values,
        local_edge_flip_steps_values=(0,),
        outer_local_vertex_average_steps_values=outer_local_vertex_average_steps_values,
    )
    rows = list(report["rows"])
    for row in rows:
        theta_factor = float(row["theta_factor"])
        energy_factor = float(row["energy_factor"])
        outer_far = float(row["outer_far_ratio"])
        row["balanced_parity_score"] = float(
            np.hypot(
                np.log(max(theta_factor, 1e-18)),
                np.log(max(energy_factor, 1e-18)),
            )
        )
        row["eligible"] = bool(np.isfinite(outer_far) and floor <= outer_far <= ceiling)
    eligible_rows = [row for row in rows if bool(row["eligible"])]
    if len(eligible_rows) == 0:
        raise ValueError(
            f"No eligible safe-sweep rows for outer_far_floor={floor:.6g}; "
            f"rows_tested={len(rows)}."
        )
    selected = min(
        eligible_rows,
        key=lambda row: (
            float(row["section_score_internal_bands_finite_outer_l2_log"]),
            float(row["balanced_parity_score"]),
            abs(float(row["energy_factor"]) - 1.0),
            abs(float(row["theta_factor"]) - 1.0),
            float(row["outer_local_refine_rmax_lambda"]),
            int(row["outer_local_vertex_average_steps"]),
        ),
    )
    return {
        "meta": {
            "mode": "kh_discrete_quality_safe_sweep",
            "fixture": str(report["meta"]["fixture"]),
            "optimize_preset": str(optimize_preset),
            "refine_level": int(refine_level),
            "constraints": {
                "flip_steps": 0,
                "outer_far_floor": floor,
                "outer_far_ceiling": ceiling,
            },
        },
        "rows": rows,
        "selected_best": selected,
    }


def run_flat_disk_kh_discrete_quality_safe_refine_trend(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    optimize_preset: str = "kh_strict_outerfield_unpinned",
    refine_levels: Sequence[int] = (2, 3),
    outer_local_refine_rmax_lambda: float = 8.0,
    outer_local_vertex_average_steps: int = 2,
    outer_far_floor: float = 0.85,
) -> dict[str, Any]:
    """Measure refine-level convergence for a locked safe-quality recipe."""
    levels = [int(x) for x in refine_levels]
    if len(levels) == 0:
        raise ValueError("refine_levels must be non-empty.")
    rows: list[dict[str, Any]] = []
    for level in levels:
        report = run_flat_disk_kh_discrete_quality_sweep(
            fixture=fixture,
            optimize_preset=optimize_preset,
            refine_level=int(level),
            outer_local_refine_rmax_lambda_values=(
                float(outer_local_refine_rmax_lambda),
            ),
            local_edge_flip_steps_values=(0,),
            outer_local_vertex_average_steps_values=(
                int(outer_local_vertex_average_steps),
            ),
        )
        row = dict(report["rows"][0])
        outer_far = float(row["outer_far_ratio"])
        row["eligible"] = bool(
            np.isfinite(outer_far) and outer_far >= float(outer_far_floor)
        )
        realized_refine = int(row["realized_refine_level"])
        if realized_refine != int(level):
            raise ValueError(
                "quality-safe-refine-trend requested refine_level="
                f"{int(level)} but optimize_preset={optimize_preset} realized "
                f"refine_level={realized_refine}. Use an unpinned preset."
            )
        rows.append(
            {
                "refine_level": int(level),
                "requested_refine_level": int(level),
                "realized_refine_level": realized_refine,
                "theta_factor": float(row["theta_factor"]),
                "energy_factor": float(row["energy_factor"]),
                "section_score_internal_bands_finite_outer_l2_log": float(
                    row["section_score_internal_bands_finite_outer_l2_log"]
                ),
                "disk_ratio": float(row["disk_ratio"]),
                "outer_near_ratio": float(row["outer_near_ratio"]),
                "outer_far_ratio": float(row["outer_far_ratio"]),
                "outer_far_eligible": bool(row["eligible"]),
            }
        )

    def _log_abs_ratio(v: float) -> float:
        return float(abs(np.log(max(float(v), 1e-18))))

    deltas: list[dict[str, Any]] = []
    for i in range(len(rows) - 1):
        a = rows[i]
        b = rows[i + 1]
        deltas.append(
            {
                "from_refine": int(a["refine_level"]),
                "to_refine": int(b["refine_level"]),
                "delta_theta_factor": float(b["theta_factor"])
                - float(a["theta_factor"]),
                "delta_energy_factor": float(b["energy_factor"])
                - float(a["energy_factor"]),
                "delta_section_score": float(
                    b["section_score_internal_bands_finite_outer_l2_log"]
                )
                - float(a["section_score_internal_bands_finite_outer_l2_log"]),
                "delta_disk_abs_log_error": _log_abs_ratio(float(b["disk_ratio"]))
                - _log_abs_ratio(float(a["disk_ratio"])),
                "delta_outer_near_abs_log_error": _log_abs_ratio(
                    float(b["outer_near_ratio"])
                )
                - _log_abs_ratio(float(a["outer_near_ratio"])),
                "delta_outer_far_abs_log_error": _log_abs_ratio(
                    float(b["outer_far_ratio"])
                )
                - _log_abs_ratio(float(a["outer_far_ratio"])),
            }
        )
    monotone = {
        "section_score_non_worsening": all(
            float(d["delta_section_score"]) <= 1e-12 for d in deltas
        ),
        "disk_abs_log_error_non_worsening": all(
            float(d["delta_disk_abs_log_error"]) <= 1e-12 for d in deltas
        ),
        "outer_near_abs_log_error_non_worsening": all(
            float(d["delta_outer_near_abs_log_error"]) <= 1e-12 for d in deltas
        ),
        "outer_far_abs_log_error_non_worsening": all(
            float(d["delta_outer_far_abs_log_error"]) <= 1e-12 for d in deltas
        ),
    }
    return {
        "meta": {
            "mode": "kh_discrete_quality_safe_refine_trend",
            "fixture": str(Path(fixture)),
            "optimize_preset": str(optimize_preset),
            "constraints": {
                "flip_steps": 0,
                "outer_far_floor": float(outer_far_floor),
                "outer_local_refine_rmax_lambda": float(outer_local_refine_rmax_lambda),
                "outer_local_vertex_average_steps": int(
                    outer_local_vertex_average_steps
                ),
            },
            "refine_levels": levels,
        },
        "trend": {
            "rows": rows,
            "deltas": deltas,
            "monotone_flags": monotone,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "KH strict error-source attribution audit. "
            "Note: pinned strict presets intentionally fail in "
            "--quality-safe-refine-trend if requested refine differs from realized."
        )
    )
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--primary-preset", type=str, default="kh_strict_outerfield_tight")
    ap.add_argument("--reference-preset", type=str, default="kh_strict_outerband_tight")
    ap.add_argument("--refine-levels", type=int, nargs="+", default=[2, 3])
    ap.add_argument(
        "--mass-modes", type=str, nargs="+", default=["consistent", "lumped"]
    )
    ap.add_argument(
        "--partition-modes",
        type=str,
        nargs="+",
        default=["centroid", "fractional"],
    )
    ap.add_argument("--candidate-bakeoff", action="store_true")
    ap.add_argument("--fractional-refinement-trend", action="store_true")
    ap.add_argument("--quality-candidate-sweep", action="store_true")
    ap.add_argument("--quality-safe-sweep", action="store_true")
    ap.add_argument("--quality-safe-refine-trend", action="store_true")
    ap.add_argument(
        "--optimize-presets",
        type=str,
        nargs="+",
        default=[
            "kh_strict_outerfield_tight",
            "kh_strict_outerband_tight",
            "kh_strict_outerfield_averaged",
        ],
    )
    ap.add_argument("--refine-level", type=int, default=2)
    ap.add_argument(
        "--quality-rmax-values", type=float, nargs="+", default=[8.0, 9.0, 10.0]
    )
    ap.add_argument("--quality-flip-steps-values", type=int, nargs="+", default=[0, 1])
    ap.add_argument(
        "--quality-average-steps-values", type=int, nargs="+", default=[2, 3]
    )
    ap.add_argument(
        "--quality-safe-rmax-values", type=float, nargs="+", default=[7.0, 8.0, 9.0]
    )
    ap.add_argument(
        "--quality-safe-average-steps-values", type=int, nargs="+", default=[1, 2, 3]
    )
    ap.add_argument("--quality-safe-refine-levels", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--quality-safe-rmax", type=float, default=8.0)
    ap.add_argument("--quality-safe-avg-steps", type=int, default=2)
    ap.add_argument("--outer-far-floor", type=float, default=0.85)
    ap.add_argument("--outer-far-ceiling", type=float, default=1.20)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if bool(args.candidate_bakeoff):
        report = run_flat_disk_kh_error_source_candidate_bakeoff(
            fixture=args.fixture,
            optimize_presets=tuple(args.optimize_presets),
            refine_level=int(args.refine_level),
            mass_modes=tuple(args.mass_modes),
            partition_modes=tuple(args.partition_modes),
        )
    elif bool(args.fractional_refinement_trend):
        report = run_flat_disk_kh_fractional_refinement_trend(
            fixture=args.fixture,
            optimize_preset=str(args.primary_preset),
            refine_levels=tuple(args.refine_levels),
            mass_mode=str(args.mass_modes[0]),
        )
    elif bool(args.quality_candidate_sweep):
        report = run_flat_disk_kh_discrete_quality_sweep(
            fixture=args.fixture,
            optimize_preset=str(args.primary_preset),
            refine_level=int(args.refine_level),
            outer_local_refine_rmax_lambda_values=tuple(args.quality_rmax_values),
            local_edge_flip_steps_values=tuple(args.quality_flip_steps_values),
            outer_local_vertex_average_steps_values=tuple(
                args.quality_average_steps_values
            ),
        )
    elif bool(args.quality_safe_sweep):
        preset = str(args.primary_preset)
        if preset == "kh_strict_outerfield_tight":
            preset = "kh_strict_outerfield_unpinned"
        report = run_flat_disk_kh_discrete_quality_safe_sweep(
            fixture=args.fixture,
            optimize_preset=preset,
            refine_level=int(args.refine_level),
            outer_local_refine_rmax_lambda_values=tuple(args.quality_safe_rmax_values),
            outer_local_vertex_average_steps_values=tuple(
                args.quality_safe_average_steps_values
            ),
            outer_far_floor=float(args.outer_far_floor),
            outer_far_ceiling=float(args.outer_far_ceiling),
        )
    elif bool(args.quality_safe_refine_trend):
        preset = str(args.primary_preset)
        if preset == "kh_strict_outerfield_tight":
            preset = "kh_strict_outerfield_unpinned"
        report = run_flat_disk_kh_discrete_quality_safe_refine_trend(
            fixture=args.fixture,
            optimize_preset=preset,
            refine_levels=tuple(args.quality_safe_refine_levels),
            outer_local_refine_rmax_lambda=float(args.quality_safe_rmax),
            outer_local_vertex_average_steps=int(args.quality_safe_avg_steps),
            outer_far_floor=float(args.outer_far_floor),
        )
    else:
        report = run_flat_disk_kh_error_source_audit(
            fixture=args.fixture,
            primary_preset=args.primary_preset,
            reference_preset=args.reference_preset,
            refine_levels=tuple(args.refine_levels),
            mass_modes=tuple(args.mass_modes),
            partition_modes=tuple(args.partition_modes),
        )
    out = Path(args.output)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(report, fh, sort_keys=False)
    print(f"Wrote KH error-source audit: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
