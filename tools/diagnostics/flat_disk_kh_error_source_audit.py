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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="KH strict error-source attribution audit."
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
