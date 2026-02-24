#!/usr/bin/env python3
"""KH strict error-source audit with deterministic effect ranking."""

from __future__ import annotations

import argparse
from pathlib import Path
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
                audit = run_flat_disk_kh_term_audit(
                    fixture=fixture_path,
                    refine_level=int(refine),
                    outer_mode="disabled",
                    smoothness_model="splay_twist",
                    theta_values=(float(theta_star),),
                    tilt_mass_mode_in=str(mass_mode),
                    radial_projection_diagnostic=True,
                )
                row = audit["rows"][0]
                near = float(row["internal_outer_near_ratio_mesh_over_theory_finite"])
                far = float(row["internal_outer_far_ratio_mesh_over_theory_finite"])
                disk = float(row["internal_disk_ratio_mesh_over_theory"])
                runs.append(
                    {
                        "preset": str(preset),
                        "refine_level": int(refine),
                        "tilt_mass_mode_in": str(mass_mode),
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

    by = {(r["preset"], r["refine_level"], r["tilt_mass_mode_in"]): r for r in runs}
    mass_effects: list[float] = []
    refine_effects: list[float] = []
    for preset in (str(primary_preset), str(reference_preset)):
        for refine in levels:
            consistent_row = by.get((preset, refine, "consistent"))
            lumped_row = by.get((preset, refine, "lumped"))
            if consistent_row is not None and lumped_row is not None:
                mass_effects.append(
                    float(
                        consistent_row[
                            "section_score_internal_bands_finite_outer_l2_log"
                        ]
                    )
                    - float(
                        lumped_row["section_score_internal_bands_finite_outer_l2_log"]
                    )
                )
        c2 = by.get((preset, 2, "consistent"))
        c3 = by.get((preset, 3, "consistent"))
        if c2 is not None and c3 is not None:
            refine_effects.append(
                float(c3["section_score_internal_bands_finite_outer_l2_log"])
                - float(c2["section_score_internal_bands_finite_outer_l2_log"])
            )

    effect_sizes = {
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
            "unmeasured_effects": [
                "partition_effect",
                "solver_effect",
                "nearcut_effect",
            ],
        },
        "runs": runs,
        "attribution": _rank_effects(effect_sizes),
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
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    report = run_flat_disk_kh_error_source_audit(
        fixture=args.fixture,
        primary_preset=args.primary_preset,
        reference_preset=args.reference_preset,
        refine_levels=tuple(args.refine_levels),
        mass_modes=tuple(args.mass_modes),
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
