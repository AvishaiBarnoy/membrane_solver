#!/usr/bin/env python3
"""Region-resolved strict-KH internal parity diagnostics at optimized theta_B."""

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
DEFAULT_OUT = (
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "flat_disk_kh_region_parity.yaml"
)


def _ensure_repo_root_on_sys_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _score_from_region_ratios(disk_ratio: float, outer_ratio: float) -> float:
    """Return balanced region mismatch score around exact value 1."""
    return float(
        np.hypot(
            np.log(max(float(disk_ratio), 1e-18)),
            np.log(max(float(outer_ratio), 1e-18)),
        )
    )


def run_flat_disk_kh_region_parity(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    optimize_presets: Sequence[str] = (
        "kh_strict_energy_tight",
        "kh_strict_partition_tight",
    ),
    refine_level: int = 1,
    rim_local_refine_steps: Sequence[int] | None = None,
    rim_local_refine_band_lambdas: Sequence[float] | None = None,
    baseline_optimize_preset: str = "kh_strict_energy_tight",
) -> dict[str, Any]:
    """Compare strict candidates by disk/outer internal energy ratios at theta*."""
    _ensure_repo_root_on_sys_path()
    from tools.diagnostics.flat_disk_kh_term_audit import run_flat_disk_kh_term_audit
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

    steps_override = None
    if rim_local_refine_steps is not None:
        steps_override = [int(x) for x in rim_local_refine_steps]
        if len(steps_override) == 0:
            raise ValueError("rim_local_refine_steps must be non-empty when provided.")
    bands_override = None
    if rim_local_refine_band_lambdas is not None:
        bands_override = [float(x) for x in rim_local_refine_band_lambdas]
        if len(bands_override) == 0:
            raise ValueError(
                "rim_local_refine_band_lambdas must be non-empty when provided."
            )

    rows: list[dict[str, float | int | str]] = []
    complexity_rank = 0
    for preset in presets:
        if steps_override is None or bands_override is None:
            # Probe preset-default strict mesh controls.
            probe = run_flat_disk_one_leaflet_benchmark(
                fixture=fixture_path,
                refine_level=int(refine_level),
                outer_mode="disabled",
                smoothness_model="splay_twist",
                theta_mode="optimize",
                parameterization="kh_physical",
                optimize_preset=str(preset),
                tilt_mass_mode_in="consistent",
            )
            step_values = [int(probe["meta"]["rim_local_refine_steps"])]
            band_values = [float(probe["meta"]["rim_local_refine_band_lambda"])]
        else:
            step_values = steps_override
            band_values = bands_override

        for rim_steps in step_values:
            for rim_band in band_values:
                t0 = perf_counter()
                bench = run_flat_disk_one_leaflet_benchmark(
                    fixture=fixture_path,
                    refine_level=int(refine_level),
                    outer_mode="disabled",
                    smoothness_model="splay_twist",
                    theta_mode="optimize",
                    parameterization="kh_physical",
                    optimize_preset=str(preset),
                    tilt_mass_mode_in="consistent",
                    rim_local_refine_steps=int(rim_steps),
                    rim_local_refine_band_lambda=float(rim_band),
                )
                runtime_seconds = float(perf_counter() - t0)
                theta_star = float(bench["mesh"]["theta_star"])
                audit = run_flat_disk_kh_term_audit(
                    fixture=fixture_path,
                    refine_level=int(bench["meta"]["refine_level"]),
                    outer_mode="disabled",
                    smoothness_model="splay_twist",
                    theta_values=(theta_star,),
                    tilt_mass_mode_in="consistent",
                    rim_local_refine_steps=int(bench["meta"]["rim_local_refine_steps"]),
                    rim_local_refine_band_lambda=float(
                        bench["meta"]["rim_local_refine_band_lambda"]
                    ),
                )
                row = audit["rows"][0]
                disk_ratio = float(row["internal_disk_ratio_mesh_over_theory"])
                outer_ratio = float(row["internal_outer_ratio_mesh_over_theory"])
                if not (np.isfinite(disk_ratio) and np.isfinite(outer_ratio)):
                    raise ValueError(
                        "Non-finite internal region ratios in strict region parity "
                        f"candidate preset={preset} rim_steps={rim_steps} "
                        f"rim_band_lambda={rim_band}."
                    )
                rows.append(
                    {
                        "optimize_preset": str(preset),
                        "theta_star": theta_star,
                        "theta_factor": float(bench["parity"]["theta_factor"]),
                        "energy_factor": float(bench["parity"]["energy_factor"]),
                        "runtime_seconds": runtime_seconds,
                        "rim_local_refine_steps": int(
                            bench["meta"]["rim_local_refine_steps"]
                        ),
                        "rim_local_refine_band_lambda": float(
                            bench["meta"]["rim_local_refine_band_lambda"]
                        ),
                        "internal_disk_ratio_mesh_over_theory": disk_ratio,
                        "internal_outer_ratio_mesh_over_theory": outer_ratio,
                        "region_parity_score": _score_from_region_ratios(
                            disk_ratio, outer_ratio
                        ),
                        "complexity_rank": int(complexity_rank),
                    }
                )
                complexity_rank += 1

    if len(rows) == 0:
        raise ValueError("No strict region parity candidates were generated.")
    selected = min(
        rows,
        key=lambda x: (
            float(x["region_parity_score"]),
            float(x["runtime_seconds"]),
            int(x["complexity_rank"]),
        ),
    )
    baseline_rows = [
        row
        for row in rows
        if str(row["optimize_preset"]) == str(baseline_optimize_preset)
    ]
    baseline_best = (
        min(
            baseline_rows,
            key=lambda x: (
                float(x["region_parity_score"]),
                float(x["runtime_seconds"]),
                int(x["complexity_rank"]),
            ),
        )
        if baseline_rows
        else None
    )
    selected_vs_baseline_partition_delta = None
    if baseline_best is not None:
        selected_vs_baseline_partition_delta = float(
            float(selected["region_parity_score"])
            - float(baseline_best["region_parity_score"])
        )
    return {
        "meta": {
            "mode": "flat_disk_kh_region_parity",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "parameterization": "kh_physical",
            "optimize_presets": presets,
            "baseline_optimize_preset": str(baseline_optimize_preset),
            "rim_local_refine_steps": (
                None if steps_override is None else [int(x) for x in steps_override]
            ),
            "rim_local_refine_band_lambdas": (
                None if bands_override is None else [float(x) for x in bands_override]
            ),
        },
        "rows": rows,
        "selected_best": selected,
        "baseline_best": baseline_best,
        "selected_vs_baseline_partition_score_delta": (
            None
            if selected_vs_baseline_partition_delta is None
            else float(selected_vs_baseline_partition_delta)
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--optimize-presets", nargs="+", default=None)
    ap.add_argument(
        "--baseline-optimize-preset",
        default="kh_strict_energy_tight",
    )
    ap.add_argument("--refine-level", type=int, default=1)
    ap.add_argument("--rim-local-refine-steps", type=int, nargs="+", default=None)
    ap.add_argument(
        "--rim-local-refine-band-lambdas", type=float, nargs="+", default=None
    )
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    presets = (
        tuple(str(x) for x in args.optimize_presets)
        if args.optimize_presets is not None
        else ("kh_strict_energy_tight", "kh_strict_partition_tight")
    )
    report = run_flat_disk_kh_region_parity(
        fixture=args.fixture,
        optimize_presets=presets,
        refine_level=args.refine_level,
        rim_local_refine_steps=args.rim_local_refine_steps,
        rim_local_refine_band_lambdas=args.rim_local_refine_band_lambdas,
        baseline_optimize_preset=args.baseline_optimize_preset,
    )
    out = Path(args.output)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
