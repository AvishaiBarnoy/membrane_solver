#!/usr/bin/env python3
"""Strict-KH partition ablation diagnostic at optimized theta_B."""

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
    ROOT
    / "benchmarks"
    / "outputs"
    / "diagnostics"
    / "flat_disk_kh_partition_ablation.yaml"
)


def _ensure_repo_root_on_sys_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _partition_score(disk_ratio: float, outer_ratio: float) -> float:
    """Balanced region mismatch score around exact ratio 1."""
    return float(
        np.hypot(
            np.log(max(float(disk_ratio), 1e-18)),
            np.log(max(float(outer_ratio), 1e-18)),
        )
    )


def run_flat_disk_kh_partition_ablation(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    optimize_presets: Sequence[str] = (
        "kh_strict_energy_tight",
        "kh_strict_partition_tight",
    ),
    baseline_optimize_preset: str = "kh_strict_energy_tight",
    refine_level: int = 1,
) -> dict[str, Any]:
    """Compare strict presets by partition diagnostics at optimized theta_B."""
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

    rows: list[dict[str, float | int | str]] = []
    for rank, preset in enumerate(presets):
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
                "Non-finite partition ratios for preset "
                f"{preset}: disk={disk_ratio} outer={outer_ratio}"
            )
        rows.append(
            {
                "optimize_preset": str(preset),
                "theta_star": theta_star,
                "theta_factor": float(bench["parity"]["theta_factor"]),
                "energy_factor": float(bench["parity"]["energy_factor"]),
                "runtime_seconds": runtime_seconds,
                "rim_local_refine_steps": int(bench["meta"]["rim_local_refine_steps"]),
                "rim_local_refine_band_lambda": float(
                    bench["meta"]["rim_local_refine_band_lambda"]
                ),
                "internal_disk_ratio_mesh_over_theory": disk_ratio,
                "internal_outer_ratio_mesh_over_theory": outer_ratio,
                "partition_score": _partition_score(disk_ratio, outer_ratio),
                "mesh_internal_disk_core": float(row["mesh_internal_disk_core"]),
                "mesh_internal_rim_band": float(row["mesh_internal_rim_band"]),
                "mesh_internal_outer_near": float(row["mesh_internal_outer_near"]),
                "mesh_internal_outer_far": float(row["mesh_internal_outer_far"]),
                "rim_band_h_over_lambda_median": float(
                    row["rim_band_h_over_lambda_median"]
                ),
                "complexity_rank": int(rank),
            }
        )

    selected = min(
        rows,
        key=lambda x: (
            float(x["partition_score"]),
            float(x["energy_factor"]),
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
                float(x["partition_score"]),
                float(x["energy_factor"]),
                float(x["runtime_seconds"]),
                int(x["complexity_rank"]),
            ),
        )
        if baseline_rows
        else None
    )
    delta_partition = None
    delta_energy_factor = None
    if baseline_best is not None:
        delta_partition = float(selected["partition_score"]) - float(
            baseline_best["partition_score"]
        )
        delta_energy_factor = float(selected["energy_factor"]) - float(
            baseline_best["energy_factor"]
        )

    return {
        "meta": {
            "mode": "flat_disk_kh_partition_ablation",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "parameterization": "kh_physical",
            "optimize_presets": presets,
            "baseline_optimize_preset": str(baseline_optimize_preset),
        },
        "rows": rows,
        "selected_best": selected,
        "baseline_best": baseline_best,
        "selected_vs_baseline_partition_score_delta": (
            None if delta_partition is None else float(delta_partition)
        ),
        "selected_vs_baseline_energy_factor_delta": (
            None if delta_energy_factor is None else float(delta_energy_factor)
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--optimize-presets", nargs="+", default=None)
    ap.add_argument("--baseline-optimize-preset", default="kh_strict_energy_tight")
    ap.add_argument("--refine-level", type=int, default=1)
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    presets = (
        tuple(str(x) for x in args.optimize_presets)
        if args.optimize_presets is not None
        else ("kh_strict_energy_tight", "kh_strict_partition_tight")
    )
    report = run_flat_disk_kh_partition_ablation(
        fixture=args.fixture,
        optimize_presets=presets,
        baseline_optimize_preset=args.baseline_optimize_preset,
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
