#!/usr/bin/env python3
"""Strict-KH rim/interface fidelity diagnostics for flat-disk benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
DEFAULT_OUT = (
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "flat_disk_kh_rim_fidelity.yaml"
)


def _ensure_repo_root_on_sys_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def run_flat_disk_kh_rim_fidelity(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    optimize_preset: str = "kh_strict_fast",
    refine_level: int = 1,
    rim_local_refine_steps: int = 1,
    rim_local_refine_band_lambda: float = 4.0,
) -> dict[str, Any]:
    """Return strict-KH rim continuity/fidelity metrics from one benchmark run."""
    _ensure_repo_root_on_sys_path()
    from tools.reproduce_flat_disk_one_leaflet import (
        run_flat_disk_one_leaflet_benchmark,
    )

    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    report = run_flat_disk_one_leaflet_benchmark(
        fixture=fixture_path,
        refine_level=int(refine_level),
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        parameterization="kh_physical",
        optimize_preset=str(optimize_preset),
        tilt_mass_mode_in="consistent",
        rim_local_refine_steps=int(rim_local_refine_steps),
        rim_local_refine_band_lambda=float(rim_local_refine_band_lambda),
    )

    mesh = report["mesh"]
    parity = report["parity"]
    continuity = mesh["rim_continuity"]
    boundary = mesh["rim_boundary_realization"]
    leakage = mesh["leakage"]

    rim_abs_median = float(mesh["profile"]["rim_abs_median"])
    jump_abs_median = float(continuity["jump_abs_median"])
    jump_ratio = jump_abs_median / max(rim_abs_median, 1e-18)

    return {
        "meta": {
            "mode": "flat_disk_kh_rim_fidelity",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "parameterization": "kh_physical",
            "optimize_preset": str(optimize_preset),
            "refine_level": int(report["meta"]["refine_level"]),
            "rim_local_refine_steps": int(report["meta"]["rim_local_refine_steps"]),
            "rim_local_refine_band_lambda": float(
                report["meta"]["rim_local_refine_band_lambda"]
            ),
        },
        "parity": {
            "theta_factor": float(parity["theta_factor"]),
            "energy_factor": float(parity["energy_factor"]),
            "meets_factor_2": bool(parity["meets_factor_2"]),
        },
        "rim_fidelity": {
            "jump_abs_median": jump_abs_median,
            "jump_abs_max": float(continuity["jump_abs_max"]),
            "jump_ratio": float(jump_ratio),
            "rim_theta_error_abs_median": float(boundary["rim_theta_error_abs_median"]),
            "rim_theta_error_abs_max": float(boundary["rim_theta_error_abs_max"]),
            "inner_tphi_over_trad_median": float(
                leakage["inner_tphi_over_trad_median"]
            ),
            "outer_tphi_over_trad_median": float(
                leakage["outer_tphi_over_trad_median"]
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--optimize-preset", default="kh_strict_fast")
    ap.add_argument("--refine-level", type=int, default=1)
    ap.add_argument("--rim-local-refine-steps", type=int, default=1)
    ap.add_argument("--rim-local-refine-band-lambda", type=float, default=4.0)
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    report = run_flat_disk_kh_rim_fidelity(
        fixture=args.fixture,
        optimize_preset=args.optimize_preset,
        refine_level=args.refine_level,
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
