#!/usr/bin/env python3
"""Canonical lane-separated parity scoreboard for flat-disk theory reproduction."""

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
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "flat_disk_parity_scoreboard.yaml"
)


def _ensure_repo_root_on_sys_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _balanced_score(theta_factor: float, energy_factor: float) -> float:
    """Return balanced parity score used for lane-level comparability."""
    return float(
        np.hypot(
            np.log(max(float(theta_factor), 1e-18)),
            np.log(max(float(energy_factor), 1e-18)),
        )
    )


def run_flat_disk_parity_scoreboard(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    refine_level: int = 1,
    outer_mode: str = "disabled",
    legacy_smoothness_model: str = "dirichlet",
    legacy_theta_mode: str = "scan",
    kh_smoothness_model: str = "splay_twist",
    kh_theta_mode: str = "optimize",
    kh_optimize_preset: str = "kh_strict_fast",
    tilt_mass_mode_in: str = "consistent",
) -> dict[str, Any]:
    """Run canonical legacy/KH lanes and emit a locked-reference scoreboard."""
    _ensure_repo_root_on_sys_path()
    from tools.reproduce_flat_disk_one_leaflet import (
        run_flat_disk_one_leaflet_benchmark,
    )

    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    legacy = run_flat_disk_one_leaflet_benchmark(
        fixture=fixture_path,
        refine_level=int(refine_level),
        outer_mode=str(outer_mode),
        smoothness_model=str(legacy_smoothness_model),
        theta_mode=str(legacy_theta_mode),
        parameterization="legacy",
    )

    kh = run_flat_disk_one_leaflet_benchmark(
        fixture=fixture_path,
        refine_level=int(refine_level),
        outer_mode=str(outer_mode),
        smoothness_model=str(kh_smoothness_model),
        theta_mode=str(kh_theta_mode),
        parameterization="kh_physical",
        optimize_preset=str(kh_optimize_preset),
        tilt_mass_mode_in=str(tilt_mass_mode_in),
    )

    legacy_theta = float(legacy["parity"]["theta_factor"])
    legacy_energy = float(legacy["parity"]["energy_factor"])
    kh_theta = float(kh["parity"]["theta_factor"])
    kh_energy = float(kh["parity"]["energy_factor"])

    return {
        "meta": {
            "mode": "flat_disk_parity_scoreboard",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "reference_lock": {
                "legacy": {
                    "theory_model": str(legacy["meta"]["theory_model"]),
                    "theory_source": str(legacy["meta"]["theory_source"]),
                },
                "kh_physical": {
                    "theory_model": str(kh["meta"]["theory_model"]),
                    "theory_source": str(kh["meta"]["theory_source"]),
                },
            },
        },
        "lanes": {
            "legacy": {
                "theta_factor": legacy_theta,
                "energy_factor": legacy_energy,
                "balanced_parity_score": _balanced_score(legacy_theta, legacy_energy),
                "theta_star": float(legacy["mesh"]["theta_star"]),
                "total_energy": float(legacy["mesh"]["total_energy"]),
                "meets_factor_2": bool(legacy["parity"]["meets_factor_2"]),
                "theta_mode": str(legacy["meta"]["theta_mode"]),
                "optimize_preset_effective": str(
                    legacy["meta"]["optimize_preset_effective"]
                ),
            },
            "kh_physical": {
                "theta_factor": kh_theta,
                "energy_factor": kh_energy,
                "balanced_parity_score": _balanced_score(kh_theta, kh_energy),
                "theta_star": float(kh["mesh"]["theta_star"]),
                "total_energy": float(kh["mesh"]["total_energy"]),
                "meets_factor_2": bool(kh["parity"]["meets_factor_2"]),
                "theta_mode": str(kh["meta"]["theta_mode"]),
                "optimize_preset_effective": str(
                    kh["meta"]["optimize_preset_effective"]
                ),
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--refine-level", type=int, default=1)
    ap.add_argument("--outer-mode", choices=("disabled", "free"), default="disabled")
    ap.add_argument(
        "--legacy-smoothness-model",
        choices=("dirichlet", "splay_twist"),
        default="dirichlet",
    )
    ap.add_argument(
        "--legacy-theta-mode",
        choices=("scan", "optimize", "optimize_full"),
        default="scan",
    )
    ap.add_argument(
        "--kh-smoothness-model",
        choices=("dirichlet", "splay_twist"),
        default="splay_twist",
    )
    ap.add_argument(
        "--kh-theta-mode",
        choices=("scan", "optimize", "optimize_full"),
        default="optimize",
    )
    ap.add_argument("--kh-optimize-preset", default="kh_strict_fast")
    ap.add_argument(
        "--tilt-mass-mode-in",
        choices=("auto", "lumped", "consistent"),
        default="consistent",
    )
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    report = run_flat_disk_parity_scoreboard(
        fixture=args.fixture,
        refine_level=args.refine_level,
        outer_mode=args.outer_mode,
        legacy_smoothness_model=args.legacy_smoothness_model,
        legacy_theta_mode=args.legacy_theta_mode,
        kh_smoothness_model=args.kh_smoothness_model,
        kh_theta_mode=args.kh_theta_mode,
        kh_optimize_preset=args.kh_optimize_preset,
        tilt_mass_mode_in=args.tilt_mass_mode_in,
    )

    out = Path(args.output)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
