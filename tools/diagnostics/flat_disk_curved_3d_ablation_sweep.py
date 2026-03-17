"""Compact curved 3D ablation sweep over the reduced audit smoke surface."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from tools.diagnostics.flat_disk_curved_3d_audit import (
    DEFAULT_FIXTURE,
    run_flat_disk_curved_3d_audit,
)


def _score_row(row: dict[str, Any]) -> float:
    """Compute a simple ranking score from observed parity factors."""

    return float(
        abs(float(row["theta_factor_observed"]) - 1.0)
        + abs(float(row["energy_factor_observed"]) - 1.0)
    )


def run_flat_disk_curved_3d_ablation_sweep(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    sweep: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a bounded curved-audit sweep and return ranked candidates."""

    sweep_cfg = sweep or {}
    refine_levels = [int(v) for v in sweep_cfg.get("refine_levels", [2])]
    z_gauges = [str(v) for v in sweep_cfg.get("z_gauges", ["mean_zero"])]
    profiles = [str(v) for v in sweep_cfg.get("curved_acceptance_profiles", ["fast"])]
    theta_initials = [float(v) for v in sweep_cfg.get("theta_initials", [0.12])]
    theta_steps = [int(v) for v in sweep_cfg.get("theta_optimize_steps", [8])]
    theta_deltas = [float(v) for v in sweep_cfg.get("theta_optimize_deltas", [0.01])]
    theta_inner_steps = [
        int(v) for v in sweep_cfg.get("theta_optimize_inner_steps", [20])
    ]
    outer_mode = str(sweep_cfg.get("outer_mode", "free"))
    smoothness_model = str(sweep_cfg.get("smoothness_model", "splay_twist"))

    rows: list[dict[str, Any]] = []
    for (
        refine_level,
        z_gauge,
        profile,
        theta_initial,
        optimize_steps,
        optimize_delta,
        optimize_inner,
    ) in itertools.product(
        refine_levels,
        z_gauges,
        profiles,
        theta_initials,
        theta_steps,
        theta_deltas,
        theta_inner_steps,
    ):
        config = {
            "refine_level": int(refine_level),
            "z_gauge": str(z_gauge),
            "curved_acceptance_profile": str(profile),
            "theta_initial": float(theta_initial),
            "theta_optimize_steps": int(optimize_steps),
            "theta_optimize_delta": float(optimize_delta),
            "theta_optimize_inner_steps": int(optimize_inner),
        }
        try:
            audit = run_flat_disk_curved_3d_audit(
                fixture=fixture,
                refine_level=int(refine_level),
                outer_mode=outer_mode,
                smoothness_model=smoothness_model,
                theta_mode="optimize",
                theta_initial=float(theta_initial),
                theta_optimize_steps=int(optimize_steps),
                theta_optimize_every=1,
                theta_optimize_delta=float(optimize_delta),
                theta_optimize_inner_steps=int(optimize_inner),
                z_gauge=str(z_gauge),
                curved_acceptance_profile=str(profile),
                include_sections=False,
            )
            row = {
                "status": "ok",
                "config": config,
                "theta_factor_observed": float(audit["parity"]["theta_factor"]),
                "energy_factor_observed": float(audit["parity"]["energy_factor"]),
                "theta_star_mesh": float(audit["parity"]["theta_star_mesh"]),
                "total_energy_mesh": float(audit["parity"]["total_energy_mesh"]),
                "boundary_available": bool(
                    (audit.get("boundary_at_R") or {}).get("available", False)
                ),
            }
            row["score"] = _score_row(row)
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "status": "failed",
                    "config": config,
                    "error": str(exc),
                }
            )

    ranked = sorted(
        (row for row in rows if row.get("status") == "ok"),
        key=lambda row: float(row["score"]),
    )
    return {
        "meta": {
            "mode": "curved_3d_ablation_sweep_smoke",
            "fixture": str(Path(fixture)),
            "candidate_count": int(len(rows)),
            "ok_count": int(len(ranked)),
            "failed_count": int(len(rows) - len(ranked)),
        },
        "sweep_config": {
            "refine_levels": [int(v) for v in refine_levels],
            "z_gauges": [str(v) for v in z_gauges],
            "curved_acceptance_profiles": [str(v) for v in profiles],
            "theta_initials": [float(v) for v in theta_initials],
            "theta_optimize_steps": [int(v) for v in theta_steps],
            "theta_optimize_deltas": [float(v) for v in theta_deltas],
            "theta_optimize_inner_steps": [int(v) for v in theta_inner_steps],
        },
        "best_candidate": None if not ranked else ranked[0],
        "ranked_candidates": ranked,
        "all_candidates": rows,
    }
