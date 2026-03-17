"""Compact curved 3D boundary-condition sweep over the reduced audit surface."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np

from tools.diagnostics.flat_disk_curved_3d_audit import (
    DEFAULT_FIXTURE,
    run_flat_disk_curved_3d_audit,
)


def _penalty(value: float) -> float:
    """Return a finite deviation penalty for a reported factor."""

    scalar = float(value)
    if not np.isfinite(scalar):
        return 1.0e6
    return abs(scalar - 1.0)


def _score_row(row: dict[str, Any]) -> float:
    """Compute a compact ranking score from parity and boundary factors."""

    return float(
        _penalty(row["theta_factor"])
        + _penalty(row["energy_factor"])
        + (0.5 * _penalty(row["kink_angle_factor"]))
        + _penalty(row["tilt_in_factor"])
        + _penalty(row["tilt_out_factor"])
    )


def run_flat_disk_curved_3d_bc_sweep(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    sweep: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a bounded curved-3d sweep and return ranked candidates."""

    sweep_cfg = sweep or {}
    refine_levels = [int(v) for v in sweep_cfg.get("refine_levels", [2])]
    z_gauges = [str(v) for v in sweep_cfg.get("z_gauges", ["mean_zero"])]
    profiles = [str(v) for v in sweep_cfg.get("curved_acceptance_profiles", ["fast"])]
    theta_initials = [float(v) for v in sweep_cfg.get("theta_initials", [0.12])]
    theta_steps = [int(v) for v in sweep_cfg.get("theta_optimize_steps", [6])]
    theta_deltas = [float(v) for v in sweep_cfg.get("theta_optimize_deltas", [0.01])]
    theta_inner_steps = [
        int(v) for v in sweep_cfg.get("theta_optimize_inner_steps", [12])
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
            boundary = audit["boundary_at_R"] or {}
            row = {
                "status": "ok",
                "config": config,
                "theta_factor": float(audit["parity"]["theta_factor"]),
                "energy_factor": float(audit["parity"]["energy_factor"]),
                "kink_angle_factor": float(
                    boundary.get("kink_angle_factor", float("inf"))
                ),
                "tilt_in_factor": float(boundary.get("tilt_in_factor", float("inf"))),
                "tilt_out_factor": float(boundary.get("tilt_out_factor", float("inf"))),
                "boundary_available": bool(boundary.get("available", False)),
            }
            penalties = {
                "kink_angle": _penalty(row["kink_angle_factor"]),
                "tilt_in": _penalty(row["tilt_in_factor"]),
                "tilt_out": _penalty(row["tilt_out_factor"]),
            }
            row["dominant_metric"] = str(max(penalties, key=penalties.get))
            row["dominant_penalty"] = float(penalties[row["dominant_metric"]])
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
            "mode": "curved_3d_bc_sweep_smoke",
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
