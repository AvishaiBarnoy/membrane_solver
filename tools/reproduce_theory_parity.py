#!/usr/bin/env python3
"""Reproduce theory-parity metrics and write a YAML report."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.context import CommandContext
from commands.executor import execute_command_line
from geometry.geom_io import load_data, parse_geometry
from geometry.tilt_operators import p1_vertex_divergence
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MESH = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
DEFAULT_OUT = (
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "theory_parity_report.yaml"
)
DEFAULT_PROTOCOL = ("g10", "r", "V2", "t5e-3", "g8", "t2e-3", "g12")


def _build_context(mesh_path: Path) -> CommandContext:
    mesh = parse_geometry(load_data(str(mesh_path)))
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    return CommandContext(mesh, minim, minim.stepper)


def _tilt_stats_quantiles(mesh) -> dict[str, float]:
    mesh.build_position_cache()
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return {
            "tstat_in_p90_norm": 0.0,
            "tstat_out_p90_norm": 0.0,
        }

    tin = np.asarray(mesh.tilts_in_view(), dtype=float)
    tout = np.asarray(mesh.tilts_out_view(), dtype=float)

    # Keep parity metrics compact and robust.
    mags_in = np.linalg.norm(tin, axis=1)
    mags_out = np.linalg.norm(tout, axis=1)
    _div_in, _ = p1_vertex_divergence(
        n_vertices=len(mesh.vertex_ids),
        positions=positions,
        tilts=tin,
        tri_rows=tri_rows,
    )
    _div_out, _ = p1_vertex_divergence(
        n_vertices=len(mesh.vertex_ids),
        positions=positions,
        tilts=tout,
        tri_rows=tri_rows,
    )

    return {
        "tstat_in_p90_norm": float(np.quantile(mags_in, 0.90)),
        "tstat_out_p90_norm": float(np.quantile(mags_out, 0.90)),
    }


def _collect_report(mesh_path: Path, protocol: tuple[str, ...]) -> dict[str, Any]:
    ctx = _build_context(mesh_path)
    minim = ctx.minimizer

    for cmd in protocol:
        execute_command_line(ctx, cmd)

    breakdown = minim.compute_energy_breakdown()
    tilt_stats = _tilt_stats_quantiles(ctx.mesh)

    report = {
        "meta": {
            "fixture": str(mesh_path.relative_to(ROOT)),
            "protocol": list(protocol),
            "format": "yaml",
        },
        "metrics": {
            "final_energy": float(minim.compute_energy()),
            "thetaB_value": float(
                ctx.mesh.global_parameters.get("tilt_thetaB_value") or 0.0
            ),
            "breakdown": {
                "bending_tilt_in": float(breakdown.get("bending_tilt_in") or 0.0),
                "bending_tilt_out": float(breakdown.get("bending_tilt_out") or 0.0),
                "tilt_in": float(breakdown.get("tilt_in") or 0.0),
                "tilt_out": float(breakdown.get("tilt_out") or 0.0),
                "tilt_thetaB_contact_in": float(
                    breakdown.get("tilt_thetaB_contact_in") or 0.0
                ),
            },
            "tilt_stats": tilt_stats,
        },
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--protocol",
        nargs="+",
        default=list(DEFAULT_PROTOCOL),
        help="Command sequence to reproduce parity metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mesh_path = Path(args.mesh)
    out_path = Path(args.out)
    protocol = tuple(str(x) for x in args.protocol)

    report = _collect_report(mesh_path=mesh_path, protocol=protocol)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
