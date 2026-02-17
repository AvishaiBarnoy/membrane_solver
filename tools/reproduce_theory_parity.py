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
from scipy import special

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
DEFAULT_THEORY_RADIUS = 7.0 / 15.0


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
    gp = ctx.mesh.global_parameters

    kappa = float(
        (gp.get("bending_modulus_in") or 0.0) + (gp.get("bending_modulus_out") or 0.0)
    )
    kappa_t = float(
        (gp.get("tilt_modulus_in") or 0.0) + (gp.get("tilt_modulus_out") or 0.0)
    )
    drive = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)
    r_theory = float(gp.get("theory_radius") or DEFAULT_THEORY_RADIUS)

    theta_meas = float(gp.get("tilt_thetaB_value") or 0.0)
    contact_meas = float(breakdown.get("tilt_thetaB_contact_in") or 0.0)
    elastic_meas = float(
        (breakdown.get("tilt_in") or 0.0)
        + (breakdown.get("tilt_out") or 0.0)
        + (breakdown.get("bending_tilt_in") or 0.0)
        + (breakdown.get("bending_tilt_out") or 0.0)
    )
    total_meas = float(minim.compute_energy())

    theta_star = 0.0
    elastic_star = 0.0
    contact_star = 0.0
    total_star = 0.0
    if kappa > 0.0 and kappa_t > 0.0 and drive != 0.0 and r_theory > 0.0:
        lam = float(np.sqrt(kappa_t / kappa))
        x = float(lam * r_theory)
        ratio_i = float(special.iv(0, x) / special.iv(1, x))
        ratio_k = float(special.kv(0, x) / special.kv(1, x))
        den = float(ratio_i + 0.5 * ratio_k)
        theta_star = float(drive / (np.sqrt(kappa * kappa_t) * den))
        fin_star = float(np.pi * kappa * r_theory * lam * ratio_i * theta_star**2)
        fout_star = float(
            np.pi * kappa * r_theory * lam * 0.5 * ratio_k * theta_star**2
        )
        elastic_star = float(fin_star + fout_star)
        contact_star = float(-2.0 * np.pi * r_theory * drive * theta_star)
        total_star = float(elastic_star + contact_star)

    def _ratio(meas: float, theory: float) -> float:
        if abs(theory) < 1e-16:
            return 0.0
        return float(meas / theory)

    report = {
        "meta": {
            "fixture": str(mesh_path.relative_to(ROOT)),
            "protocol": list(protocol),
            "format": "yaml",
        },
        "metrics": {
            "final_energy": total_meas,
            "thetaB_value": theta_meas,
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
            "reduced_terms": {
                "elastic_measured": elastic_meas,
                "contact_measured": contact_meas,
                "total_measured": total_meas,
            },
            "theory": {
                "radius": r_theory,
                "kappa": kappa,
                "kappa_t": kappa_t,
                "drive": drive,
                "thetaB_star": theta_star,
                "elastic_star": elastic_star,
                "contact_star": contact_star,
                "total_star": total_star,
                "ratios": {
                    "theta_ratio": _ratio(theta_meas, theta_star),
                    "elastic_ratio": _ratio(elastic_meas, elastic_star),
                    "contact_ratio": _ratio(contact_meas, contact_star),
                    "total_ratio": _ratio(total_meas, total_star),
                },
            },
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
