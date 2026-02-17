#!/usr/bin/env python3
"""Reproduce theory-parity metrics and write a YAML report."""

from __future__ import annotations

import argparse
import copy
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
DEFAULT_EXPANSION_POLICY = (
    ROOT / "tests" / "fixtures" / "theory_parity_expansion_policy.yaml"
)
DEFAULT_EXPANSION_STATE = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "diagnostics"
    / "theory_parity_expansion_state.yaml"
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


def _collect_report_from_context(
    *,
    ctx: CommandContext,
    mesh_path: Path,
    protocol: tuple[str, ...],
) -> dict[str, Any]:
    minim = ctx.minimizer
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


def _collect_report(
    mesh_path: Path, protocol: tuple[str, ...], fixed_polish_steps: int = 0
) -> dict[str, Any]:
    ctx = _build_context(mesh_path)
    for cmd in protocol:
        execute_command_line(ctx, cmd)
    for _ in range(int(fixed_polish_steps)):
        execute_command_line(ctx, "g1")
    report = _collect_report_from_context(
        ctx=ctx, mesh_path=mesh_path, protocol=protocol
    )
    report["meta"]["fixed_polish_steps"] = int(fixed_polish_steps)
    return report


def _load_yaml(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return copy.deepcopy(default) if default is not None else {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _max_ratio_drift(
    cur_ratios: dict[str, float], prev_ratios: dict[str, float]
) -> float:
    keys = sorted(set(cur_ratios.keys()) & set(prev_ratios.keys()))
    if not keys:
        return 0.0
    return float(max(abs(float(cur_ratios[k]) - float(prev_ratios[k])) for k in keys))


def _is_finite_metrics(report: dict[str, Any]) -> bool:
    vals = [
        float(report["metrics"]["final_energy"]),
        float(report["metrics"]["reduced_terms"]["elastic_measured"]),
        float(report["metrics"]["reduced_terms"]["contact_measured"]),
        float(report["metrics"]["reduced_terms"]["total_measured"]),
    ]
    vals.extend(float(v) for v in report["metrics"]["theory"]["ratios"].values())
    return all(np.isfinite(v) for v in vals)


def _default_state() -> dict[str, Any]:
    return {
        "current_stage": 0,
        "stage3_pass_streak": 0,
        "stage4_fail_streak": 0,
        "stage4_locked": False,
        "last_stage3_ratios": {},
        "last_stage3_energy": None,
        "stage3_anchor_ratios": {},
        "stage3_anchor_energy": None,
        "history": [],
    }


def update_expansion_state(
    *,
    state: dict[str, Any],
    report: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Update expansion state from one run and return (new_state, decisions)."""
    new_state = copy.deepcopy(state) if state else _default_state()
    thresholds = policy["thresholds"]
    stage = int(new_state.get("current_stage", 0))
    ratios = {
        str(k): float(v) for k, v in report["metrics"]["theory"]["ratios"].items()
    }
    energy = float(report["metrics"]["final_energy"])
    decisions: dict[str, Any] = {
        "stage_before": stage,
        "promoted_to_stage4": False,
        "rolled_back_to_stage3": False,
        "stage4_validation_passed": None,
    }

    finite_ok = _is_finite_metrics(report)
    if stage == 3:
        prev_ratios = new_state.get("last_stage3_ratios") or {}
        prev_energy = new_state.get("last_stage3_energy")
        if prev_ratios and prev_energy is not None:
            ratio_drift = _max_ratio_drift(ratios, prev_ratios)
            energy_drift = abs(energy - float(prev_energy))
            pass_ok = (
                finite_ok
                and ratio_drift <= float(thresholds["ratio_drift_max"])
                and energy_drift <= float(thresholds["energy_drift_max"])
            )
        else:
            pass_ok = finite_ok
        if pass_ok:
            new_state["stage3_pass_streak"] = (
                int(new_state.get("stage3_pass_streak", 0)) + 1
            )
        else:
            new_state["stage3_pass_streak"] = 0
        new_state["last_stage3_ratios"] = ratios
        new_state["last_stage3_energy"] = energy
        if not bool(new_state.get("stage4_locked", False)) and int(
            new_state["stage3_pass_streak"]
        ) >= int(thresholds["consecutive_passes_to_promote"]):
            new_state["current_stage"] = 4
            new_state["stage4_fail_streak"] = 0
            new_state["stage3_anchor_ratios"] = ratios
            new_state["stage3_anchor_energy"] = energy
            decisions["promoted_to_stage4"] = True

    elif stage == 4:
        stage4_cfg = thresholds["stage4"]
        anchor_ratios = new_state.get("stage3_anchor_ratios") or {}
        ratio_delta = _max_ratio_drift(ratios, anchor_ratios) if anchor_ratios else 0.0
        pass_ok = (
            finite_ok
            and ratio_delta <= float(stage4_cfg["ratio_delta_vs_stage3_max"])
            and np.isfinite(energy)
        )
        decisions["stage4_validation_passed"] = bool(pass_ok)
        if pass_ok:
            new_state["stage4_fail_streak"] = 0
        else:
            new_state["stage4_fail_streak"] = (
                int(new_state.get("stage4_fail_streak", 0)) + 1
            )
            if int(new_state["stage4_fail_streak"]) >= int(
                stage4_cfg["max_consecutive_failures_before_rollback"]
            ):
                new_state["current_stage"] = 3
                new_state["stage4_locked"] = True
                new_state["stage3_pass_streak"] = 0
                new_state["stage4_fail_streak"] = 0
                decisions["rolled_back_to_stage3"] = True

    decisions["stage_after"] = int(new_state.get("current_stage", stage))
    new_state.setdefault("history", []).append(
        {
            "stage_before": int(stage),
            "stage_after": int(new_state.get("current_stage", stage)),
            "energy": float(energy),
            "ratios": ratios,
            "decisions": decisions,
        }
    )
    return new_state, decisions


def _stage_suffix(policy: dict[str, Any], stage: int) -> list[str]:
    stages = policy["stages"]
    key = f"stage_{stage}"
    return [str(x) for x in stages.get(key, [])]


def _attach_stage_metadata(
    report: dict[str, Any], *, stage: int, suffix: list[str], ctx: CommandContext
) -> None:
    report["meta"]["expansion"] = {
        "stage": int(stage),
        "stage_suffix": list(suffix),
    }
    if stage == 4:
        report["meta"]["expansion"]["post_refine_mesh"] = {
            "vertex_count": int(len(ctx.mesh.vertices)),
            "triangle_count": int(len(ctx.mesh.facets)),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--protocol-mode",
        choices=("fixed", "expanded"),
        default="fixed",
        help="Use fixed protocol or convergence-gated expansion ladder.",
    )
    parser.add_argument(
        "--protocol",
        nargs="+",
        default=list(DEFAULT_PROTOCOL),
        help="Command sequence to reproduce parity metrics.",
    )
    parser.add_argument(
        "--expansion-policy",
        type=Path,
        default=DEFAULT_EXPANSION_POLICY,
        help="YAML policy for expanded protocol stages and gates.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_EXPANSION_STATE,
        help="Persistent YAML state file for expansion mode.",
    )
    parser.add_argument(
        "--fixed-polish-steps",
        type=int,
        default=0,
        help="Additional trailing g1 steps for fixed mode only (default: 0).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mesh_path = Path(args.mesh)
    out_path = Path(args.out)
    protocol = tuple(str(x) for x in args.protocol)
    fixed_polish_steps = int(args.fixed_polish_steps)
    if fixed_polish_steps < 0:
        raise ValueError("--fixed-polish-steps must be >= 0")

    if str(args.protocol_mode) == "fixed":
        report = _collect_report(
            mesh_path=mesh_path,
            protocol=protocol,
            fixed_polish_steps=fixed_polish_steps,
        )
    else:
        policy = _load_yaml(Path(args.expansion_policy))
        state_path = Path(args.state_file)
        state = _load_yaml(state_path, default=_default_state())
        stage = int(state.get("current_stage", 0))
        suffix = _stage_suffix(policy, stage)
        full_protocol = protocol + tuple(suffix)
        ctx = _build_context(mesh_path)
        for cmd in full_protocol:
            execute_command_line(ctx, cmd)
        report = _collect_report_from_context(
            ctx=ctx,
            mesh_path=mesh_path,
            protocol=full_protocol,
        )
        _attach_stage_metadata(report, stage=stage, suffix=suffix, ctx=ctx)
        new_state, decisions = update_expansion_state(
            state=state,
            report=report,
            policy=policy,
        )
        report["meta"]["expansion"]["decisions"] = decisions
        _save_yaml(state_path, new_state)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
