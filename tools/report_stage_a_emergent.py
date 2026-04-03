#!/usr/bin/env python3
"""Run the Stage A emergent lane and write a YAML report."""

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
from tools.build_stage_a_fixtures import (
    BASE_FIXTURE,
    SEEDED_FIXTURE,
    STAGE_A_RIM_SOURCE_STRENGTH,
    write_stage_a_fixtures,
)
from tools.diagnostics.flat_disk_one_leaflet_theory import (
    compute_flat_disk_theory,
    tex_reference_params,
)
from tools.reproduce_flat_disk_one_leaflet import _build_minimizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = (
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "stage_a_emergent_report.yaml"
)

TARGET_EMERGENT_THETA = 0.18
EMERGENT_Z_THRESHOLD = 5.0e-5
EMERGENT_THETA_THRESHOLD = 1.0e-2


def _load_mesh_from_fixture(path: Path):
    return parse_geometry(load_data(str(path)))


def _disk_boundary_rows(mesh) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") != "disk":
            continue
        rows.append(int(mesh.vertex_index_to_row[int(vid)]))
    return np.asarray(rows, dtype=int)


def _disk_rows(mesh) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if str(opts.get("preset") or "") != "disk":
            continue
        rows.append(int(mesh.vertex_index_to_row[int(vid)]))
    return np.asarray(rows, dtype=int)


def _radial_frame(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_vec = np.array(positions, copy=True)
    r_vec[:, 2] = 0.0
    r_len = np.linalg.norm(r_vec, axis=1)
    r_hat = np.zeros_like(r_vec)
    valid = r_len > 1.0e-12
    r_hat[valid] = r_vec[valid] / r_len[valid, None]
    return r_len, r_hat


def _measure_state(mesh, minimizer) -> dict[str, Any]:
    positions = mesh.positions_view()
    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    tilts_sum = tilts_in + tilts_out

    boundary_rows = _disk_boundary_rows(mesh)
    boundary_r, boundary_hat = _radial_frame(positions[boundary_rows])
    boundary_valid = boundary_r > 1.0e-12

    theta_total = np.sum(tilts_sum[boundary_rows] * boundary_hat, axis=1)
    theta_in = np.sum(tilts_in[boundary_rows] * boundary_hat, axis=1)
    theta_out = np.sum(tilts_out[boundary_rows] * boundary_hat, axis=1)

    disk_rows = _disk_rows(mesh)
    disk_r, _ = _radial_frame(positions[disk_rows])
    disk_theta = np.sum(
        tilts_sum[disk_rows] * _radial_frame(positions[disk_rows])[1], axis=1
    )
    disk_valid = disk_r > 1.0e-12

    profile_rmse = float("nan")
    if np.any(disk_valid) and np.any(boundary_valid):
        r_disk = disk_r[disk_valid]
        theta_disk = disk_theta[disk_valid]
        order = np.argsort(r_disk)
        r_disk = r_disk[order]
        theta_disk = theta_disk[order]
        r_unique = np.unique(np.round(r_disk, 6))
        pairs: list[tuple[float, float]] = []
        for radius in r_unique:
            mask = np.isclose(r_disk, radius, atol=1.0e-6)
            pairs.append((float(radius), float(np.mean(theta_disk[mask]))))
        if pairs:
            radius_max = max(radius for radius, _ in pairs)
            theta_boundary = max(value for radius, value in pairs if radius > 0.4)
            lambda_value = float(np.sqrt(1.0 / 225.0))
            errors: list[float] = []
            for radius, value in pairs:
                if radius <= 1.0e-8 or abs(theta_boundary) <= 1.0e-12:
                    continue
                ref = float(
                    special.iv(1, radius / lambda_value)
                    / special.iv(1, radius_max / lambda_value)
                )
                errors.append(float((value / theta_boundary) - ref))
            if errors:
                profile_rmse = float(np.sqrt(np.mean(np.square(errors))))

    theta_mean = float(np.mean(theta_total[boundary_valid]))
    theta_in_mean = float(np.mean(theta_in[boundary_valid]))
    theta_out_mean = float(np.mean(theta_out[boundary_valid]))
    theta_std = float(np.std(theta_total[boundary_valid]))
    zmax = float(np.max(np.abs(positions[:, 2])))
    total_energy = float(minimizer.compute_energy())
    breakdown = minimizer.compute_energy_breakdown()

    branch = "emergent_curved"
    if theta_mean <= EMERGENT_THETA_THRESHOLD and zmax <= EMERGENT_Z_THRESHOLD:
        branch = "flat_control"

    return {
        "branch": branch,
        "theta_mean": theta_mean,
        "theta_in_mean": theta_in_mean,
        "theta_out_mean": theta_out_mean,
        "phi_mean": theta_out_mean,
        "theta_std": theta_std,
        "zmax": zmax,
        "total_energy": total_energy,
        "profile_rmse": profile_rmse,
        "vertex_count": int(len(mesh.vertex_ids)),
        "energy_breakdown": {str(k): float(v) for k, v in breakdown.items()},
    }


def _run_commands(ctx: CommandContext, commands: list[str]) -> None:
    for command in commands:
        execute_command_line(ctx, command)


def _run_fixture(path: Path, commands: list[str]) -> dict[str, Any]:
    mesh = _load_mesh_from_fixture(path)
    minimizer = _build_minimizer(mesh)
    ctx = CommandContext(mesh, minimizer, minimizer.stepper)
    _run_commands(ctx, commands)
    metrics = _measure_state(ctx.mesh, minimizer)
    metrics["commands"] = list(commands)
    metrics["fixture"] = _report_fixture_path(path)
    return metrics


def _run_continuation(path: Path) -> dict[str, Any]:
    commands: list[str] = []
    mesh = _load_mesh_from_fixture(path)
    minimizer = _build_minimizer(mesh)
    ctx = CommandContext(mesh, minimizer, minimizer.stepper)
    for strength in (0.125, 0.25, 0.375, STAGE_A_RIM_SOURCE_STRENGTH):
        mesh.global_parameters.set("tilt_rim_source_strength", float(strength))
        commands.append(f"set_tilt_rim_source_strength={strength:.3f}")
        commands.append("g3")
        execute_command_line(ctx, "g3")
    metrics = _measure_state(ctx.mesh, minimizer)
    metrics["commands"] = commands
    metrics["fixture"] = _report_fixture_path(path)
    return metrics


def _report_fixture_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def build_stage_a_report(
    *,
    base_fixture: Path = BASE_FIXTURE,
    seeded_fixture: Path = SEEDED_FIXTURE,
) -> dict[str, Any]:
    write_stage_a_fixtures()
    theory = compute_flat_disk_theory(tex_reference_params())
    report = {
        "meta": {
            "base_fixture": _report_fixture_path(base_fixture),
            "seeded_fixture": _report_fixture_path(seeded_fixture),
            "target_emergent_theta": TARGET_EMERGENT_THETA,
            "emergent_theta_threshold": EMERGENT_THETA_THRESHOLD,
            "emergent_z_threshold": EMERGENT_Z_THRESHOLD,
        },
        "flat_reference": {
            "theta_star": float(theory.theta_star),
            "total_energy": float(theory.total),
        },
        "cases": {
            "base": _run_fixture(base_fixture, ["g5"]),
            "seeded": _run_fixture(seeded_fixture, ["g5"]),
            "continuation": _run_continuation(base_fixture),
            "refined": _run_fixture(base_fixture, ["g3", "r", "g3"]),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="YAML output path for the Stage A report.",
    )
    args = parser.parse_args()

    report = build_stage_a_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
