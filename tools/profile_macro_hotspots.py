#!/usr/bin/env python3
"""Profile macro command hotspots with step timings and optional cProfile."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import re
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the project root is in sys.path when running directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.context import CommandContext
from commands.executor import execute_command_line
from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MESH = (
    ROOT
    / "meshes"
    / "caveolin"
    / "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml"
)
DEFAULT_OUTDIR = ROOT / "benchmarks" / "outputs" / "profiles"
DEFAULT_MACRO = "profile_relax_light"


def _build_context(mesh_path: Path) -> CommandContext:
    """Create a command context for a mesh file."""
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


def _slug(value: str) -> str:
    """Return a filename-safe slug."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "cmd"


def _run_macro_steps(
    *,
    context: CommandContext,
    macro_name: str,
    max_steps: int,
) -> dict[str, Any]:
    """Execute macro commands one-by-one and return timing rows."""
    macros = getattr(context.mesh, "macros", {}) or {}
    if macro_name not in macros:
        raise KeyError(f"macro not found: {macro_name}")

    steps = list(macros[macro_name])
    if max_steps > 0:
        steps = steps[:max_steps]

    rows: list[dict[str, Any]] = []
    macro_start = time.perf_counter()
    for idx, cmd in enumerate(steps, start=1):
        t0 = time.perf_counter()
        execute_command_line(context, str(cmd))
        dt = time.perf_counter() - t0
        rows.append(
            {
                "step": idx,
                "command": str(cmd),
                "seconds": float(dt),
            }
        )
    total = time.perf_counter() - macro_start
    return {
        "macro": str(macro_name),
        "num_steps": len(rows),
        "total_seconds": float(total),
        "steps": rows,
    }


def _run_profile_command(
    *,
    mesh_path: Path,
    pre_commands: list[str],
    profile_command: str,
    pstats_path: Path,
    summary_path: Path,
    top: int,
) -> dict[str, Any]:
    """Run ``profile_command`` under cProfile after optional warm-up commands."""
    context = _build_context(mesh_path)
    for cmd in pre_commands:
        execute_command_line(context, str(cmd))

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    execute_command_line(context, str(profile_command))
    profiler.disable()
    elapsed = time.perf_counter() - t0
    profiler.dump_stats(pstats_path)

    if top > 0:
        stats = pstats.Stats(profiler)
        with summary_path.open("w", encoding="utf-8") as f:
            stats.stream = f
            stats.sort_stats("cumulative")
            stats.print_stats(top)
            f.write("\n")
            stats.sort_stats("tottime")
            stats.print_stats(top)

    return {
        "command": str(profile_command),
        "pre_commands": [str(c) for c in pre_commands],
        "elapsed_seconds": float(elapsed),
        "pstats": str(pstats_path),
        "summary": str(summary_path) if top > 0 else None,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh",
        default=str(DEFAULT_MESH),
        help="Path to JSON/YAML mesh input.",
    )
    parser.add_argument(
        "--macro",
        default=DEFAULT_MACRO,
        help="Macro name to time step-by-step.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="If > 0, only execute the first N macro steps.",
    )
    parser.add_argument(
        "--profile-command",
        default="",
        help="Optional command to profile under cProfile (e.g. g1).",
    )
    parser.add_argument(
        "--pre-command",
        action="append",
        default=[],
        help="Optional command to execute before --profile-command. Repeatable.",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help="Directory for JSON/pstats/text outputs.",
    )
    parser.add_argument(
        "--label",
        default="macro_hotspots",
        help="Basename label for output files.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="Top entries for cProfile text summary (0 to skip summary).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step timing table output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run macro timing and optional command profiling."""
    args = _parse_args(argv)
    mesh_path = Path(args.mesh)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    context = _build_context(mesh_path)
    step_report = _run_macro_steps(
        context=context,
        macro_name=str(args.macro),
        max_steps=int(args.max_steps),
    )
    step_report["mesh"] = str(mesh_path)
    step_report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    steps_json = outdir / f"{args.label}_steps.json"
    steps_json.write_text(json.dumps(step_report, indent=2), encoding="utf-8")

    if not bool(args.quiet):
        print("step\tseconds\tcommand")
        for row in step_report["steps"]:
            print(f"{row['step']}\t{row['seconds']:.6f}\t{row['command']}")
        print(f"total\t{step_report['total_seconds']:.6f}\t{args.macro}")
    print(f"wrote: {steps_json}")

    profile_command = str(args.profile_command).strip()
    if profile_command:
        slug = _slug(profile_command)
        pstats_path = outdir / f"{args.label}_{slug}.pstats"
        summary_path = outdir / f"{args.label}_{slug}.txt"
        profile_report = _run_profile_command(
            mesh_path=mesh_path,
            pre_commands=[str(c) for c in args.pre_command],
            profile_command=profile_command,
            pstats_path=pstats_path,
            summary_path=summary_path,
            top=int(args.top),
        )
        print(f"wrote: {pstats_path}")
        if int(args.top) > 0:
            print(f"wrote: {summary_path}")

        report_json = outdir / f"{args.label}_{slug}.json"
        report_json.write_text(json.dumps(profile_report, indent=2), encoding="utf-8")
        print(f"wrote: {report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
