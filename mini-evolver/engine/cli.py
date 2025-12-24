"""CLI entrypoint for mini evolver."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

from . import numeric
from .commands import execute_commands
from .engine import run
from .parser import load_mesh
from .plot import LivePlotter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal Python reimplementation of Evolver-style area minimization."
    )
    parser.add_argument(
        "input_file",
        help="Path to geometry input file (.json, .yaml, or .fe)",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=100.0,
        help="Penalty weight enforcing target volume (default: 100.0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help="Maximum gradient descent steps (default: 400)",
    )
    parser.add_argument(
        "--enforce-volume",
        action="store_true",
        help="Nudge each step along the volume gradient to keep volume on target",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a 3D matplotlib plot of the final surface",
    )
    parser.add_argument(
        "--save-plot",
        metavar="PATH",
        help="Save a 3D plot image instead of (or in addition to) showing it",
    )
    parser.add_argument(
        "--refine",
        type=int,
        default=0,
        help="Apply Evolver-style 'r' refinement this many times before minimizing",
    )
    parser.add_argument(
        "--no-gogo",
        action="store_true",
        help="Ignore gogo macro even if present in the input",
    )
    parser.add_argument(
        "--hessian-steps",
        type=int,
        default=5,
        help="Number of quasi-Newton steps for each 'hessian' command",
    )
    parser.add_argument(
        "--macro",
        help="Macro name to execute in non-interactive mode",
    )
    parser.add_argument(
        "--no-numpy",
        action="store_true",
        help="Disable numpy acceleration even if numpy is installed",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run once and exit (interactive is default)",
    )
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="Show live plot updates during minimization",
    )
    parser.add_argument(
        "--volume-mode",
        choices=["penalty", "saddle"],
        default="penalty",
        help="Volume constraint mode (default: penalty)",
    )
    args = parser.parse_args()
    if args.no_numpy:
        numeric.set_use_numpy(False)
    mesh = load_mesh(args.input_file)
    if args.volume_mode == "saddle":
        args.enforce_volume = True

    def run_read_commands() -> Dict[int, Tuple[float, float, float]]:
        positions = mesh.current_positions()
        if mesh.read_commands:
            commands = []
            for line in mesh.read_commands:
                commands.extend([c.strip() for c in line.split(";") if c.strip()])
            _, positions, _ = execute_commands(
                mesh,
                positions,
                commands,
                args.penalty,
                args.enforce_volume,
                args.hessian_steps,
                args.volume_mode,
            )
        return positions

    def print_status(positions: Dict[int, Tuple[float, float, float]]) -> None:
        energy_penalty = args.penalty if args.volume_mode == "penalty" else 0.0
        energy, area, volume, _, grad_norm = mesh.energy(positions, energy_penalty)
        print(
            f"energy {energy:.6f}  area {area:.6f}  volume {volume:.6f}  grad {grad_norm:.3e}"
        )

    if args.non_interactive:
        positions = run_read_commands()
        macro_name = args.macro.lower() if args.macro else None
        if macro_name and macro_name in mesh.macros and not args.no_gogo:
            mesh, positions, _ = execute_commands(
                mesh,
                positions,
                [macro_name],
                args.penalty,
                args.enforce_volume,
                args.hessian_steps,
                args.volume_mode,
            )
            print_status(positions)
            if args.plot or args.save_plot:
                run(
                    mesh,
                    penalty=args.penalty,
                    max_steps=0,
                    enforce_volume=args.enforce_volume,
                    plot=args.plot,
                    save_plot=args.save_plot,
                    refine_steps=0,
                    use_gogo=False,
                    hessian_steps=args.hessian_steps,
                    macro_name=macro_name or "gogo",
                    volume_mode=args.volume_mode,
                    live_plot=args.live_plot,
                )
            return
        run(
            mesh,
            penalty=args.penalty,
            max_steps=args.steps,
            enforce_volume=args.enforce_volume,
            plot=args.plot,
            save_plot=args.save_plot,
            refine_steps=args.refine,
            use_gogo=False,
            hessian_steps=args.hessian_steps,
            macro_name=macro_name or "gogo",
            volume_mode=args.volume_mode,
            live_plot=args.live_plot,
        )
        return

    positions = run_read_commands()
    plotter = LivePlotter(mesh) if args.live_plot else None
    on_step = plotter.update if plotter else None
    if plotter:
        plotter.update(positions, 0)
    print(
        "Interactive mode. Type commands (e.g., 'g 5; r; g 5') or a macro name. 'q' to quit."
    )
    while True:
        try:
            line = input("evolver> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in {"q", "quit", "exit"}:
            break
        commands = [c.strip() for c in line.split(";") if c.strip()]
        if any(cmd.lower() == "s" for cmd in commands):
            if not plotter:
                plotter = LivePlotter(mesh)
                on_step = plotter.update
                plotter.update(positions, 0)
            commands = [cmd for cmd in commands if cmd.lower() != "s"]
            if not commands:
                continue
        mesh, positions, _ = execute_commands(
            mesh,
            positions,
            commands,
            args.penalty,
            args.enforce_volume,
            args.hessian_steps,
            args.volume_mode,
            on_step=on_step,
        )
        print_status(positions)
    if plotter:
        plotter.close()
