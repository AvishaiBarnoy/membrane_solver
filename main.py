import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt

from commands.context import CommandContext
from commands.registry import get_command
from geometry.geom_io import load_data, parse_geometry, save_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.logging_config import setup_logging
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

logger = logging.getLogger("membrane_solver")


def resolve_json_path(path: str) -> str:
    """Return a valid JSON file path, allowing path without extension."""
    if os.path.isfile(path):
        return path
    if not path.lower().endswith(".json"):
        alt = path + ".json"
        if os.path.isfile(alt):
            return alt
    raise FileNotFoundError(f"Cannot find file '{path}' or '{path}.json'")


def main():
    parser = argparse.ArgumentParser(description="Membrane Solver Simulation Driver")
    parser.add_argument("-i", "--input", help="Input mesh JSON file")
    parser.add_argument("-o", "--output", default=None, help="Output mesh JSON file")
    parser.add_argument(
        "--debugger",
        action="store_true",
        help="Enter a post-mortem debugger (ipdb/pdb) on uncaught exceptions.",
    )
    parser.add_argument(
        "--compact-output-json",
        action="store_true",
        help="Write output JSON in compact (single-line) form.",
    )
    parser.add_argument(
        "--instructions", help="Optional instruction file (one command per line)"
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Visualize the input geometry and exit (no minimization).",
    )
    parser.add_argument(
        "--viz-save",
        default=None,
        help="Save the visualization image to PATH instead of only showing it.",
    )
    parser.add_argument(
        "--viz-no-facets",
        action="store_true",
        help="Disable drawing of polygonal facets in --viz mode.",
    )
    parser.add_argument(
        "--viz-no-edges",
        action="store_true",
        help="Disable drawing of edges in --viz mode.",
    )
    parser.add_argument(
        "--viz-scatter",
        action="store_true",
        help="Draw vertices as red scatter points in --viz mode.",
    )
    parser.add_argument(
        "--viz-show-indices",
        action="store_true",
        help="Annotate vertices with their indices in --viz mode.",
    )
    parser.add_argument(
        "--viz-transparent",
        action="store_true",
        help="Render facets semi-transparent in --viz mode.",
    )
    parser.add_argument(
        "--viz-no-axes",
        action="store_true",
        help="Remove axes from the plot in --viz mode.",
    )
    parser.add_argument("--log", default=None, help="Optional log file")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress console output"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip interactive mode after executing instructions",
    )
    parser.add_argument(
        "--properties",
        action="store_true",
        help="Print basic physical properties (volume, surface area, etc.) and exit",
    )
    parser.add_argument(
        "--volume-mode",
        choices=["lagrange", "penalty"],
        default=None,
        help="Override volume constraint mode (lagrange = hard constraint; penalty = soft energy).",
    )
    parser.add_argument(
        "--line-tension",
        type=float,
        default=None,
        help="Assign a line-tension modulus to edges. When combined with "
        "`--line-tension-edges`, only the specified edge IDs are tagged. "
        "Otherwise all edges receive line tension.",
    )
    parser.add_argument(
        "--line-tension-edges",
        type=str,
        default=None,
        help="Comma-separated edge IDs to receive CLI line tension.",
    )
    args = parser.parse_args()

    old_excepthook = sys.excepthook
    if args.debugger:
        import traceback

        def _post_mortem_excepthook(exc_type, exc, tb):
            if issubclass(exc_type, KeyboardInterrupt):
                return old_excepthook(exc_type, exc, tb)
            traceback.print_exception(exc_type, exc, tb)
            try:
                import ipdb  # type: ignore

                ipdb.post_mortem(tb)
            except Exception:
                import pdb

                pdb.post_mortem(tb)

        sys.excepthook = _post_mortem_excepthook

    if not args.input:
        try:
            args.input = input("Input mesh JSON file: ").strip()
        except EOFError:
            print("No input file provided.", file=sys.stderr)
            sys.exit(1)
    try:
        args.input = resolve_json_path(args.input)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    global logger
    logger = setup_logging(
        args.log if args.log else "membrane_solver.log",
        quiet=args.quiet,
        debug=args.debug,
    )

    # Load mesh and parameters
    data = load_data(args.input)
    mesh = parse_geometry(data)

    if args.viz or args.viz_save:
        from visualization.plotting import plot_geometry

        show = args.viz_save is None
        plot_geometry(
            mesh,
            show_indices=args.viz_show_indices,
            scatter=args.viz_scatter,
            transparent=args.viz_transparent,
            draw_facets=not args.viz_no_facets,
            draw_edges=not args.viz_no_edges,
            no_axes=args.viz_no_axes,
            show=show,
        )
        if args.viz_save:
            fig = plt.gcf()
            fig.savefig(args.viz_save, bbox_inches="tight")
            logger.info("Saved visualization to %s", args.viz_save)
        return

    def _apply_cli_line_tension(value: float, edge_ids: list[int] | None) -> None:
        targets = edge_ids or list(mesh.edges.keys())
        invalid = [idx for idx in targets if idx not in mesh.edges]
        if invalid:
            logger.warning("Ignoring unknown edge IDs for line tension: %s", invalid)
        updated = False
        for idx in targets:
            edge = mesh.edges.get(idx)
            if edge is None:
                continue
            opts = edge.options if hasattr(edge, "options") else {}
            if isinstance(opts.get("energy"), str):
                opts["energy"] = [opts["energy"]]
            opts.setdefault("energy", [])
            if "line_tension" not in opts["energy"]:
                opts["energy"].append("line_tension")
            opts["line_tension"] = value
            edge.options = opts
            updated = True
        if updated and "line_tension" not in mesh.energy_modules:
            mesh.energy_modules.append("line_tension")

    if args.line_tension is not None:
        ids = None
        if args.line_tension_edges:
            try:
                ids = [
                    int(token.strip())
                    for token in args.line_tension_edges.split(",")
                    if token.strip()
                ]
            except ValueError as exc:
                print(
                    f"Invalid --line-tension-edges value '{args.line_tension_edges}': {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
        _apply_cli_line_tension(args.line_tension, ids)

    fixed_count = sum(1 for v in mesh.vertices.values() if getattr(v, "fixed", False))
    logger.debug(f"Number of fixed vertices: {fixed_count} / {len(mesh.vertices)}")
    if mesh.bodies:
        body0 = mesh.bodies.get(0)
        if body0 is not None:
            target_vol = body0.options.get("target_volume")
            logger.debug(f"Body 0 target_volume: {target_vol}")

    global_params = mesh.global_parameters
    if args.volume_mode:
        global_params.set("volume_constraint_mode", args.volume_mode)
    energy_manager = EnergyModuleManager(mesh.energy_modules)
    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
    stepper = GradientDescent()

    # Initialize Minimizer
    minimizer = Minimizer(
        mesh,
        global_params,
        stepper,
        energy_manager,
        constraint_manager,
        quiet=args.quiet,
    )
    minimizer.step_size = global_params.get("step_size", 0.001)

    # Initialize Command Context
    context = CommandContext(mesh, minimizer, stepper)

    if args.properties:
        cmd, _ = get_command("properties")
        cmd.execute(context, [])
        return

    # Load instructions from file or mesh
    if args.instructions:
        with open(args.instructions, "r") as f:
            lines = f.readlines()
    else:
        lines = mesh.instructions if hasattr(mesh, "instructions") else []

    # Execute initial instructions
    logger.debug(f"Executing {len(lines)} initial instructions.")
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cmd_name = parts[0]
        cmd_args = parts[1:]

        command, extra_args = get_command(cmd_name)
        if command:
            command.execute(context, extra_args + cmd_args)
        else:
            logger.warning(f"Unknown instruction: {cmd_name}")

    if not args.non_interactive:
        while not context.should_exit:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue

            parts = line.split()
            cmd_name = parts[0]
            cmd_args = parts[1:]

            command, extra_args = get_command(cmd_name)
            if command:
                try:
                    command.execute(context, extra_args + cmd_args)
                except Exception as e:
                    logger.error(f"Error executing command '{cmd_name}': {e}")
            else:
                print(f"Unknown command: {cmd_name}")

    try:
        if args.output:
            save_geometry(context.mesh, args.output, compact=args.compact_output_json)
            logger.info(f"Simulation complete. Output saved to {args.output}")
        else:
            logger.info("Simulation complete. No output file written.")
    finally:
        if args.debugger:
            sys.excepthook = old_excepthook


if __name__ == "__main__":
    main()
