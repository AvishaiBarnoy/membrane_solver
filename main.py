import argparse
import atexit
import logging
import os
import sys
from pathlib import Path

from commands.completion import command_line_completions
from commands.context import CommandContext
from commands.executor import execute_command_line
from commands.registry import COMMAND_REGISTRY, get_command
from core.exceptions import BodyOrientationError
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


def _setup_interactive_history() -> None:
    """Enable up/down arrow history in the interactive prompt (when available).

    Uses the stdlib ``readline`` module when present. History is persisted to a
    file across sessions when running in a TTY.

    Environment variables
    ---------------------
    MEMBRANE_HISTORY_FILE:
        Override the path for the history file (default:
        ``~/.membrane_solver_history``).
    MEMBRANE_HISTORY_LENGTH:
        Max number of history entries to keep (default: 2000).
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return

    try:
        import readline  # noqa: F401
    except ImportError:
        return

    import readline

    history_path = os.environ.get("MEMBRANE_HISTORY_FILE")
    if not history_path:
        history_path = str(Path.home() / ".membrane_solver_history")

    try:
        history_len = int(os.environ.get("MEMBRANE_HISTORY_LENGTH", "2000"))
    except ValueError:
        history_len = 2000

    try:
        readline.set_history_length(history_len)
    except Exception:
        pass

    history_file = Path(history_path).expanduser()
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        readline.read_history_file(str(history_file))
    except FileNotFoundError:
        pass
    except Exception:
        # History is a UX improvement; never fail program startup on it.
        return

    def _save_history() -> None:
        try:
            readline.write_history_file(str(history_file))
        except Exception:
            pass

    atexit.register(_save_history)


def _setup_interactive_completion(context: CommandContext) -> None:
    """Enable tab-completion for interactive commands (when available)."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return

    try:
        import readline  # noqa: F401
    except ImportError:
        return

    import readline

    def _completer(text: str, state: int):
        macros = getattr(context.mesh, "macros", {}) or {}
        candidates = command_line_completions(
            text=text,
            line_buffer=readline.get_line_buffer(),
            command_names=COMMAND_REGISTRY.keys(),
            macro_names=macros.keys(),
        )
        if state < len(candidates):
            return candidates[state]
        return None

    try:
        readline.set_completer(_completer)
        readline.parse_and_bind("tab: complete")
    except Exception:
        # Completion is a UX improvement; never fail program startup on it.
        return


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
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument(
        "--viz-tilt",
        action="store_true",
        help="Color facets by |t| (tilt magnitude) in --viz mode.",
    )
    viz_group.add_argument(
        "--viz-tilt-div",
        action="store_true",
        help="Color facets by div(t) in --viz mode.",
    )
    parser.add_argument(
        "--viz-tilt-arrows",
        action="store_true",
        help="Overlay tilt arrows in --viz mode.",
    )
    parser.add_argument(
        "--viz-tilt-arrows-max",
        type=int,
        default=2000,
        help="Maximum number of tilt arrows to draw (default: 2000).",
    )
    parser.add_argument(
        "--viz-tilt-arrow-scale",
        type=float,
        default=0.1,
        help="Arrow length as a fraction of the plot span (default: 0.1).",
    )
    parser.add_argument(
        "--viz-tilt-streamlines",
        action="store_true",
        help="Overlay simple tilt streamlines in --viz mode.",
    )
    parser.add_argument(
        "--viz-patch-boundaries",
        action="store_true",
        help="Overlay facet patch boundaries in --viz mode.",
    )
    parser.add_argument(
        "--viz-patch-key",
        default="disk_patch",
        help="Facet option key used for patch labels (default: disk_patch).",
    )
    parser.add_argument(
        "--viz-boundary-loops",
        action="store_true",
        help='Overlay mesh boundary loops ("holes") in --viz mode.',
    )
    parser.add_argument(
        "--viz-boundary-geodesic",
        action="store_true",
        help="Annotate boundary loops with geodesic curvature sums in --viz mode.",
    )
    parser.add_argument(
        "--viz-no-axes",
        action="store_true",
        help="Remove axes from the plot in --viz mode.",
    )
    parser.add_argument(
        "--log",
        nargs="?",
        const="auto",
        default=None,
        help="Write logs to a file. If PATH is omitted, writes a log file next to the input mesh.",
    )
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
        "--radius-of-gyration",
        action="store_true",
        help="Print the surface radius of gyration and exit",
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
    log_path = None
    if args.log is not None:
        if args.log == "auto":
            inp = Path(args.input)
            log_path = str(inp.with_suffix(".log"))
        else:
            log_path = str(args.log)
    logger = setup_logging(log_path, quiet=args.quiet, debug=args.debug)
    if args.debug and args.output is None:
        if os.access("/tmp", os.W_OK):
            args.output = "/tmp/out.yaml"
        else:
            logger.warning(
                "Debug output requested, but /tmp is not writable; no output file will be written."
            )

    # Load mesh and parameters
    data = load_data(args.input)
    try:
        mesh = parse_geometry(data)
    except BodyOrientationError as exc:
        logger.error("%s", exc)
        bad_mesh = getattr(exc, "mesh", None)
        if bad_mesh is None:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            print(
                "Body orientation is inconsistent. Run in a TTY to fix interactively.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            answer = (
                input(
                    "Body orientation is inconsistent. Fix and save corrected geometry? [y/N] "
                )
                .strip()
                .lower()
            )
        except EOFError:
            sys.exit(1)
        if answer not in {"y", "yes"}:
            sys.exit(1)

        flipped = 0
        outward_flipped = 0
        for bid in sorted(bad_mesh.bodies):
            flipped += bad_mesh.orient_body_facets(bid)
            outward_flipped += bad_mesh.orient_body_outward(bid)
        bad_mesh.validate_body_orientation()
        bad_mesh.validate_body_outwardness()

        inp = Path(args.input)
        fixed_path = inp.with_name(f"{inp.stem}.oriented.json")
        save_geometry(bad_mesh, str(fixed_path), compact=args.compact_output_json)
        logger.info(
            "Saved oriented geometry to %s (flipped %d facets, outward flips %d).",
            fixed_path,
            flipped,
            outward_flipped,
        )
        mesh = bad_mesh

    macros = getattr(mesh, "macros", {}) or {}
    if macros:
        print("Macros:")
        for name, steps in macros.items():
            if isinstance(steps, list):
                body = "; ".join(str(step) for step in steps)
            else:
                body = str(steps)
            print(f"  {name}: {body}")

    if args.viz or args.viz_save:
        import matplotlib.pyplot as plt

        from visualization.plotting import plot_geometry

        show = args.viz_save is None
        color_by = None
        if getattr(args, "viz_tilt_div", False):
            color_by = "tilt_div"
        elif getattr(args, "viz_tilt", False):
            color_by = "tilt_mag"
        plot_geometry(
            mesh,
            show_indices=args.viz_show_indices,
            scatter=args.viz_scatter,
            transparent=args.viz_transparent,
            draw_facets=not args.viz_no_facets,
            draw_edges=not args.viz_no_edges,
            no_axes=args.viz_no_axes,
            color_by=color_by,
            show_tilt_arrows=getattr(args, "viz_tilt_arrows", False),
            tilt_arrows_max=(
                None
                if int(getattr(args, "viz_tilt_arrows_max", 2000)) <= 0
                else int(getattr(args, "viz_tilt_arrows_max", 2000))
            ),
            tilt_arrow_scale=float(getattr(args, "viz_tilt_arrow_scale", 0.1)),
            show_tilt_streamlines=getattr(args, "viz_tilt_streamlines", False),
            show_patch_boundaries=getattr(args, "viz_patch_boundaries", False),
            patch_key=str(getattr(args, "viz_patch_key", "disk_patch")),
            show_boundary_loops=getattr(args, "viz_boundary_loops", False),
            annotate_boundary_geodesic=getattr(args, "viz_boundary_geodesic", False),
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
    if args.radius_of_gyration:
        total_rg = mesh.compute_surface_radius_of_gyration()
        print(f"Surface radius of gyration: {total_rg:.6f}")
        if mesh.bodies:
            print()
            print("Per‑body surface radius of gyration:")
            for body_idx, body in mesh.bodies.items():
                body_rg = mesh.compute_surface_radius_of_gyration(body.facet_indices)
                print(f"  Body {body_idx}: surface Rg = {body_rg:.6f}")
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
        execute_command_line(context, line, get_command_fn=get_command)

    if not args.non_interactive:
        _setup_interactive_history()
        _setup_interactive_completion(context)
        while not context.should_exit:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            try:
                execute_command_line(context, line, get_command_fn=get_command)
            except Exception as e:
                logger.error(f"Error executing command '{line}': {e}")

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
