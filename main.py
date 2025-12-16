import argparse
import json
import logging
import os
import sys

from geometry.entities import Mesh
from geometry.geom_io import load_data, parse_geometry, save_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.equiangulation import equiangulate_mesh
from runtime.logging_config import setup_logging
from runtime.minimizer import Minimizer
from runtime.refinement import refine_polygonal_facets, refine_triangle_mesh
from runtime.steppers.conjugate_gradient import ConjugateGradient
from runtime.steppers.gradient_descent import GradientDescent
from runtime.topology import detect_vertex_edge_collisions
from runtime.vertex_average import vertex_average

logger = logging.getLogger("membrane_solver.log")
logger.addHandler(logging.NullHandler())


def print_physical_properties(mesh: Mesh) -> None:
    """Print basic physical properties of the mesh.

    Includes global surface area and volume, as well as per‑body volumes
    and surface areas when bodies are present.
    """
    total_area = mesh.compute_total_surface_area()
    total_volume = mesh.compute_total_volume()

    print("=== Physical Properties ===")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Edges   : {len(mesh.edges)}")
    print(f"Facets  : {len(mesh.facets)}")
    print(f"Bodies  : {len(mesh.bodies)}")
    print()
    print(f"Total surface area: {total_area:.6f}")
    print(f"Total volume      : {total_volume:.6f}")

    if mesh.bodies:
        print()
        print("Per‑body properties:")
        for body_idx, body in mesh.bodies.items():
            body_vol = body.compute_volume(mesh)
            body_area = body.compute_surface_area(mesh)
            print(
                f"  Body {body_idx}: volume = {body_vol:.6f}, "
                f"surface area = {body_area:.6f}"
            )


def resolve_json_path(path: str) -> str:
    """Return a valid JSON file path, allowing path without extension."""
    if os.path.isfile(path):
        return path
    if not path.lower().endswith('.json'):
        alt = path + '.json'
        if os.path.isfile(alt):
            return alt
    raise FileNotFoundError(f"Cannot find file '{path}' or '{path}.json'")

def load_mesh_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    # You need to implement this function to construct a Mesh from JSON
    return Mesh.from_json(data)

def save_mesh_to_json(mesh, path):
    with open(path, 'w') as f:
        json.dump(mesh.to_json(), f, indent=2)

def parse_instructions(instr):
    # Accepts a string or list of instructions, returns a list of (cmd, arg) tuples
    result = []
    if isinstance(instr, str):
        instr = instr.split()

    for inst in instr:
        cmd = inst
        if cmd.startswith('g'):
            result.append(cmd)
        elif cmd.startswith('r'):
            result.append(cmd)
        elif cmd == 'cg':
            result.append('cg')
        elif cmd == 'gd':
            result.append('gd')
        elif cmd in {'h', 'help', '?'}:
            result.append('help')
        elif cmd in {'properties', 'props', 'p', 'i'}:
            result.append('properties')
        elif cmd == "visualize" or cmd == "s":
            result.append(cmd)
        elif cmd.startswith("V") or cmd == "vertex_average":
            result.append(cmd)
        elif cmd == 'u':
            result.append('u')
        elif cmd.startswith('t'):
            result.append(cmd)
        elif cmd == "save":
            result.append(cmd)
        else:
            logger.warning(f"Unknown instruction: {cmd}")
    return result

def execute_command(cmd, mesh, minimizer, stepper):
    """Handle a single simulation command."""

    if cmd == 'help':
        # Print interactive command help and CLI options
        print("Interactive commands:")
        print("  gN            Run N minimization steps (e.g. g5, g10)")
        print("  gd / cg       Switch to Gradient Descent / Conjugate Gradient stepper")
        print("  tX            Set step size to X (e.g. t1e-3)")
        print("  r             Refine mesh (triangle refinement + polygonal)")
        print("  V / Vn        Vertex averaging once or n times (e.g. V5)")
        print("  vertex_average Same as V")
        print("  u             Equiangulate mesh")
        print("  visualize / s Plot current geometry")
        print("  properties    Print physical properties (area, volume, etc.)")
        print("  save          Save geometry to 'interactive.temp'")
        print("  quit / exit / q  Leave interactive mode")
        print()
        print("Command-line options (when starting the solver):")
        print("  -i, --input PATH          Input mesh JSON file")
        print("  -o, --output PATH         Output mesh JSON file (default: output.json)")
        print("  --instructions PATH       Instruction file (one command per line)")
        print("  --log PATH                Log file path (default: membrane_solver.log)")
        print("  -q, --quiet               Suppress console output")
        print("  --debug                   Enable verbose debug logging")
        print("  --non-interactive         Do not enter interactive prompt after instructions")
    elif cmd == 'cg':
        logger.info("Switching to Conjugate Gradient stepper.")
        stepper = ConjugateGradient()
        minimizer.stepper = stepper
    elif cmd == 'gd':
        logger.info("Switching to Gradient Descent stepper.")
        stepper = GradientDescent()
        minimizer.stepper = stepper
    elif cmd.startswith('g'):
        cmd = cmd.replace(' ', '')
        # Accept bare 'g' as one step; otherwise require integer suffix.
        if cmd == 'g':
            n_steps = 1
        else:
            steps_str = cmd[1:]
            if not steps_str.isnumeric():
                logger.warning(
                    "Invalid minimization command '%s'; expected 'g' or 'gN' with integer N.",
                    cmd,
                )
                return mesh, stepper
            n_steps = int(steps_str)

        logger.debug(
            f"Minimizing for {n_steps} steps using {stepper.__class__.__name__}"
        )
        logger.debug(
            f"Step size: {minimizer.step_size}, Tolerance: {minimizer.tol}"
        )
        result = minimizer.minimize(n_steps=n_steps)
        mesh = result["mesh"]
        logger.info(
            f"Minimization complete. Final energy: {result['energy'] if result else 'N/A'}"
        )
        # NEW: Check for collisions after minimization
        collisions = detect_vertex_edge_collisions(mesh)
        if collisions:
            logger.warning(f"TOPOLOGY WARNING: {len(collisions)} vertex-edge collisions detected!")
            # Optional: Visualize them or auto-correct (future work)

    elif cmd.startswith('t'):
        new_ts = cmd.replace(' ', '')
        try:
            minimizer.step_size = float(new_ts[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid step size format {new_ts[1:]}") from exc
        logger.info(f"Updated step size to {minimizer.step_size}")
    elif cmd.startswith('r'):
        count = 1
        arg = cmd[1:]
        if arg:
            if arg.isdigit():
                count = int(arg)
            else:
                logger.warning("Invalid refine command '%s'; expected 'r' or 'rN'.", cmd)
                return mesh, stepper
        for i in range(count):
            logger.info("Refining mesh... (%d/%d)", i + 1, count)
            mesh = refine_polygonal_facets(mesh)
            mesh = refine_triangle_mesh(mesh)
            minimizer.mesh = mesh
            minimizer.enforce_constraints_after_mesh_ops(mesh)
        logger.info("Mesh refinement complete after %d pass(es).", count)
    elif cmd.startswith('V'):
        # Vertex averaging: V for a single pass, or VN for N passes.
        cmd = cmd.replace(" ", "")
        if cmd == "V":
            n_passes = 1
        else:
            passes_str = cmd[1:]
            if not passes_str.isnumeric():
                logger.warning(
                    "Invalid vertex averaging command '%s'; expected 'V' or 'VN' "
                    "with integer N.",
                    cmd,
                )
                return mesh, stepper
            n_passes = int(passes_str)

        for _ in range(n_passes):
            vertex_average(mesh)
        logger.info("Vertex averaging done.")
        # After vertex averaging, explicitly re‑enforce hard constraints such
        # as fixed volume so subsequent minimization starts from a consistent
        # state, even if averaging introduced small drifts.
        minimizer.enforce_constraints_after_mesh_ops(mesh)
    elif cmd == 'vertex_average':
        vertex_average(mesh)
        logger.info("Vertex averaging done.")
    elif cmd == 'u':
        logger.info("Starting equiangulation...")
        mesh = equiangulate_mesh(mesh)
        minimizer.mesh = mesh
        minimizer.enforce_constraints_after_mesh_ops(mesh)
        logger.info("Equiangulation complete.")
    elif cmd == 'properties':
        print_physical_properties(mesh)
    elif cmd == 'visualize' or cmd == "s":
        #from visualize_geometry import plot_geometry
        from visualization.plotting import plot_geometry
        plot_geometry(mesh, show_indices=False, )
    elif cmd == 'save':
        # fall back to a default name
        save_geometry(mesh, 'interactive.temp')
        logger.info("Saved geometry to interactive.temp")
    else:
        logger.warning(f"Unknown instruction: {cmd}")

    return mesh, stepper

def interactive_loop(mesh, minimizer, stepper):
    """Run an interactive command loop."""

    while True:
        try:
            line = input('> ').strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in {'quit', 'exit', 'q'}:
            break
        commands = parse_instructions(''.join(line.split()))
        for cmd in commands:
            mesh, stepper = execute_command(cmd, mesh, minimizer, stepper)

    return mesh

def main():
    parser = argparse.ArgumentParser(description="Membrane Solver Simulation Driver")
    parser.add_argument('-i', '--input', help='Input mesh JSON file')
    parser.add_argument('-o', '--output', default=None, help='Output mesh JSON file')
    parser.add_argument('--instructions', help='Optional instruction file (one command per line)')
    parser.add_argument('--log', default=None, help='Optional log file')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress console output')
    parser.add_argument('--debug', action='store_true',
                        help='Enable verbose debug logging')
    parser.add_argument('--non-interactive', action='store_true',
                        help="Skip interactive mode after executing instructions")
    parser.add_argument(
        '--properties',
        action='store_true',
        help='Print basic physical properties (volume, surface area, etc.) and exit',
    )
    parser.add_argument(
        '--volume-mode',
        choices=['lagrange', 'penalty'],
        default=None,
        help='Override volume constraint mode (lagrange = hard constraint; penalty = soft energy).',
    )
    parser.add_argument(
        '--line-tension',
        type=float,
        default=None,
        help='Assign a line-tension modulus to edges. When combined with '
        '`--line-tension-edges`, only the specified edge IDs are tagged. '
        'Otherwise all edges receive line tension.',
    )
    parser.add_argument(
        '--line-tension-edges',
        type=str,
        default=None,
        help='Comma-separated edge IDs to receive CLI line tension.',
    )
    args = parser.parse_args()

    if not args.input:
        try:
            args.input = input('Input mesh JSON file: ').strip()
        except EOFError:
            print('No input file provided.', file=sys.stderr)
            sys.exit(1)
    try:
        args.input = resolve_json_path(args.input)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    global logger
    logger = setup_logging(
        args.log if args.log else 'membrane_solver.log',
        quiet=args.quiet,
        debug=args.debug,
    )

    # Load mesh and parameters
    data = load_data(args.input)
    mesh = parse_geometry(data)

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

    fixed_count = sum(1 for v in mesh.vertices.values() if getattr(v, 'fixed', False))
    logger.debug(f"Number of fixed vertices: {fixed_count} / {len(mesh.vertices)}")
    # Log target volume information only when a body and target are defined.
    if mesh.bodies:
        body0 = mesh.bodies.get(0)
        if body0 is not None:
            target_vol = body0.options.get("target_volume")
            logger.debug(f"Body 0 target_volume: {target_vol}")

    global_params = mesh.global_parameters
    if args.volume_mode:
        global_params.set("volume_constraint_mode", args.volume_mode)
    energy_manager = EnergyModuleManager(mesh.energy_modules)

    logger.debug(mesh.energy_modules)

    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
    # Use Conjugate Gradient as the default stepper for faster convergence.
    stepper = ConjugateGradient()

    # Optional: report physical properties and exit early if requested.
    if args.properties:
        print_physical_properties(mesh)
        return

    # Load instructions
    if args.instructions:
        with open(args.instructions, 'r') as f:
            instr = f.read().split()
    else:
        instr = mesh.instructions if hasattr(mesh, 'instructions') else []
    instructions = parse_instructions(instr)
    logger.debug(f"Instructions to execute: {instructions}")

    if not args.quiet:
        print("=== Membrane Solver ===")
        print(f"Input file: {args.input}")
        print(f"Output file: {args.output or '(not saving)'}")
        print(f"Energy modules: {mesh.energy_modules}")
        print(f"Constraint modules: {mesh.constraint_modules}")
        print(f"Instructions: {instructions}")

    minimizer = Minimizer(mesh, global_params, stepper, energy_manager,
                          constraint_manager, quiet=args.quiet)
    logger.debug(global_params)

    minimizer.step_size = global_params.get("step_size", 0.001)

    # Simulation loop
    for cmd in instructions:
        mesh, stepper = execute_command(cmd, mesh, minimizer, stepper)

    if not args.non_interactive:
        mesh = interactive_loop(mesh, minimizer, stepper)
    # Save final mesh
    #save_mesh_to_json(mesh, args.output)
    if args.output:
        save_geometry(mesh, args.output)
        logger.info(f"Simulation complete. Output saved to {args.output}")
    else:
        logger.info("Simulation complete. No output file written.")


if __name__ == "__main__":
    main()
