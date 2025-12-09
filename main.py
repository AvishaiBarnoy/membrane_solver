import argparse
import json
import sys
import os
from logging_config import setup_logging
from geometry.geom_io import load_data, save_geometry, parse_geometry
from geometry.entities import Mesh
from runtime.minimizer import Minimizer
from parameters.resolver import ParameterResolver
from runtime.steppers.gradient_descent import GradientDescent
from runtime.steppers.conjugate_gradient import ConjugateGradient
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
from runtime.refinement import refine_triangle_mesh, refine_polygonal_facets
from runtime.vertex_average import vertex_average
from runtime.equiangulation import equiangulate_mesh
from visualize_geometry import plot_geometry

logger = None


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
        elif cmd == 'r':
            result.append('r')
            # TODO 1: accept "r #n" for multiple refining.
        elif cmd == 'cg':
            result.append('cg')
        elif cmd == 'gd':
            result.append('gd')
        elif cmd in {'h', 'help', '?'}:
            result.append('help')
        elif cmd == "visualize" or cmd == "s":
            result.append(cmd)
        elif cmd.startswith("V") or cmd == "vertex_average":
            # TODO: expand to be target specific vertices "vertex_average [2,5]" 
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
    elif cmd.startswith('t'):
        new_ts = cmd.replace(' ', '')
        try:
            minimizer.step_size = float(new_ts[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid step size format {new_ts[1:]}") from exc
        logger.info(f"Updated step size to {minimizer.step_size}")
    elif cmd == 'r':
        logger.info("Refining mesh...")
        mesh = refine_triangle_mesh(mesh)
        mesh = refine_polygonal_facets(mesh)
        minimizer.mesh = mesh
        logger.info("Mesh refinement complete.")
    elif cmd.startswith('V'):
        if cmd != 'V':
            cmd = ''.join(cmd.split())
            for i in range(1, int(cmd[1:]) + 1):
                vertex_average(mesh)
                logger.info("Vertex averaging done.")
        elif cmd == "V":
            vertex_average(mesh)
            logger.info("Vertex averaging done.")
    elif cmd == 'vertex_average':
        vertex_average(mesh)
        logger.info("Vertex averaging done.")
    elif cmd == 'u':
        logger.info("Starting equiangulation...")
        mesh = equiangulate_mesh(mesh)
        minimizer.mesh = mesh
        logger.info("Equiangulation complete.")
    elif cmd == 'visualize' or cmd == "s":
        plot_geometry(mesh, show_indices=False)
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

    fixed_count = sum(1 for v in mesh.vertices.values() if getattr(v, 'fixed', False))
    logger.debug(f"Number of fixed vertices: {fixed_count} / {len(mesh.vertices)}")
    logger.debug(f"Target volume of body: {mesh.bodies[0].options['target_volume']}")

    global_params = mesh.global_parameters
    param_resolver = ParameterResolver(global_params)
    energy_manager = EnergyModuleManager(mesh.energy_modules)

    logger.debug(mesh.energy_modules)

    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
    # Use Conjugate Gradient as the default stepper for faster convergence.
    stepper = ConjugateGradient()

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
