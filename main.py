import argparse
import json
import sys
import os
from logging_config import setup_logging
from geometry.geom_io import load_data, save_geometry, parse_geometry
from geometry.entities import Mesh
from runtime.minimizer import Minimizer
from parameters.resolver import ParameterResolver
from modules.steppers.gradient_descent import GradientDescent
from modules.steppers.conjugate_gradient import ConjugateGradient
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
from runtime.refinement import refine_triangle_mesh
from visualize_geometry import plot_geometry

logger = None

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
        elif cmd == 'cg':
            result.append('cg')
        elif cmd == 'gd':
            result.append('gd')
        elif cmd == "visualize":
            result.append('visualize')
        elif cmd.startswith('t'):
            result.append(cmd)
        elif cmd == "save":
            result.append(cmd)
        else:
            logger.warning(f"Unknown instruction: {cmd}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Membrane Solver Simulation Driver")
    parser.add_argument('-i', '--input', required=True, help='Input mesh JSON file')
    parser.add_argument('-o', '--output', required=True, help='Output mesh JSON file')
    parser.add_argument('--instructions', help='Optional instruction file (one command per line)')
    parser.add_argument('--log', default=None, help='Optional log file')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress console output')
    args = parser.parse_args()

    global logger
    logger = setup_logging(args.log if args.log else 'membrane_solver.log',
                           quiet=args.quiet)

    # Load mesh and parameters
    data = load_data(args.input)
    mesh = parse_geometry(data)
    #print(f"[DEBUG] Loaded bodies:\n {mesh.bodies}")

    fixed_count = sum(1 for v in mesh.vertices.values() if getattr(v, 'fixed', False))
    logger.debug(f"Number of fixed vertices: {fixed_count} / {len(mesh.vertices)}")
    logger.debug(f"Target volume of body: {mesh.bodies[0].options['target_volume']}")

    global_params = mesh.global_parameters
    param_resolver = ParameterResolver(global_params)
    energy_manager = EnergyModuleManager(mesh.energy_modules)

    logger.debug("###########")
    logger.debug(mesh.energy_modules)
    logger.debug("###########")

    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
    stepper = GradientDescent()

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
        print(f"Output file: {args.output}")
        print(f"Energy modules: {mesh.energy_modules}")
        print(f"Constraint modules: {mesh.constraint_modules}")
        print(f"Instructions: {instructions}")

    minimizer = Minimizer(mesh, global_params, stepper, energy_manager,
                          constraint_manager, quiet=args.quiet)
    logger.debug(global_params)

    minimizer.step_size = global_params.get("step_size", 0.001)

    # Simulation loop
    for cmd in instructions:
        if cmd == 'cg':
            logger.info("Switching to Conjugate Gradient stepper.")
            stepper = ConjugateGradient()
            minimizer.stepper = stepper
        elif cmd == 'gd':
            logger.info("Switching to Gradient Descent stepper.")
            stepper = GradientDescent()
            minimizer.stepper = stepper
        elif cmd.startswith('g'):
            cmd = cmd.replace(" ", "")  # remove whitespaces
            if cmd == "g":
                cmd = "g1"
            assert cmd[1:].isnumeric(), "#n steps should be in the form of 'g 5' or 'g5'"
            logger.debug(minimizer.step_size)

            logger.info(
                f"Minimizing for {cmd[1:]} steps using {stepper.__class__.__name__}"
            )
            n_steps = int(cmd[1:])

            logger.debug(f"Step size: {minimizer.step_size}, Tolerance: {minimizer.tol}")
            result = minimizer.minimize(n_steps=n_steps)
            mesh = result["mesh"]
            logger.info(f"Minimization complete. Final energy: {result['energy'] if result else 'N/A'}")
        elif cmd.startswith('t'):
            new_ts = cmd.replace(' ', '')
            try:
                minimizer.step_size = float(new_ts[1:])
            except ValueError:
                raise ValueError(f"Invalid step size format {new_ts[1:]}")
            logger.info(f"Updated step size to {minimizer.step_size}")
        elif cmd == 'r':
            logger.info("Refining mesh...")
            mesh = refine_triangle_mesh(mesh)
            minimizer.mesh = mesh
            logger.info("Mesh refinement complete.")
        elif cmd == "visualize":
            plot_geometry(mesh, show_indices=False)
        elif cmd == "save":
            save_geometry(mesh, args.input + ".temp")
            logger.info(f"Saved geometry to {args.input}.temp")
        else:
            logger.warning(f"Unknown instruction: {cmd}")

    # Save final mesh
    #save_mesh_to_json(mesh, args.output)
    save_geometry(mesh, args.output)
    logger.info(f"Simulation complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()
