import argparse
import json
import sys
import os
from logging_config import setup_logging
from geometry.geom_io import load_data, save_geometry, parse_geometry
from geometry.entities import Mesh
from modules.minimizer import Minimizer, ParameterResolver
from modules.steppers.gradient_descent import GradientDescent
from modules.steppers.conjugate_gradient import ConjugateGradient
from runtime.energy_manager import EnergyModuleManager
from runtime.refinement import refine_triangle_mesh
from visualize_geometry import plot_geometry

logger = setup_logging('membrane_solver')

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
            #plot_geometry(mesh, show_indices=False)
        else:
            logger.warning(f"Unknown instruction: {cmd}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Membrane Solver Simulation Driver")
    parser.add_argument('-i', '--input', required=True, help='Input mesh JSON file')
    parser.add_argument('-o', '--output', required=True, help='Output mesh JSON file')
    parser.add_argument('--instructions', help='Optional instruction file (one command per line)')
    parser.add_argument('--log', default=None, help='Optional log file')
    args = parser.parse_args()

    if args.log:
        logger.addHandler(setup_logging(args.log))

    # Load mesh and parameters
    data = load_data(args.input)
    mesh = parse_geometry(data)
    print(f"[DEBUG] Loaded bodies:\n {mesh.bodies}")

    fixed_count = sum(1 for v in mesh.vertices.values() if getattr(v, 'fixed', False))
    print(f"[DEBUG] Number of fixed vertices: {fixed_count} / {len(mesh.vertices)}")
    print(f"[DEBUG] Target volume of body: {mesh.bodies[0].options['target_volume']}")

    global_params = mesh.global_parameters
    param_resolver = ParameterResolver(global_params)
    energy_manager = EnergyModuleManager(mesh.energy_modules)
    stepper = GradientDescent()

    # Load instructions
    if args.instructions:
        with open(args.instructions, 'r') as f:
            instr = f.read().split()
    else:
        instr = mesh.instructions if hasattr(mesh, 'instructions') else []
    instructions = parse_instructions(instr)
    print(f"[DEBUG] Instructions to execute: {instructions}")

    # Simulation loop
    for cmd in instructions:
        if cmd.startswith('g'):
            cmd = cmd.replace(" ", "")  # remove whitespaces
            if cmd == "g": cmd = "g1"
            assert cmd[1:].isnumeric(), "#n steps should be in the form of 'g 5' or 'g5'"
            logger.info(f"Minimizing for {cmd[1:]} steps using {stepper.__class__.__name__}")
            minimizer = Minimizer(mesh, global_params, stepper, energy_manager)
            minimizer.max_iter = int(cmd[1:])
            minimizer.step_size = global_params.get("step_size", 1e-4)

            print(f"[DEBUG] Step size: {minimizer.step_size}, Tolerance: {minimizer.tol}")
            result = minimizer.minimize()
            mesh = result["mesh"]
            logger.info(f"Minimization complete. Final energy: {result['energy'] if result else 'N/A'}")
        elif cmd == 'r':
            logger.info("Refining mesh...")
            new_mesh = refine_triangle_mesh(mesh)
        elif cmd == 'cg':
            logger.info("Switching to Conjugate Gradient stepper.")
            stepper = ConjugateGradient()
        elif cmd == 'gd':
            logger.info("Switching to Gradient Descent stepper.")
            stepper = GradientDescent()
        elif cmd == "visualize":
            plot_geometry(mesh, show_indices=False)
        else:
            logger.warning(f"Unknown instruction: {cmd}")

    # Save final mesh
    #save_mesh_to_json(mesh, args.output)
    save_geometry(mesh, args.output)
    logger.info(f"Simulation complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()
