"""Run an interactive minimization visualized with PyVista."""

from __future__ import annotations

import argparse

from geometry.geom_io import load_data, parse_geometry
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
from runtime.interactive_visualizer import visualize_minimization


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive minimization viewer")
    parser.add_argument("input", help="Input mesh JSON file")
    parser.add_argument("-n", "--steps", type=int, default=50, help="Number of minimization steps")
    args = parser.parse_args()

    data = load_data(args.input)
    mesh = parse_geometry(data)

    global_params = mesh.global_parameters
    energy_manager = EnergyModuleManager(mesh.energy_modules)
    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
    stepper = GradientDescent()

    minimizer = Minimizer(mesh, global_params, stepper, energy_manager, constraint_manager)

    visualize_minimization(mesh, minimizer, n_steps=args.steps)


if __name__ == "__main__":
    main()
