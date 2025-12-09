#!/usr/bin/env python3
"""Run cube_good_min_routine.json with a fixed Evolver-like sequence.

This script is a small driver that:
  - Loads meshes/cube_good_min_routine.json
  - Uses Conjugate Gradient as the stepper
  - Executes the instruction sequence from that file
  - Prints final energy, area, and volume
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure we import the project-local main.py, not any site-packages main.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.conjugate_gradient import ConjugateGradient
from logging_config import setup_logging
import main as main_mod


def run_cube_good():
    root = Path(__file__).resolve().parent.parent
    input_path = root / "meshes" / "cube_good_min_routine.json"

    data = load_data(str(input_path))
    mesh = parse_geometry(data)

    # Initialize logging similarly to main so execute_command can use the logger.
    if main_mod.logger is None:
        main_mod.logger = setup_logging("membrane_solver.log", quiet=False, debug=False)

    global_params = mesh.global_parameters
    energy_manager = EnergyModuleManager(mesh.energy_modules)
    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
    stepper = ConjugateGradient()

    minimizer = Minimizer(
        mesh=mesh,
        global_params=global_params,
        stepper=stepper,
        energy_manager=energy_manager,
        constraint_manager=constraint_manager,
        quiet=False,
    )
    minimizer.step_size = global_params.get("step_size", 1e-3)

    # Use instructions from the JSON file to mimic the Evolver run.
    instr = mesh.instructions if hasattr(mesh, "instructions") else []
    instructions = main_mod.parse_instructions(instr)
    print(f"Instructions: {instructions}")

    for cmd in instructions:
        mesh, stepper = main_mod.execute_command(cmd, mesh, minimizer, stepper)

    # After executing the sequence, report final diagnostics.
    final_energy = minimizer.compute_energy()
    final_area = mesh.compute_total_surface_area()
    final_volume = mesh.compute_total_volume()

    print("\n=== Final diagnostics ===")
    print(f"Energy:  {final_energy:.6f}")
    print(f"Area:    {final_area:.6f}")
    print(f"Volume:  {final_volume:.6f}")


if __name__ == "__main__":
    run_cube_good()
