from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure the project root is in sys.path when running via benchmarks/suite.py.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent

BASE_INPUT = Path(__file__).resolve().parent / "inputs" / "bench_bending_analytic.json"


def benchmark():
    mesh = parse_geometry(load_data(BASE_INPUT))
    gp = mesh.global_parameters

    em = EnergyModuleManager(["bending"])
    cm = ConstraintModuleManager(["volume"])
    stepper = GradientDescent()

    minimizer = Minimizer(mesh, gp, stepper, em, cm, quiet=True)

    n_steps = 20
    start_time = time.perf_counter()
    result = minimizer.minimize(n_steps=n_steps)
    end_time = time.perf_counter()

    iterations = int(result.get("iterations", n_steps))
    iterations = max(iterations, 1)
    avg_time = (end_time - start_time) / iterations
    return avg_time


if __name__ == "__main__":
    t = benchmark()
    print(f"Average step time (Bending): {t:.4f}s")
