import time

from commands.context import CommandContext
from commands.executor import execute_command_line
from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def run_profile_relax_light(config_path: str) -> float:
    data = load_data(config_path)
    mesh = parse_geometry(data)
    energy_manager = EnergyModuleManager(mesh.energy_modules)
    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
    stepper = GradientDescent()
    minimizer = Minimizer(
        mesh,
        mesh.global_parameters,
        stepper,
        energy_manager,
        constraint_manager,
        quiet=True,
    )
    minimizer.step_size = mesh.global_parameters.get("step_size", 0.001)
    context = CommandContext(mesh, minimizer, stepper)

    start = time.perf_counter()
    execute_command_line(context, "profile_relax_light")
    return time.perf_counter() - start


if __name__ == "__main__":
    config = "meshes/caveolin/kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml"
    elapsed = run_profile_relax_light(config)
    print(f"profile_relax_light elapsed: {elapsed:.6f}s")
