import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.refinement import refine_triangle_mesh
from runtime.steppers.gradient_descent import GradientDescent


def _relax_energy(mesh, *, inner_steps: int) -> float:
    mesh.global_parameters.update(
        {
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.05,
            "tilt_inner_steps": inner_steps,
            "tilt_tol": 1e-12,
        }
    )
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    minim._relax_tilts(positions=mesh.positions_view(), mode="nested")
    return float(minim.compute_energy())


def test_tilt_decay_energy_decreases_under_refinement() -> None:
    """Convergence: as we refine, the relaxed tilt energy should decrease."""
    mesh = parse_geometry(
        load_data("meshes/tilt_benchmarks/tilt_source_rect_single.yaml")
    )

    e0 = _relax_energy(mesh, inner_steps=800)
    mesh = refine_triangle_mesh(mesh)
    e1 = _relax_energy(mesh, inner_steps=800)
    mesh = refine_triangle_mesh(mesh)
    e2 = _relax_energy(mesh, inner_steps=800)

    assert e0 > e1 > e2
    # 1D screened-Laplace estimate is ~0.5 for this geometry (k_s=k_t=1, W=4, H=1).
    # Our discrete 2D setup should trend toward O(0.5) as refinement increases.
    assert 0.35 < e2 < 0.70
