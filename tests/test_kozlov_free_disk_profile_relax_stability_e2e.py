import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.refinement import refine_triangle_mesh  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402
from runtime.vertex_average import vertex_average  # noqa: E402


@pytest.mark.e2e
def test_profile_relax_stability_after_refine():
    """Regression: short relax after refine should not blow up bending_tilt energy."""
    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kozlov_1disk_3d_free_disk_theory_parity.yaml",
    )
    mesh = parse_geometry(load_data(path))
    gp = mesh.global_parameters

    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 1.0e-3)
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 10)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )

    # Coarse relax.
    minim.minimize(n_steps=20)

    # Refine + average, then short relax to ensure stability.
    mesh = refine_triangle_mesh(mesh)
    vertex_average(mesh)

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )
    minim.minimize(n_steps=20)
    breakdown = minim.compute_energy_breakdown()

    bend_in = float(breakdown.get("bending_tilt_in") or 0.0)
    bend_out = float(breakdown.get("bending_tilt_out") or 0.0)

    # Guard against catastrophic blow-ups (order-of-magnitude check).
    assert bend_in < 1.0e3
    assert bend_out < 1.0e3
