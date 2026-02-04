import numpy as np
import pytest

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _outer_band_mask(mesh, *, r_min: float, r_max: float) -> np.ndarray:
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    return (r >= float(r_min)) & (r <= float(r_max))


@pytest.mark.e2e
def test_kozlov_free_disk_outer_leaflet_tilt_becomes_nontrivial() -> None:
    """Outer leaflet should respond (non-zero tilt) in the outer membrane region.

    This is a prerequisite for matching the bilayer theory in docs/tex/1_disk_3d.tex.
    """
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    gp = mesh.global_parameters

    # For this setup, `tilt_out` becomes non-trivial primarily through coupling
    # via the shared shape/curvature (not in a pure tilt-only relaxation on a
    # flat surface). So we must allow a small number of coupled shape+tilt
    # minimization iterations.
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 10)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)

    gp.set("tilt_thetaB_optimize", True)
    gp.set("tilt_thetaB_optimize_every", 1)
    gp.set("tilt_thetaB_optimize_delta", 0.05)
    gp.set("tilt_thetaB_optimize_inner_steps", 5)

    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 1.0e-3)

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )

    minim.minimize(n_steps=3)

    mask = _outer_band_mask(mesh, r_min=1.5, r_max=10.5)
    assert int(np.sum(mask)) > 0

    tin = mesh.tilts_in_view()
    tout = mesh.tilts_out_view()
    norms_in = np.linalg.norm(tin[mask], axis=1)
    norms_out = np.linalg.norm(tout[mask], axis=1)
    med_in = float(np.median(norms_in))
    q95_out = float(np.quantile(norms_out, 0.95))

    # Inner should be clearly non-zero in this driven case.
    assert med_in > 1e-9
    # Outer should be non-trivial once coupling is active. Use a robust tail
    # statistic (not the median) since only part of the outer band responds
    # after a few iterations.
    assert q95_out > 1e-9
