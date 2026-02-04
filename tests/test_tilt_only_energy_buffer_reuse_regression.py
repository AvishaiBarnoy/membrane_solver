import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


@pytest.mark.regression
def test_tilt_only_energy_buffer_reuse_is_deterministic() -> None:
    """Guard against stale-buffer reuse affecting energy results."""
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    gp = mesh.global_parameters
    gp.set("tilt_thetaB_optimize", False)

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-8,
    )

    positions = mesh.positions_view()
    rng = np.random.default_rng(1)
    tin = 1e-2 * rng.standard_normal(size=mesh.tilts_in_view().shape)
    tout = 1e-2 * rng.standard_normal(size=mesh.tilts_out_view().shape)

    grad_dummy = np.ones_like(positions) * 123.0
    e1 = minim._compute_tilt_dependent_energy_with_leaflet_tilts(
        positions=positions,
        tilts_in=tin,
        tilts_out=tout,
        grad_dummy=grad_dummy,
    )
    # Overwrite buffer with junk and ensure the result remains the same.
    grad_dummy[:] = -999.0
    e2 = minim._compute_tilt_dependent_energy_with_leaflet_tilts(
        positions=positions,
        tilts_in=tin,
        tilts_out=tout,
        grad_dummy=grad_dummy,
    )

    assert e1 == pytest.approx(e2, rel=0.0, abs=0.0)
