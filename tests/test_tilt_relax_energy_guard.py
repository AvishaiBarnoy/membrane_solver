import numpy as np

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def test_tilt_relaxation_energy_guard_rolls_back_tilts():
    data = load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    mesh = parse_geometry(data)
    mesh.global_parameters.set("tilt_relax_energy_guard_factor", 1.1)
    mesh.global_parameters.set("tilt_relax_energy_guard_min", 1e-9)

    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    tilts_in[:] = 0.0
    tilts_out[:] = 0.0
    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)
    baseline_in = mesh.tilts_in_view().copy()
    baseline_out = mesh.tilts_out_view().copy()

    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )
    original_energy = minim.compute_energy

    def fake_relax(*, positions, mode):
        _ = positions, mode
        tin = mesh.tilts_in_view().copy()
        tout = mesh.tilts_out_view().copy()
        tin[:, 0] += 1.0
        tout[:, 0] -= 1.0
        mesh.set_tilts_in_from_array(tin)
        mesh.set_tilts_out_from_array(tout)

    call_state = {"count": 0}

    def fake_energy():
        call_state["count"] += 1
        if call_state["count"] == 1:
            return 1.0
        if call_state["count"] == 2:
            return 5.0
        return original_energy()

    minim._relax_leaflet_tilts = fake_relax
    minim.compute_energy = fake_energy
    minim.minimize(1)

    np.testing.assert_allclose(mesh.tilts_in_view(), baseline_in, rtol=0, atol=1e-10)
    np.testing.assert_allclose(mesh.tilts_out_view(), baseline_out, rtol=0, atol=1e-10)
