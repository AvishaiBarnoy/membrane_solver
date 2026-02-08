import os
import sys

sys.path.insert(0, os.getcwd())

from geometry.geom_io import load_data, parse_geometry, save_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent
from tests.test_kozlov_free_disk_theory_parity_e2e import _tensionless_thetaB_prediction
from tools.diagnostics.free_disk_energy_split import compute_energy_split


def _relax_tilts_and_thetaB(mesh, steps: int = 4) -> None:
    gp = mesh.global_parameters
    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )
    tilt_mode = str(gp.get("tilt_solve_mode") or "coupled")
    for i in range(steps):
        minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode=tilt_mode)
        minim._optimize_thetaB_scalar(tilt_mode=tilt_mode, iteration=i)


def test_free_membrane_energy_within_order_of_magnitude(tmp_path):
    base = load_data("tests/fixtures/kozlov_free_disk_coarse_refinable.yaml")
    mesh = parse_geometry(base)

    _relax_tilts_and_thetaB(mesh, steps=4)

    out_path = tmp_path / "out.yaml"
    save_geometry(mesh, str(out_path), compact=True)
    out = load_data(out_path)

    split = compute_energy_split(base, out)

    gp = base.get("global_parameters") or {}
    kappa_in = float(gp.get("bending_modulus_in") or gp.get("bending_modulus") or 0.0)
    kappa_out = float(gp.get("bending_modulus_out") or gp.get("bending_modulus") or 0.0)
    kappa = float(kappa_in + kappa_out)
    kappa_t_in = float(gp.get("tilt_modulus_in") or 0.0)
    kappa_t_out = float(gp.get("tilt_modulus_out") or 0.0)
    kappa_t = float(kappa_t_in + kappa_t_out)
    drive = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)

    theta_star, fin, fout, _, _ = _tensionless_thetaB_prediction(
        kappa=kappa, kappa_t=kappa_t, drive=drive, R=7.0 / 15.0
    )
    assert theta_star > 0.0

    predicted_free = fin + fout
    assert predicted_free > 0.0

    free_membrane_energy = split["outer_total"]
    assert abs(free_membrane_energy) <= 10.0 * predicted_free
