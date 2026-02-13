import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402
from tests.test_kozlov_free_disk_theory_parity_e2e import (  # noqa: E402
    _tensionless_thetaB_prediction,
)


@pytest.mark.e2e
def test_kozlov_named_mesh_matches_tensionless_theory_within_15pct() -> None:
    """Acceptance: the named free-disk mesh reproduces 1_disk_3d tensionless theory."""
    mesh_path = (
        Path(__file__).resolve().parent.parent
        / "meshes"
        / "caveolin"
        / "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml"
    )
    mesh = parse_geometry(load_data(str(mesh_path)))
    gp = mesh.global_parameters

    # Keep the YAML's thetaB optimizer knobs; only make relaxation deterministic.
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 5)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)
    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 1e-3)

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )

    kappa_in = float(gp.get("bending_modulus_in") or gp.get("bending_modulus") or 0.0)
    kappa_out = float(gp.get("bending_modulus_out") or gp.get("bending_modulus") or 0.0)
    kappa = float(kappa_in + kappa_out)
    kappa_t_in = float(gp.get("tilt_modulus_in") or 0.0)
    kappa_t_out = float(gp.get("tilt_modulus_out") or 0.0)
    kappa_t = float(kappa_t_in + kappa_t_out)
    drive = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)
    assert kappa > 0.0 and kappa_t > 0.0 and drive != 0.0

    R = 7.0 / 15.0
    theta_star, fin_star, fout_star, fcont_star, ftot_star = (
        _tensionless_thetaB_prediction(kappa=kappa, kappa_t=kappa_t, drive=drive, R=R)
    )
    fel_star = float(fin_star + fout_star)

    # Acceptance workflow: run a small number of reduced-energy outer scans.
    # This should move thetaB toward theory while keeping runtime bounded.
    mode = str(gp.get("tilt_solve_mode") or "coupled")
    theta_hist: list[float] = []
    for i in range(6):
        minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode=mode)
        minim._optimize_thetaB_scalar(tilt_mode=mode, iteration=i)
        theta_hist.append(float(gp.get("tilt_thetaB_value") or 0.0))

    theta_meas = float(theta_hist[-1])
    errors = np.abs(np.asarray(theta_hist, dtype=float) - float(theta_star))
    # Trend check: the final scan should be the closest to theory.
    assert errors[-1] <= float(np.min(errors))

    breakdown = minim.compute_energy_breakdown()
    contact_meas = float(breakdown.get("tilt_thetaB_contact_in") or 0.0)
    fel_meas = float(sum(float(v) for v in breakdown.values()) - contact_meas)
    ftot_meas = float(fel_meas + contact_meas)

    assert theta_meas == pytest.approx(theta_star, rel=0.15, abs=0.02), (
        f"theta_hist={np.array(theta_hist)} theta_star={theta_star:.8f}"
    )
    assert fel_meas == pytest.approx(fel_star, rel=0.15, abs=0.05), (
        f"fel_meas={fel_meas:.8f} fel_star={fel_star:.8f} theta_hist={np.array(theta_hist)}"
    )
    assert ftot_meas == pytest.approx(ftot_star, rel=0.15, abs=0.05), (
        f"ftot_meas={ftot_meas:.8f} ftot_star={ftot_star:.8f} theta_hist={np.array(theta_hist)}"
    )
