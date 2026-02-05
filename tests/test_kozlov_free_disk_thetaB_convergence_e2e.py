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
from tests.test_kozlov_free_disk_theory_parity_e2e import (  # noqa: E402
    _tensionless_thetaB_prediction,
)


def _run_thetaB_history(*, mesh, n_steps: int) -> list[float]:
    gp = mesh.global_parameters

    # Keep this diagnostic deterministic and reasonably fast.
    # Use the fixture's theory-mode parameters by default; we only pin the
    # shape step size to keep runtime bounded across platforms.
    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 1.0e-3)

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-8,
    )

    hist: list[float] = []

    def cb(_mesh, _i):
        hist.append(float(gp.get("tilt_thetaB_value") or 0.0))

    minim.minimize(n_steps=n_steps, callback=cb)
    hist.append(float(gp.get("tilt_thetaB_value") or 0.0))
    return hist


@pytest.mark.e2e
def test_kozlov_free_disk_thetaB_converges_toward_tex_prediction() -> None:
    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kozlov_1disk_3d_free_disk_theory_parity.yaml",
    )
    mesh0 = parse_geometry(load_data(path))

    gp0 = mesh0.global_parameters
    kappa_in = float(gp0.get("bending_modulus_in") or gp0.get("bending_modulus") or 0.0)
    kappa_out = float(
        gp0.get("bending_modulus_out") or gp0.get("bending_modulus") or 0.0
    )
    kappa = float(kappa_in + kappa_out)
    kappa_t_in = float(gp0.get("tilt_modulus_in") or 0.0)
    kappa_t_out = float(gp0.get("tilt_modulus_out") or 0.0)
    kappa_t = float(kappa_t_in + kappa_t_out)
    drive = float(gp0.get("tilt_thetaB_contact_strength_in") or 0.0)

    R = 7.0 / 15.0
    theta_star, *_ = _tensionless_thetaB_prediction(
        kappa=kappa, kappa_t=kappa_t, drive=drive, R=R
    )
    assert theta_star > 0.0

    hist0 = _run_thetaB_history(mesh=mesh0, n_steps=12)
    theta_final0 = float(hist0[-1])

    # Convergence toward the predicted value (not necessarily monotone).
    # Use the same broad tolerance as the existing parity test, but make the
    # failure message include the whole thetaB history for diagnosis.
    assert theta_final0 == pytest.approx(theta_star, rel=0.40, abs=0.03), (
        f"thetaB*={theta_star:.6g}, hist={np.array(hist0)}"
    )

    # NOTE: We intentionally do *not* assert refinement stability yet.
    #
    # In the current codebase, the remaining elastic-parity gap (tracked by an
    # xfail in `tests/test_kozlov_free_disk_theory_parity_e2e.py`) can shift the
    # reduced-energy optimum for thetaB under refinement. Once that physics gap
    # is closed, we should tighten this diagnostic by adding a refinement-level
    # stability assertion here.


@pytest.mark.e2e
def test_kozlov_free_disk_thetaB_updates_are_bounded_by_scan_delta() -> None:
    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kozlov_1disk_3d_free_disk_theory_parity.yaml",
    )
    mesh = parse_geometry(load_data(path))
    gp = mesh.global_parameters

    n_steps = 25
    hist = _run_thetaB_history(mesh=mesh, n_steps=n_steps)
    deltas = np.abs(np.diff(np.asarray(hist, dtype=float)))

    # The thetaB scalar scan only considers base +/- delta (or base), so
    # consecutive updates should never exceed delta. This helps prevent
    # ping-ponging between distant thetaB values as the shape evolves.
    scan_delta = float(gp.get("tilt_thetaB_optimize_delta") or 0.0)
    assert scan_delta > 0.0
    assert float(np.max(deltas) if deltas.size else 0.0) <= scan_delta + 1e-12, (
        f"delta={scan_delta}, max_step={float(np.max(deltas) if deltas.size else 0.0)}, "
        f"hist={np.array(hist)}"
    )
