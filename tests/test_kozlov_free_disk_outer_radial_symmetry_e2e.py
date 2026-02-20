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


def _outer_band_mask(mesh, *, r_min: float, r_max: float) -> np.ndarray:
    pos = mesh.positions_view()
    r = np.linalg.norm(pos[:, :2], axis=1)
    return (r >= float(r_min)) & (r <= float(r_max))


def _outer_radial_symmetry_metric(
    mesh, *, r_min: float, r_max: float
) -> tuple[float, float, float]:
    """Return (m_same, m_oppo, m_ref) for radial tilt symmetry in outer band."""
    mask = _outer_band_mask(mesh, r_min=r_min, r_max=r_max)
    if not np.any(mask):
        raise AssertionError("Outer band is empty.")

    pos = mesh.positions_view()
    r = np.linalg.norm(pos[:, :2], axis=1)
    ers = np.zeros_like(pos)
    nz = r > 1e-12
    ers[nz, 0] = pos[nz, 0] / r[nz]
    ers[nz, 1] = pos[nz, 1] / r[nz]

    tin = mesh.tilts_in_view()
    tout = mesh.tilts_out_view()
    tin_r = np.sum(tin * ers, axis=1)
    tout_r = np.sum(tout * ers, axis=1)

    m_same = float(np.median(np.abs(tin_r[mask] - tout_r[mask])))
    m_oppo = float(np.median(np.abs(tin_r[mask] + tout_r[mask])))
    m_ref = float(np.median(np.abs(tin_r[mask])))
    return m_same, m_oppo, m_ref


@pytest.mark.e2e
def test_kozlov_free_disk_outer_radial_tilts_match_in_outer_band() -> None:
    """Tensionless theory: outer tilts should match (up to sign) in outer band."""
    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kozlov_1disk_3d_free_disk_theory_parity.yaml",
    )
    mesh = parse_geometry(load_data(path))
    gp = mesh.global_parameters

    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 10)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)

    gp.set("tilt_thetaB_optimize", True)
    gp.set("tilt_thetaB_optimize_every", 1)
    gp.set("tilt_thetaB_optimize_delta", 0.02)
    gp.set("tilt_thetaB_optimize_inner_steps", 2)

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

    # Allow a few coupled iterations so outer tilts respond.
    minim.minimize(n_steps=3)

    m_same, m_oppo, m_ref = _outer_radial_symmetry_metric(mesh, r_min=1.5, r_max=10.5)
    m_best = min(m_same, m_oppo)
    # Require mismatch to be small relative to signal when the radial tilt is
    # above numerical floor, otherwise enforce a strict absolute floor.
    assert m_ref > 0.0
    if m_ref >= 2.0e-8:
        assert (m_best / m_ref) < 0.4, (
            f"outer radial symmetry too weak: m_same={m_same:.3e} "
            f"m_oppo={m_oppo:.3e} m_ref={m_ref:.3e}"
        )
    else:
        assert m_best < 1.0e-8, (
            f"outer radial mismatch above floor: m_same={m_same:.3e} "
            f"m_oppo={m_oppo:.3e} m_ref={m_ref:.3e}"
        )
