from pathlib import Path

import numpy as np

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _outer_radial_tilt_symmetry_metric(
    mesh, *, r_min: float, r_max: float
) -> tuple[float, float]:
    """Return (median |tin_r - tout_r|, median |tin_r + tout_r|) in an outer band."""
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    mask = (r >= float(r_min)) & (r <= float(r_max))
    if not np.any(mask):
        return 0.0, 0.0

    ers = np.zeros_like(positions)
    nz = r > 1e-12
    ers[nz, 0] = positions[nz, 0] / r[nz]
    ers[nz, 1] = positions[nz, 1] / r[nz]

    tin = mesh.tilts_in_view()
    tout = mesh.tilts_out_view()
    tin_r = np.sum(tin * ers, axis=1)
    tout_r = np.sum(tout * ers, axis=1)
    return (
        float(np.median(np.abs(tin_r[mask] - tout_r[mask]))),
        float(np.median(np.abs(tin_r[mask] + tout_r[mask]))),
    )


def test_kozlov_free_disk_thetaB_theory_mode_smoke_and_symmetry() -> None:
    mesh = parse_geometry(
        load_data(
            str(
                Path(__file__).resolve().parent
                / "fixtures"
                / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
            )
        )
    )

    # Keep this test robust and cheap:
    # - ensure theory-mode flags are present
    # - run a small number of outer iterations
    # - assert basic physical sanity + outer-leaflet symmetry trend.
    gp = mesh.global_parameters
    # Ensure this config runs in theory-aligned mode (no penalty energy leakage).
    penalty_mode = str(gp.get("tilt_thetaB_contact_penalty_mode") or "").strip().lower()
    assert penalty_mode not in {"legacy", "on", "true", "1"}
    assert "tilt_thetaB_boundary_in" in (mesh.constraint_modules or [])

    em = EnergyModuleManager(mesh.energy_modules)
    cm = ConstraintModuleManager(mesh.constraint_modules)
    minim = Minimizer(mesh, gp, GradientDescent(), em, cm, quiet=True)

    # Keep this test fast: cap inner tilt work and thetaB scan effort.
    gp.set("tilt_kkt_projection_during_relaxation", False)
    gp.set("tilt_inner_steps", 3)
    gp.set("tilt_step_size", 1e-6)
    gp.set("tilt_tol", 0.0)
    gp.set("tilt_thetaB_optimize", True)
    gp.set("tilt_thetaB_optimize_every", 1)
    gp.set("tilt_thetaB_optimize_delta", 0.01)
    gp.set("tilt_thetaB_optimize_inner_steps", 3)

    res = minim.minimize(n_steps=2)
    assert res["mesh"] is mesh

    thetaB = float(gp.get("tilt_thetaB_value") or 0.0)
    assert thetaB > 0.0

    bd = minim.compute_energy_breakdown()
    # Pure contact work should be negative for gamma>0 and thetaB>0.
    assert float(bd.get("tilt_thetaB_contact_in", 0.0)) < 0.0

    # Far from the disk boundary, the theory predicts opposite-leaflet tilts match
    # up to sign convention. Check that the opposite-sign mismatch is small.
    diff_same, diff_oppo = _outer_radial_tilt_symmetry_metric(
        mesh, r_min=1.5, r_max=10.5
    )
    assert diff_oppo <= diff_same
