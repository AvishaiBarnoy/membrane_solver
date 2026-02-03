import os
import sys

import numpy as np
import pytest
from scipy import special

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


def _disk_group_rows(mesh, group: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if (
            opts.get("rim_slope_match_group") == group
            or opts.get("tilt_thetaB_group") == group
        ):
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    return np.asarray(rows, dtype=int)


def _order_by_angle(pts: np.ndarray, *, center: np.ndarray) -> np.ndarray:
    rel = pts - center[None, :]
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    return np.argsort(ang)


def _arc_length_weights(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=float)
    diffs_next = pts[(np.arange(n) + 1) % n] - pts
    diffs_prev = pts - pts[(np.arange(n) - 1) % n]
    l_next = np.linalg.norm(diffs_next, axis=1)
    l_prev = np.linalg.norm(diffs_prev, axis=1)
    return 0.5 * (l_next + l_prev)


@pytest.mark.e2e
def test_kozlov_free_disk_thetaB_matches_tensionless_theory() -> None:
    """E2E: thetaB optimization produces theory-scale thetaB on the free-disk mesh.

    This test focuses on robust, low-cost observables:
      - thetaB is nontrivial and close to the tensionless prediction (with
        elastic coefficients from both leaflets, since contact work is applied
        only to the inner leaflet)
      - contact energy term matches -2π R_eff * drive * thetaB (using the discrete R_eff)

    The mesh is loaded from tests/data to keep this test independent of the
    hand-authored YAMLs under meshes/.
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kozlov_1disk_3d_free_disk_theory_parity.yaml",
    )
    mesh = parse_geometry(load_data(path))
    gp = mesh.global_parameters

    # Fast tilt relaxation and frequent thetaB scans keep this test quick.
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 5)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)

    gp.set("tilt_thetaB_optimize", True)
    gp.set("tilt_thetaB_optimize_every", 1)
    # Use a larger delta and a tiny inner budget so we can move thetaB away from
    # zero in a single scan while keeping runtime bounded.
    gp.set("tilt_thetaB_optimize_delta", 0.05)
    gp.set("tilt_thetaB_optimize_inner_steps", 2)

    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 0.01)

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )

    # --- Tensionless theory (docs/tex/1_disk_3d.tex) ---
    kappa_in = float(gp.get("bending_modulus_in") or gp.get("bending_modulus") or 0.0)
    kappa_out = float(gp.get("bending_modulus_out") or gp.get("bending_modulus") or 0.0)
    kappa = float(kappa_in + kappa_out)

    kappa_t_in = float(gp.get("tilt_modulus_in") or 0.0)
    kappa_t_out = float(gp.get("tilt_modulus_out") or 0.0)
    kappa_t = float(kappa_t_in + kappa_t_out)
    drive = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)
    assert kappa > 0.0 and kappa_t > 0.0 and drive != 0.0

    # Avoid expensive shape-gradient evaluations; for this theory-parity test we
    # only need the reduced-energy thetaB scan + tilt relaxation on fixed
    # geometry.
    tilt_mode = str(gp.get("tilt_solve_mode") or "coupled")
    for i in range(1):
        minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode=tilt_mode)
        minim._optimize_thetaB_scalar(tilt_mode=tilt_mode, iteration=i)

    thetaB = float(gp.get("tilt_thetaB_value") or 0.0)
    assert thetaB > 1.0e-2

    R = 7.0 / 15.0
    lam = float(np.sqrt(kappa_t / kappa))
    x = lam * R
    den = float(
        special.iv(0, x) / special.iv(1, x) + 0.5 * special.kv(0, x) / special.kv(1, x)
    )
    thetaB_star = float(drive / (np.sqrt(kappa * kappa_t) * den))

    assert thetaB == pytest.approx(thetaB_star, rel=0.30, abs=0.02)

    # --- Contact term check using the same discrete R_eff convention ---
    rows = _disk_group_rows(mesh, "disk")
    assert rows.size > 0
    positions = mesh.positions_view()
    center = np.asarray(
        gp.get("tilt_thetaB_center") or [0.0, 0.0, 0.0], dtype=float
    ).reshape(3)

    pts = positions[rows]
    order = _order_by_angle(pts, center=center)
    pts = pts[order]
    weights = _arc_length_weights(pts)
    wsum = float(np.sum(weights))
    assert wsum > 1e-12

    r_len = np.linalg.norm((pts - center[None, :])[:, :2], axis=1)
    R_eff = float(np.sum(weights * r_len) / wsum)

    breakdown = minim.compute_energy_breakdown()
    contact_energy = float(breakdown.get("tilt_thetaB_contact_in") or 0.0)
    contact_pred = float(-2.0 * np.pi * R_eff * drive * thetaB)

    assert contact_energy == pytest.approx(contact_pred, rel=0.05, abs=0.02)
