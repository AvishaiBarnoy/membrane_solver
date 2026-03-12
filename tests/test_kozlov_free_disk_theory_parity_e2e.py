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
from tools.diagnostics.free_disk_profile_protocol import (  # noqa: E402
    measure_free_disk_curved_bilayer_near_rim,
    run_free_disk_curved_bilayer_protocol,
)


def _disk_group_rows(mesh, group: str) -> np.ndarray:
    """Return rows whose rim-slope match group equals ``group``."""
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") == group:
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    return np.asarray(rows, dtype=int)


def _order_by_angle(pts: np.ndarray, *, center: np.ndarray) -> np.ndarray:
    """Return indices that order planar points by polar angle."""
    rel = pts - center[None, :]
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    return np.argsort(ang)


def _arc_length_weights(pts: np.ndarray) -> np.ndarray:
    """Return trapezoid-like arc weights along a closed polygonal ring."""
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=float)
    diffs_next = pts[(np.arange(n) + 1) % n] - pts
    diffs_prev = pts - pts[(np.arange(n) - 1) % n]
    l_next = np.linalg.norm(diffs_next, axis=1)
    l_prev = np.linalg.norm(diffs_prev, axis=1)
    return 0.5 * (l_next + l_prev)


def _tensionless_thetaB_prediction(
    *, kappa: float, kappa_t: float, drive: float, R: float
):
    """Return the TeX tensionless prediction tuple for compatibility helpers."""
    lam = float(np.sqrt(kappa_t / kappa))
    x = lam * float(R)
    ratio_i = float(special.iv(0, x) / special.iv(1, x))
    ratio_k = float(special.kv(0, x) / special.kv(1, x))
    den = float(ratio_i + 0.5 * ratio_k)

    theta_star = float(drive / (np.sqrt(kappa * kappa_t) * den))
    fin = float(np.pi * kappa * R * lam * ratio_i * theta_star**2)
    fout = float(np.pi * kappa * R * lam * 0.5 * ratio_k * theta_star**2)
    fcont = float(-2.0 * np.pi * R * drive * theta_star)
    return theta_star, fin, fout, fcont, float(fin + fout + fcont)


def _symmetric_leaflet_theory_params(gp) -> tuple[float, float, float]:
    """Return symmetric-theory `(kappa, kappa_t, drive)` from leaflet params."""
    kappa_in = gp.get("bending_modulus_in")
    kappa_out = gp.get("bending_modulus_out")
    kappa_bulk = gp.get("bending_modulus")
    if kappa_in is not None and kappa_out is not None:
        kappa = float(kappa_in)
        assert kappa == pytest.approx(float(kappa_out), rel=0.0, abs=1.0e-12)
    elif kappa_in is not None:
        kappa = float(kappa_in)
    elif kappa_out is not None:
        kappa = float(kappa_out)
    else:
        kappa = float(kappa_bulk or 0.0)

    kappa_t_in = gp.get("tilt_modulus_in")
    kappa_t_out = gp.get("tilt_modulus_out")
    kappa_t_bulk = gp.get("tilt_modulus")
    if kappa_t_in is not None and kappa_t_out is not None:
        kappa_t = float(kappa_t_in)
        assert kappa_t == pytest.approx(float(kappa_t_out), rel=0.0, abs=1.0e-12)
    elif kappa_t_in is not None:
        kappa_t = float(kappa_t_in)
    elif kappa_t_out is not None:
        kappa_t = float(kappa_t_out)
    else:
        kappa_t = float(kappa_t_bulk or 0.0)

    drive = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)
    return kappa, kappa_t, drive


def _build_minimizer(mesh) -> Minimizer:
    """Return a minimizer suitable for post-run energy breakdowns."""
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )


@pytest.mark.e2e
def test_kozlov_free_disk_curved_protocol_matches_near_rim_tensionless_split() -> None:
    """E2E: the curved shared-rim protocol reproduces the near-rim theory split."""
    mesh, theta_b = run_free_disk_curved_bilayer_protocol()
    metrics = measure_free_disk_curved_bilayer_near_rim(mesh, theta_b=theta_b)
    target = 0.5 * theta_b

    assert metrics["theta_disk"] == pytest.approx(theta_b, rel=0.05, abs=1.0e-3)
    assert metrics["theta_outer_in"] == pytest.approx(target, rel=0.05, abs=1.0e-3)
    assert metrics["theta_outer_out"] == pytest.approx(target, rel=0.05, abs=1.0e-3)
    assert metrics["phi_abs"] == pytest.approx(target, rel=0.05, abs=1.0e-3)
    assert metrics["closure_error"] == pytest.approx(0.0, abs=1.0e-3)
    assert metrics["theta_out_phi_gap"] == pytest.approx(0.0, abs=1.0e-3)


@pytest.mark.e2e
def test_kozlov_free_disk_curved_protocol_has_active_outer_relaxation_channels() -> (
    None
):
    """E2E: the curved protocol activates shape and outer-leaflet relaxation."""
    mesh, theta_b = run_free_disk_curved_bilayer_protocol()
    minim = _build_minimizer(mesh)
    breakdown = minim.compute_energy_breakdown()

    rows = _disk_group_rows(mesh, "rim")
    assert rows.size > 0
    positions = mesh.positions_view()
    center = np.asarray(
        mesh.global_parameters.get("tilt_thetaB_center") or [0.0, 0.0, 0.0], dtype=float
    ).reshape(3)
    pts = positions[rows]
    order = _order_by_angle(pts, center=center)
    weights = _arc_length_weights(pts[order])
    wsum = float(np.sum(weights))
    assert wsum > 1.0e-12
    r_len = np.linalg.norm((pts[order] - center[None, :])[:, :2], axis=1)
    r_eff = float(np.sum(weights * r_len) / wsum)

    drive = float(mesh.global_parameters.get("tilt_thetaB_contact_strength_in") or 0.0)
    contact_energy = float(breakdown.get("tilt_thetaB_contact_in") or 0.0)
    contact_pred = float(-2.0 * np.pi * r_eff * drive * theta_b)

    assert contact_energy == pytest.approx(contact_pred, rel=0.05, abs=0.02)
    assert float(breakdown.get("tilt_out") or 0.0) > 1.0e-3
    assert float(breakdown.get("bending_tilt_out") or 0.0) > 1.0e-3
    assert float(np.ptp(mesh.positions_view()[:, 2])) > 1.0e-3
    assert sum(float(v) for v in breakdown.values()) < 0.0


@pytest.mark.regression
def test_kozlov_free_disk_flat_fixture_remains_flat_diagnostic() -> None:
    """Diagnostic: the legacy theory fixture still represents the flat surrogate."""
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
    gp.set("tilt_inner_steps", 5)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)
    gp.set("tilt_thetaB_optimize", True)
    gp.set("tilt_thetaB_optimize_every", 1)
    gp.set("tilt_thetaB_optimize_delta", 0.02)
    gp.set("tilt_thetaB_optimize_inner_steps", 2)

    minim = _build_minimizer(mesh)
    tilt_mode = str(gp.get("tilt_solve_mode") or "coupled")
    for i in range(4):
        minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode=tilt_mode)
        minim._optimize_thetaB_scalar(tilt_mode=tilt_mode, iteration=i)

    breakdown = minim.compute_energy_breakdown()
    assert float(np.ptp(mesh.positions_view()[:, 2])) == pytest.approx(0.0, abs=1.0e-12)
    tilt_in = float(breakdown.get("tilt_in") or 0.0)
    tilt_out = float(breakdown.get("tilt_out") or 0.0)
    bend_out = float(breakdown.get("bending_tilt_out") or 0.0)
    assert tilt_out < max(1.0e-5, 1.0e-4 * max(tilt_in, 1.0))
    assert bend_out < 1.0e-6
