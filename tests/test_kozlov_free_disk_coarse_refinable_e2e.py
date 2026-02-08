import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.refinement import refine_triangle_mesh  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402
from runtime.vertex_average import vertex_average  # noqa: E402


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
def test_kozlov_free_disk_coarse_refinable_runs_refine_avg_and_minimize() -> None:
    """E2E smoke: coarse refinable Kozlov mesh survives refine+avg and minimizes."""
    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kozlov_free_disk_coarse_refinable.yaml",
    )
    mesh = parse_geometry(load_data(path))

    # Mirror the interactive workflow that previously broke validation.
    mesh = refine_triangle_mesh(mesh)
    for _ in range(5):
        vertex_average(mesh)

    gp = mesh.global_parameters
    gp.set("disk_interface_validate", True)

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )
    minim.minimize(n_steps=5)

    thetaB = float(gp.get("tilt_thetaB_value") or 0.0)
    assert thetaB >= 0.0

    breakdown = minim.compute_energy_breakdown()
    contact_energy = float(breakdown.get("tilt_thetaB_contact_in") or 0.0)
    drive = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)
    if abs(drive) < 1e-12:
        return

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
    contact_pred = float(-2.0 * np.pi * R_eff * drive * thetaB)

    assert contact_energy == pytest.approx(contact_pred, rel=0.10, abs=0.05)
