import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402
from tools.diagnostics.free_disk_profile_protocol import (  # noqa: E402
    measure_free_disk_curved_bilayer_near_rim,
    run_free_disk_curved_bilayer_protocol,
)


def _build_minimizer(mesh) -> Minimizer:
    """Return a minimizer for post-run energy breakdowns."""
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
def test_kozlov_named_mesh_matches_curved_shared_rim_protocol() -> None:
    """Acceptance: the named free-disk mesh reproduces the curved near-rim split."""
    mesh_path = (
        Path(__file__).resolve().parent.parent
        / "meshes"
        / "caveolin"
        / "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml"
    )
    mesh, theta_b = run_free_disk_curved_bilayer_protocol(curved_path=mesh_path)
    metrics = measure_free_disk_curved_bilayer_near_rim(mesh, theta_b=theta_b)
    target = 0.5 * theta_b

    assert metrics["theta_disk"] == pytest.approx(theta_b, rel=0.05, abs=1.0e-3)
    assert metrics["theta_outer_in"] == pytest.approx(target, rel=0.05, abs=1.0e-3)
    assert metrics["theta_outer_out"] == pytest.approx(target, rel=0.05, abs=1.0e-3)
    assert metrics["phi_abs"] == pytest.approx(target, rel=0.05, abs=1.0e-3)
    assert metrics["closure_error"] == pytest.approx(0.0, abs=1.0e-3)
    assert metrics["theta_out_phi_gap"] == pytest.approx(0.0, abs=1.0e-3)


@pytest.mark.e2e
def test_kozlov_named_mesh_has_active_outer_relaxation_channels() -> None:
    """Acceptance: the named free-disk mesh activates curvature and outer tilt."""
    mesh_path = (
        Path(__file__).resolve().parent.parent
        / "meshes"
        / "caveolin"
        / "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml"
    )
    mesh, theta_b = run_free_disk_curved_bilayer_protocol(curved_path=mesh_path)
    minim = _build_minimizer(mesh)
    breakdown = minim.compute_energy_breakdown()
    metrics = measure_free_disk_curved_bilayer_near_rim(mesh, theta_b=theta_b)

    assert float(breakdown.get("tilt_out") or 0.0) > 1.0e-3
    assert float(breakdown.get("bending_tilt_out") or 0.0) > 1.0e-3
    assert metrics["z_span"] > 1.0e-3

    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    mask = (r >= 1.5) & (r <= 10.5)
    assert np.any(mask)
    r_hat = np.zeros_like(positions)
    good = r > 1.0e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    theta_in = np.einsum("ij,ij->i", mesh.tilts_in_view(), r_hat)
    theta_out = np.einsum("ij,ij->i", mesh.tilts_out_view(), r_hat)
    diff_same = float(np.median(np.abs(theta_in[mask] - theta_out[mask])))
    diff_oppo = float(np.median(np.abs(theta_in[mask] + theta_out[mask])))
    assert diff_oppo <= diff_same + 1.0e-9
