import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from tools.diagnostics.free_disk_profile_protocol import (  # noqa: E402
    optimize_free_disk_theta_b,
    run_free_disk_curved_bilayer_theta_sweep,
)


@pytest.mark.regression
def test_kozlov_free_disk_flat_thetaB_scan_stays_on_flat_surrogate_branch() -> None:
    """Diagnostic: the legacy flat free-disk scan still lands on the flat branch."""
    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kozlov_1disk_3d_free_disk_theory_parity.yaml",
    )
    mesh = parse_geometry(load_data(path))
    theta_b = optimize_free_disk_theta_b(mesh, scans=4)

    assert theta_b == pytest.approx(0.04, abs=0.02)
    assert float(np.ptp(mesh.positions_view()[:, 2])) == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.e2e
def test_kozlov_free_disk_curved_theta_sweep_scales_linearly_with_imposed_drive() -> (
    None
):
    """E2E: the curved shared-rim protocol tracks imposed thetaB on the named mesh."""
    mesh_path = (
        Path(__file__).resolve().parent.parent
        / "meshes"
        / "caveolin"
        / "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml"
    )
    rows = run_free_disk_curved_bilayer_theta_sweep(
        [0.02, 0.04, 0.10],
        curved_path=mesh_path,
    )

    assert len(rows) == 3
    for row in rows:
        target = 0.5 * row["theta_b"]
        assert row["theta_disk"] == pytest.approx(row["theta_b"], rel=0.05, abs=1.0e-3)
        assert row["theta_outer_in"] == pytest.approx(target, rel=0.05, abs=1.0e-3)
        assert row["theta_outer_out"] == pytest.approx(target, rel=0.05, abs=1.0e-3)
        assert row["phi_abs"] == pytest.approx(target, rel=0.05, abs=1.0e-3)
        assert row["closure_error"] == pytest.approx(0.0, abs=1.0e-3)

    phi_vals = np.asarray([row["phi_abs"] for row in rows], dtype=float)
    theta_vals = np.asarray([row["theta_b"] for row in rows], dtype=float)
    assert np.all(np.diff(phi_vals) > 0.0)
    assert np.allclose(phi_vals / theta_vals, 0.5, rtol=0.05, atol=1.0e-3)
