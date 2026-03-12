import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.free_disk_profile_protocol import (  # noqa: E402
    run_free_disk_curved_bilayer_protocol,
)


def _radial_tilts(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1.0e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    theta_in = np.einsum("ij,ij->i", mesh.tilts_in_view(), r_hat)
    theta_out = np.einsum("ij,ij->i", mesh.tilts_out_view(), r_hat)
    return positions, r, theta_in, theta_out


def _rows_at_radius(
    r: np.ndarray, target: float, *, atol: float = 1.0e-6
) -> np.ndarray:
    rows = np.where(np.isclose(r, float(target), atol=atol))[0]
    if rows.size == 0:
        raise AssertionError(f"No rows found near r={target:.6f}")
    return rows


def _first_free_ring_radius(r: np.ndarray, *, R: float) -> float:
    radii = sorted(
        {
            round(float(rr), 6)
            for rr in r
            if rr > float(R) + 1.0e-6 and rr < 12.0 - 1.0e-6
        }
    )
    if not radii:
        raise AssertionError("No free ring found outside the physical disk edge")
    return float(radii[0])


@pytest.mark.acceptance
def test_curved_bilayer_near_rim_matches_tensionless_theory() -> None:
    mesh, theta_b = run_free_disk_curved_bilayer_protocol()
    positions, r, theta_in, theta_out = _radial_tilts(mesh)

    R = 7.0 / 15.0
    target = 0.5 * theta_b
    ring_r = _first_free_ring_radius(r, R=R)

    disk_rows = _rows_at_radius(r, R)
    outer_rows = _rows_at_radius(r, ring_r)

    theta_disk = float(np.median(theta_in[disk_rows]))
    theta_outer_in = float(np.median(theta_in[outer_rows]))
    # Shared-rim staggered mode evaluates the outer leaflet directly on the
    # first free ring, so the near-rim parity check uses the raw radial sign.
    theta_outer_out = float(np.median(theta_out[outer_rows]))
    dr = float(np.median(r[outer_rows]) - np.median(r[disk_rows]))
    phi = abs(
        (np.median(positions[outer_rows, 2]) - np.median(positions[disk_rows, 2])) / dr
    )

    assert abs(theta_disk - theta_b) <= max(0.10 * abs(theta_b), 1.0e-3)
    assert abs(theta_outer_in - target) <= max(0.25 * abs(target), 1.0e-3)
    assert abs(theta_outer_out - target) <= max(0.25 * abs(target), 1.0e-3)
    assert abs(phi - target) <= max(0.25 * abs(target), 1.0e-3)
