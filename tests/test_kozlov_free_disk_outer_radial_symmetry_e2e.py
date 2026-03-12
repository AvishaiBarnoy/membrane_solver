import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.free_disk_profile_protocol import (  # noqa: E402
    run_free_disk_curved_bilayer_protocol,
)


def _outer_band_mask(mesh, *, r_min: float, r_max: float) -> np.ndarray:
    """Return a boolean mask for vertices in the requested radial band."""
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
    """E2E: curved shared-rim protocol matches the outer-band opposite-sign branch."""
    mesh, _ = run_free_disk_curved_bilayer_protocol()

    m_same, m_oppo, m_ref = _outer_radial_symmetry_metric(mesh, r_min=1.5, r_max=10.5)
    assert m_ref > 0.0
    assert m_oppo < 1.0e-8, (
        f"outer opposite-sign mismatch above floor: m_same={m_same:.3e} "
        f"m_oppo={m_oppo:.3e} m_ref={m_ref:.3e}"
    )
    assert m_oppo <= m_same + 1.0e-9
