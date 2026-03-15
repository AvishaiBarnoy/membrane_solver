import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _outer_band_radial_tilt_metrics(
    mesh, *, r_min: float, r_max: float
) -> tuple[float, float, float]:
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    mask = (r >= float(r_min)) & (r <= float(r_max))
    if not np.any(mask):
        raise AssertionError(f"Outer band empty: r in [{r_min}, {r_max}]")

    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]

    theta_in = np.einsum("ij,ij->i", mesh.tilts_in_view(), r_hat)
    theta_out = np.einsum("ij,ij->i", mesh.tilts_out_view(), r_hat)
    diff_same = float(np.median(np.abs(theta_in[mask] - theta_out[mask])))
    diff_oppo = float(np.median(np.abs(theta_in[mask] + theta_out[mask])))
    signal = float(
        np.median(np.maximum(np.abs(theta_in[mask]), np.abs(theta_out[mask])))
    )
    return diff_same, diff_oppo, signal


def _collect_group_rows(mesh, group: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") == group:
            rows.append(mesh.vertex_index_to_row[int(vid)])
    out = np.asarray(rows, dtype=int)
    if out.size == 0:
        raise AssertionError(f"No rows tagged with rim_slope_match_group={group!r}")
    positions = mesh.positions_view()
    angles = np.arctan2(positions[out, 1], positions[out, 0])
    return out[np.argsort(angles)]


def _rim_slope_proxy(mesh) -> float:
    positions = mesh.positions_view()
    rim_rows = _collect_group_rows(mesh, "rim")
    shell_rows = _collect_group_rows(mesh, "outer")
    if rim_rows.size != shell_rows.size:
        raise AssertionError(
            "Rim and outer slope rings must have the same sample count for this proxy"
        )

    rim_pos = positions[rim_rows]
    shell_pos = positions[shell_rows]
    dr = np.maximum(
        np.linalg.norm(shell_pos[:, :2], axis=1)
        - np.linalg.norm(rim_pos[:, :2], axis=1),
        1.0e-6,
    )
    return float(np.mean((shell_pos[:, 2] - rim_pos[:, 2]) / dr))


@pytest.mark.acceptance
def test_free_disk_coupled_bandwise_observables_match_tensionless_theory(
    canonical_profile_protocol_result,
) -> None:
    mesh, theta_b = canonical_profile_protocol_result

    diff_same, diff_oppo, signal = _outer_band_radial_tilt_metrics(
        mesh, r_min=1.5, r_max=10.5
    )
    assert signal > 1.0e-8

    best_mismatch = min(diff_same, diff_oppo)
    assert best_mismatch / signal <= 0.05, (
        f"outer-band parity mismatch too large: diff_same={diff_same:.3e} "
        f"diff_oppo={diff_oppo:.3e} signal={signal:.3e}"
    )

    phi = _rim_slope_proxy(mesh)
    phi_theory = 0.5 * theta_b
    assert float(np.ptp(mesh.positions_view()[:, 2])) > 1.0e-3
    assert abs(phi - phi_theory) <= max(0.40 * abs(phi_theory), 2.0e-3), (
        f"rim slope proxy mismatch: phi={phi:.3e}, thetaB/2={phi_theory:.3e}"
    )
