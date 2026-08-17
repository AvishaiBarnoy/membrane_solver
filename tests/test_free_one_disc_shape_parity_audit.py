from __future__ import annotations

import numpy as np
import pytest

from tools.free_one_disc_shape_parity_audit import _unit, run_shape_parity_audit


def test_unit_returns_normalized_direction() -> None:
    values = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])

    result = _unit(values)

    assert np.linalg.norm(result) == 1.0
    np.testing.assert_array_equal(result[1], np.zeros(3))


def test_unit_handles_zero_direction() -> None:
    values = np.zeros((3, 3), dtype=float)

    np.testing.assert_array_equal(_unit(values), values)


@pytest.mark.slow
def test_oriented_curvature_tilt_cross_shape_pullback_matches_energy() -> None:
    report = run_shape_parity_audit(
        angular_sectors=12,
        near_spacing=0.02,
        epsilons=(3.0e-6,),
    )

    for name in ("bending_tilt_in", "bending_tilt_out"):
        row = report["module_shape_derivatives"][name]
        scale = max(abs(row["fd_slope"]), 1.0)
        assert abs(row["fd_minus_analytic"]) / scale < 0.05
