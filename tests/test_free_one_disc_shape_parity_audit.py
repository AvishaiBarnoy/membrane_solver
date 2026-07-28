from __future__ import annotations

import numpy as np

from tools.free_one_disc_shape_parity_audit import _unit


def test_unit_returns_normalized_direction() -> None:
    values = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])

    result = _unit(values)

    assert np.linalg.norm(result) == 1.0
    np.testing.assert_array_equal(result[1], np.zeros(3))


def test_unit_handles_zero_direction() -> None:
    values = np.zeros((3, 3), dtype=float)

    np.testing.assert_array_equal(_unit(values), values)
