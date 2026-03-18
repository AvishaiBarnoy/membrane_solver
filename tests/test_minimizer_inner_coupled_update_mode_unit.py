import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from runtime.minimizer import _apply_inner_coupled_update_mode_to_delta


def test_inner_coupled_update_mode_off_is_noop() -> None:
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ],
        dtype=float,
    )
    delta_in = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ],
        dtype=float,
    )
    updated, stats = _apply_inner_coupled_update_mode_to_delta(
        mesh=object(),
        global_params=GlobalParameters({"inner_coupled_update_mode": "off"}),
        positions=positions,
        fixed_mask_in=np.array([False, False], dtype=bool),
        delta_in=delta_in,
    )

    assert np.allclose(updated, delta_in)
    assert bool(stats["enabled"]) is False
    assert stats["mode"] == "off"


def test_inner_coupled_update_mode_caps_only_free_outer_near_radial_delta() -> None:
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [1.2, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ],
        dtype=float,
    )
    fixed_mask_in = np.array([False, False, False, False, True, False], dtype=bool)
    delta_in = np.zeros_like(positions)
    delta_in[0] = np.array([0.1, 0.0, 0.0])
    delta_in[1] = np.array([0.0, 0.1, 0.0])
    delta_in[2] = np.array([10.0, 0.0, 0.0])
    delta_in[3] = np.array([0.0, 10.0, 0.0])
    delta_in[4] = np.array([5.0, 0.0, 0.0])
    delta_in[5] = np.array([3.0, 0.0, 0.0])

    updated, stats = _apply_inner_coupled_update_mode_to_delta(
        mesh=object(),
        global_params=GlobalParameters(
            {
                "inner_coupled_update_mode": "rim_matched_radial_continuation_v1",
                "benchmark_disk_radius": 1.0,
                "benchmark_lambda_value": 0.1,
                "tilt_thetaB_center": [0.0, 0.0, 0.0],
            }
        ),
        positions=positions,
        fixed_mask_in=fixed_mask_in,
        delta_in=delta_in,
    )

    assert bool(stats["enabled"]) is True
    assert stats["mode"] == "rim_matched_radial_continuation_v1"
    assert int(stats["rim_row_count"]) == 2
    assert int(stats["candidate_row_count"]) == 2
    assert int(stats["capped_row_count"]) == 2

    updated_rad = np.array([updated[2, 0], updated[3, 1]])
    assert np.all(np.abs(updated_rad) <= float(stats["cap_magnitude"]) + 1.0e-12)

    assert np.allclose(updated[0], delta_in[0])
    assert np.allclose(updated[1], delta_in[1])
    assert np.allclose(updated[4], delta_in[4])
    assert np.allclose(updated[5], delta_in[5])


def test_inner_coupled_update_mode_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="inner_coupled_update_mode"):
        _apply_inner_coupled_update_mode_to_delta(
            mesh=object(),
            global_params=GlobalParameters({"inner_coupled_update_mode": "bad_mode"}),
            positions=np.zeros((1, 3), dtype=float),
            fixed_mask_in=np.array([False], dtype=bool),
            delta_in=np.zeros((1, 3), dtype=float),
        )
