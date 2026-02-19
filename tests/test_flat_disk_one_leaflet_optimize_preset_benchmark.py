from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.reproduce_flat_disk_one_leaflet import (
    DEFAULT_FIXTURE,
    run_flat_disk_one_leaflet_benchmark,
)


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_fast_r3_improves_runtime_with_stable_parity() -> (
    None
):
    t0 = time.perf_counter()
    baseline = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=3,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        optimize_preset="none",
    )
    baseline_seconds = float(time.perf_counter() - t0)

    t1 = time.perf_counter()
    fast = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=3,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        optimize_preset="fast_r3",
    )
    fast_seconds = float(time.perf_counter() - t1)

    assert fast_seconds <= 0.80 * baseline_seconds

    base_theta = float(baseline["parity"]["theta_factor"])
    fast_theta = float(fast["parity"]["theta_factor"])
    base_energy = float(baseline["parity"]["energy_factor"])
    fast_energy = float(fast["parity"]["energy_factor"])

    assert fast_theta <= 1.05 * base_theta
    assert fast_energy <= 1.05 * base_energy
