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


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_strict_fast_improves_runtime_with_parity() -> (
    None
):
    t0 = time.perf_counter()
    strict = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        parameterization="kh_physical",
        optimize_preset="kh_strict_refine",
        tilt_mass_mode_in="consistent",
    )
    strict_seconds = float(time.perf_counter() - t0)

    t1 = time.perf_counter()
    fast = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        parameterization="kh_physical",
        optimize_preset="kh_strict_fast",
        tilt_mass_mode_in="consistent",
    )
    fast_seconds = float(time.perf_counter() - t1)

    assert fast_seconds <= 0.80 * strict_seconds

    strict_theta = float(strict["parity"]["theta_factor"])
    strict_energy = float(strict["parity"]["energy_factor"])
    fast_theta = float(fast["parity"]["theta_factor"])
    fast_energy = float(fast["parity"]["energy_factor"])

    assert fast_theta <= 1.10 * strict_theta
    assert fast_energy <= 1.10 * strict_energy
    assert bool(fast["optimize"]["hit_step_limit"]) is False
    assert fast["optimize"]["recommended_fallback_preset"] is None


@pytest.mark.benchmark
def test_flat_disk_optimize_preset_kh_strict_fast_is_deterministic() -> None:
    run_a = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        parameterization="kh_physical",
        optimize_preset="kh_strict_fast",
        tilt_mass_mode_in="consistent",
    )
    run_b = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        parameterization="kh_physical",
        optimize_preset="kh_strict_fast",
        tilt_mass_mode_in="consistent",
    )

    assert float(run_a["mesh"]["theta_star"]) == pytest.approx(
        float(run_b["mesh"]["theta_star"]), abs=1e-12
    )
    assert float(run_a["parity"]["theta_factor"]) == pytest.approx(
        float(run_b["parity"]["theta_factor"]), abs=1e-12
    )
    assert float(run_a["parity"]["energy_factor"]) == pytest.approx(
        float(run_b["parity"]["energy_factor"]), abs=1e-12
    )
