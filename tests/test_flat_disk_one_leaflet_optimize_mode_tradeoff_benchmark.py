from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.reproduce_flat_disk_one_leaflet import (
    DEFAULT_FIXTURE,
    run_flat_disk_one_leaflet_benchmark,
)


@pytest.mark.benchmark
def test_flat_disk_optimize_vs_optimize_full_tradeoff_refine3() -> None:
    optimize = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=3,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        optimize_preset="none",
    )
    optimize_full = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=3,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize_full",
        theta_polish_delta=1.0e-4,
        theta_polish_points=3,
        optimize_preset="none",
    )

    theta_opt = float(optimize["parity"]["theta_factor"])
    theta_full = float(optimize_full["parity"]["theta_factor"])
    energy_opt = float(optimize["parity"]["energy_factor"])
    energy_full = float(optimize_full["parity"]["energy_factor"])

    assert theta_opt <= theta_full
    assert energy_full <= energy_opt


@pytest.mark.benchmark
def test_flat_disk_splay_scale_half_improves_energy_parity_refine3() -> None:
    baseline = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=3,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        optimize_preset="none",
        splay_modulus_scale_in=1.0,
    )
    tuned = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=3,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        optimize_preset="none",
        splay_modulus_scale_in=0.5,
    )

    theta_base = float(baseline["parity"]["theta_factor"])
    theta_tuned = float(tuned["parity"]["theta_factor"])
    energy_base = float(baseline["parity"]["energy_factor"])
    energy_tuned = float(tuned["parity"]["energy_factor"])

    assert theta_tuned <= 1.05 * theta_base
    assert energy_tuned <= 0.90 * energy_base
