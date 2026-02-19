from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.benchmark_flat_disk_tilt_mass_mode import benchmark_tilt_mass_mode


@pytest.mark.benchmark
@pytest.mark.regression
def test_flat_disk_tilt_mass_mode_benchmark_smoke() -> None:
    report = benchmark_tilt_mass_mode(
        refine_level=1,
        runs=1,
        theta_mode="optimize",
        optimize_preset="kh_wide",
    )

    assert report["meta"]["parameterization"] == "kh_physical"
    lumped = report["modes"]["lumped"]
    consistent = report["modes"]["consistent"]

    for mode in (lumped, consistent):
        t = mode["timing_seconds"]
        assert int(t["runs"]) == 1
        assert len(t["values"]) == 1
        assert np.isfinite(float(t["mean"]))
        assert np.isfinite(float(t["median"]))
        assert np.isfinite(float(mode["parity"]["theta_factor"]))
        assert np.isfinite(float(mode["parity"]["energy_factor"]))

    ratio = float(report["comparison"]["median_time_ratio_consistent_over_lumped"])
    assert np.isfinite(ratio)
    assert ratio > 0.0
