from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.reproduce_flat_disk_one_leaflet import (  # noqa: E402
    DEFAULT_FIXTURE,
    run_flat_disk_one_leaflet_benchmark,
)


def _assert_perf_fields(report: dict) -> None:
    perf = report["meta"]["performance"]
    for key in (
        "mesh_load_refine_seconds",
        "setup_seconds",
        "theta_optimize_seconds",
        "final_relax_report_seconds",
        "total_runtime_seconds",
    ):
        val = float(perf[key])
        assert np.isfinite(val), key
        assert val >= 0.0, key
    assert int(perf["theta_evaluations"]) >= 0


@pytest.mark.regression
def test_flat_disk_performance_telemetry_fields_optimize() -> None:
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=0,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        parameterization="kh_physical",
        theta_optimize_steps=3,
        theta_optimize_every=1,
        theta_optimize_delta=5.0e-4,
        theta_optimize_inner_steps=3,
    )
    _assert_perf_fields(report)
    perf = report["meta"]["performance"]
    assert int(perf["theta_evaluations"]) == 3
    assert float(perf["theta_optimize_seconds"]) > 0.0


@pytest.mark.regression
def test_flat_disk_performance_telemetry_fields_scan() -> None:
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=0,
        outer_mode="disabled",
        smoothness_model="dirichlet",
        theta_mode="scan",
        theta_min=0.0,
        theta_max=1.4e-3,
        theta_count=5,
    )
    _assert_perf_fields(report)
    perf = report["meta"]["performance"]
    assert int(perf["theta_evaluations"]) == 5
    assert float(perf["theta_optimize_seconds"]) == pytest.approx(0.0, abs=0.0)


@pytest.mark.regression
def test_flat_disk_optimize_plateau_stop_reduces_theta_evaluations() -> None:
    report = run_flat_disk_one_leaflet_benchmark(
        fixture=DEFAULT_FIXTURE,
        refine_level=0,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_mode="optimize",
        parameterization="kh_physical",
        theta_optimize_steps=8,
        theta_optimize_every=1,
        theta_optimize_delta=5.0e-4,
        theta_optimize_inner_steps=3,
        theta_optimize_plateau_patience=1,
        theta_optimize_plateau_abs_tol=1.0,
    )
    _assert_perf_fields(report)
    opt = report["optimize"]
    assert opt is not None
    assert bool(opt["stopped_on_plateau"]) is True
    assert int(opt["optimize_iterations_completed"]) < int(opt["optimize_steps"])
    assert int(report["meta"]["performance"]["theta_evaluations"]) == int(
        opt["optimize_iterations_completed"]
    )
