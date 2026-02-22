from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_kh_partition_ablation import (  # noqa: E402
    run_flat_disk_kh_partition_ablation,
)


@pytest.mark.regression
def test_flat_disk_kh_partition_ablation_emits_baseline_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_flat_disk_one_leaflet_benchmark(**kwargs):
        preset = str(kwargs.get("optimize_preset"))
        if preset == "kh_strict_partition_tight":
            return {
                "meta": {
                    "refine_level": 1,
                    "rim_local_refine_steps": 2,
                    "rim_local_refine_band_lambda": 10.0,
                },
                "mesh": {"theta_star": 0.146},
                "parity": {"theta_factor": 1.02, "energy_factor": 1.07},
            }
        return {
            "meta": {
                "refine_level": 1,
                "rim_local_refine_steps": 2,
                "rim_local_refine_band_lambda": 8.0,
            },
            "mesh": {"theta_star": 0.141},
            "parity": {"theta_factor": 1.02, "energy_factor": 1.09},
        }

    def _fake_run_flat_disk_kh_term_audit(**kwargs):
        theta = float(kwargs["theta_values"][0])
        if theta >= 0.145:
            row = {
                "internal_disk_ratio_mesh_over_theory": 0.93,
                "internal_outer_ratio_mesh_over_theory": 1.05,
                "mesh_internal_disk_core": 0.10,
                "mesh_internal_rim_band": 0.20,
                "mesh_internal_outer_near": 0.30,
                "mesh_internal_outer_far": 0.40,
                "rim_band_h_over_lambda_median": 0.75,
            }
        else:
            row = {
                "internal_disk_ratio_mesh_over_theory": 0.80,
                "internal_outer_ratio_mesh_over_theory": 1.40,
                "mesh_internal_disk_core": 0.11,
                "mesh_internal_rim_band": 0.22,
                "mesh_internal_outer_near": 0.33,
                "mesh_internal_outer_far": 0.44,
                "rim_band_h_over_lambda_median": 0.85,
            }
        return {"rows": [row]}

    monkeypatch.setattr(
        "tools.reproduce_flat_disk_one_leaflet.run_flat_disk_one_leaflet_benchmark",
        _fake_run_flat_disk_one_leaflet_benchmark,
    )
    monkeypatch.setattr(
        "tools.diagnostics.flat_disk_kh_term_audit.run_flat_disk_kh_term_audit",
        _fake_run_flat_disk_kh_term_audit,
    )

    report = run_flat_disk_kh_partition_ablation(
        optimize_presets=("kh_strict_energy_tight", "kh_strict_partition_tight"),
        baseline_optimize_preset="kh_strict_energy_tight",
    )
    assert report["meta"]["mode"] == "flat_disk_kh_partition_ablation"
    assert len(report["rows"]) == 2
    for row in report["rows"]:
        assert np.isfinite(float(row["theta_factor"]))
        assert np.isfinite(float(row["energy_factor"]))
        assert np.isfinite(float(row["partition_score"]))
        assert np.isfinite(float(row["mesh_internal_disk_core"]))
        assert np.isfinite(float(row["mesh_internal_rim_band"]))
        assert np.isfinite(float(row["mesh_internal_outer_near"]))
        assert np.isfinite(float(row["mesh_internal_outer_far"]))
        assert np.isfinite(float(row["rim_band_h_over_lambda_median"]))
    assert report["selected_best"]["optimize_preset"] == "kh_strict_partition_tight"
    assert report["baseline_best"]["optimize_preset"] == "kh_strict_energy_tight"
    assert float(report["selected_vs_baseline_partition_score_delta"]) < 0.0
    assert float(report["selected_vs_baseline_energy_factor_delta"]) < 0.0
