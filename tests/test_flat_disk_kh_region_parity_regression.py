from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_kh_region_parity import (  # noqa: E402
    run_flat_disk_kh_region_parity,
)


@pytest.mark.regression
def test_flat_disk_kh_region_parity_emits_ranked_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_flat_disk_one_leaflet_benchmark(**kwargs):
        preset = str(kwargs.get("optimize_preset"))
        rim_steps = int(kwargs.get("rim_local_refine_steps", 0) or 0)
        if preset == "kh_strict_continuity":
            return {
                "meta": {
                    "refine_level": 1,
                    "rim_local_refine_steps": rim_steps,
                    "rim_local_refine_band_lambda": float(
                        kwargs.get("rim_local_refine_band_lambda", 4.0)
                    ),
                },
                "mesh": {"theta_star": 0.136 + 0.001 * rim_steps},
                "parity": {"theta_factor": 1.04, "energy_factor": 1.08},
            }
        return {
            "meta": {
                "refine_level": 1,
                "rim_local_refine_steps": rim_steps,
                "rim_local_refine_band_lambda": float(
                    kwargs.get("rim_local_refine_band_lambda", 4.0)
                ),
            },
            "mesh": {"theta_star": 0.132 + 0.001 * rim_steps},
            "parity": {"theta_factor": 1.08, "energy_factor": 1.11},
        }

    def _fake_run_flat_disk_kh_term_audit(**kwargs):
        rim_steps = int(kwargs.get("rim_local_refine_steps", 0))
        theta = float(kwargs["theta_values"][0])
        if theta > 0.136 and rim_steps >= 2:
            row = {
                "internal_disk_ratio_mesh_over_theory": 0.95,
                "internal_outer_ratio_mesh_over_theory": 0.97,
            }
        else:
            row = {
                "internal_disk_ratio_mesh_over_theory": 0.85,
                "internal_outer_ratio_mesh_over_theory": 0.90,
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

    report = run_flat_disk_kh_region_parity(
        optimize_presets=("kh_strict_fast", "kh_strict_continuity"),
        rim_local_refine_steps=(1, 2),
        rim_local_refine_band_lambdas=(3.0,),
    )
    assert report["meta"]["mode"] == "flat_disk_kh_region_parity"
    assert len(report["rows"]) == 4
    for row in report["rows"]:
        assert np.isfinite(float(row["theta_star"]))
        assert np.isfinite(float(row["theta_factor"]))
        assert np.isfinite(float(row["energy_factor"]))
        assert np.isfinite(float(row["runtime_seconds"]))
        assert int(row["rim_local_refine_steps"]) >= 1
        assert np.isfinite(float(row["rim_local_refine_band_lambda"]))
        assert np.isfinite(float(row["internal_disk_ratio_mesh_over_theory"]))
        assert np.isfinite(float(row["internal_outer_ratio_mesh_over_theory"]))
        assert np.isfinite(float(row["region_parity_score"]))
        assert int(row["complexity_rank"]) >= 0
    assert report["selected_best"]["optimize_preset"] == "kh_strict_continuity"
