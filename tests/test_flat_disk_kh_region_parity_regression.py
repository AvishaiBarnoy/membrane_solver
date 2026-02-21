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
        if preset == "kh_strict_continuity":
            return {
                "meta": {
                    "refine_level": 1,
                    "rim_local_refine_steps": 2,
                    "rim_local_refine_band_lambda": 4.0,
                },
                "mesh": {"theta_star": 0.136},
                "parity": {"theta_factor": 1.04, "energy_factor": 1.08},
            }
        return {
            "meta": {
                "refine_level": 1,
                "rim_local_refine_steps": 1,
                "rim_local_refine_band_lambda": 4.0,
            },
            "mesh": {"theta_star": 0.132},
            "parity": {"theta_factor": 1.08, "energy_factor": 1.11},
        }

    def _fake_run_flat_disk_kh_term_audit(**kwargs):
        theta = float(kwargs["theta_values"][0])
        if abs(theta - 0.136) < 1e-12:
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
    )
    assert report["meta"]["mode"] == "flat_disk_kh_region_parity"
    assert len(report["rows"]) == 2
    for row in report["rows"]:
        assert np.isfinite(float(row["theta_star"]))
        assert np.isfinite(float(row["theta_factor"]))
        assert np.isfinite(float(row["energy_factor"]))
        assert np.isfinite(float(row["internal_disk_ratio_mesh_over_theory"]))
        assert np.isfinite(float(row["internal_outer_ratio_mesh_over_theory"]))
        assert np.isfinite(float(row["region_parity_score"]))
    assert report["selected_best"]["optimize_preset"] == "kh_strict_continuity"
