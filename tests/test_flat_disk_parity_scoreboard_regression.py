from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_parity_scoreboard import (  # noqa: E402
    run_flat_disk_parity_scoreboard,
)


@pytest.mark.regression
def test_flat_disk_parity_scoreboard_locks_lane_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run_flat_disk_one_leaflet_benchmark(**kwargs):
        lane = str(kwargs.get("parameterization", "legacy"))
        if lane == "kh_physical":
            return {
                "meta": {
                    "theta_mode": "optimize",
                    "optimize_preset_effective": "kh_strict_fast",
                    "theory_model": "kh_physical_strict_kh",
                    "theory_source": "kh_physical_closed_form",
                },
                "mesh": {"theta_star": 0.132, "total_energy": -0.808},
                "parity": {
                    "theta_factor": 1.08,
                    "energy_factor": 1.11,
                    "meets_factor_2": True,
                },
            }
        return {
            "meta": {
                "theta_mode": "scan",
                "optimize_preset_effective": "none",
                "theory_model": "legacy_scalar_reduced",
                "theory_source": "docs/tex/1_disk_flat.tex",
            },
            "mesh": {"theta_star": 6.36e-4, "total_energy": -1.09e-4},
            "parity": {
                "theta_factor": 1.20,
                "energy_factor": 1.30,
                "meets_factor_2": True,
            },
        }

    monkeypatch.setattr(
        "tools.reproduce_flat_disk_one_leaflet.run_flat_disk_one_leaflet_benchmark",
        _fake_run_flat_disk_one_leaflet_benchmark,
    )

    report = run_flat_disk_parity_scoreboard()
    assert report["meta"]["mode"] == "flat_disk_parity_scoreboard"
    lanes = report["lanes"]
    assert set(lanes.keys()) == {"legacy", "kh_physical"}

    lock = report["meta"]["reference_lock"]
    assert lock["legacy"]["theory_model"] == "legacy_scalar_reduced"
    assert lock["kh_physical"]["theory_model"] == "kh_physical_strict_kh"

    for lane in ("legacy", "kh_physical"):
        row = lanes[lane]
        assert np.isfinite(float(row["theta_factor"]))
        assert np.isfinite(float(row["energy_factor"]))
        assert np.isfinite(float(row["balanced_parity_score"]))
        assert np.isfinite(float(row["theta_star"]))
        assert np.isfinite(float(row["total_energy"]))
        assert isinstance(bool(row["meets_factor_2"]), bool)
