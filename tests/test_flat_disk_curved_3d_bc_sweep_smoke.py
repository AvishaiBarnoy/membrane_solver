import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.diagnostics.flat_disk_curved_3d_bc_sweep import (
    run_flat_disk_curved_3d_bc_sweep,
)


def test_flat_disk_curved_3d_bc_sweep_returns_ranked_candidates() -> None:
    sweep_cfg = yaml.safe_load(
        (
            ROOT / "tests" / "fixtures" / "flat_disk_curved_3d_bc_sweep_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    report = run_flat_disk_curved_3d_bc_sweep(
        fixture=ROOT
        / "tests"
        / "fixtures"
        / "kozlov_1disk_3d_free_disk_theory_parity.yaml",
        sweep=sweep_cfg,
    )

    assert report["meta"]["mode"] == "curved_3d_bc_sweep_smoke"
    assert int(report["meta"]["candidate_count"]) == 1
    assert int(report["meta"]["ok_count"]) == 1
    assert int(report["meta"]["failed_count"]) == 0
    assert isinstance(report["best_candidate"], dict)
    assert len(report["ranked_candidates"]) == 1

    row = report["ranked_candidates"][0]
    assert row["status"] == "ok"
    assert np.isfinite(float(row["score"]))
    assert row["dominant_metric"] in {"kink_angle", "tilt_in", "tilt_out"}
    assert np.isfinite(float(row["dominant_penalty"]))
    assert np.isfinite(float(row["theta_factor"]))
    assert np.isfinite(float(row["energy_factor"]))
    assert isinstance(row["boundary_available"], bool)
