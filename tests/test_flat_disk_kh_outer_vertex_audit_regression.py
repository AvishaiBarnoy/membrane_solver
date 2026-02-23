import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_kh_outer_vertex_audit import (
    run_flat_disk_kh_outer_vertex_audit,
)


@pytest.mark.regression
def test_flat_disk_kh_outer_vertex_audit_reports_finite_bands() -> None:
    report = run_flat_disk_kh_outer_vertex_audit(
        optimize_preset="kh_strict_outertail_balanced",
        theta=0.138,
    )
    controls = report["meta"]["controls_effective"]
    assert int(controls["refine_level"]) == 2
    assert int(controls["outer_local_refine_steps"]) == 1
    assert float(controls["outer_local_refine_rmax_lambda"]) == pytest.approx(10.0)

    assert np.isfinite(float(report["parity"]["outer_tail_balance_score"]))
    assert len(report["bands"]) == 2
    for row in report["bands"]:
        assert int(row["vertex_count"]) > 0
        assert np.isfinite(float(row["vertex_density_per_dual_area"]))
        assert np.isfinite(float(row["t_phi_over_t_rad_median"]))
