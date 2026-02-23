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
        include_frozen_analytic=True,
    )
    assert bool(report["meta"]["include_frozen_analytic"]) is True
    assert report["meta"]["outer_reference_primary"] == "infinite"
    assert report["meta"]["outer_reference_secondary"] == "finite_outer_rmax"
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

    for field in ("solved", "radial_only", "frozen_analytic"):
        rows = report["bands_by_field"][field]
        assert len(rows) == 2
        for row in rows:
            assert int(row["vertex_count"]) > 0
            assert np.isfinite(float(row["vertex_density_per_dual_area"]))
            assert np.isfinite(float(row["t_phi_over_t_rad_median"]))

        sections = report["section_energy_by_field"][field]
        for name in ("disk_total", "outer_near", "outer_far"):
            sec = sections[name]
            assert np.isfinite(float(sec["mesh"]))
            assert np.isfinite(float(sec["theory"]))
            assert np.isfinite(float(sec["ratio_mesh_over_theory"]))
        sections_finite = report["section_energy_by_field_finite_outer_reference"][
            field
        ]
        for name in ("disk_total", "outer_near", "outer_far"):
            sec = sections_finite[name]
            assert np.isfinite(float(sec["mesh"]))
            assert np.isfinite(float(sec["theory"]))
            assert np.isfinite(float(sec["ratio_mesh_over_theory"]))
