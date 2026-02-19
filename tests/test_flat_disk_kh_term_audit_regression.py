import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_kh_term_audit import (
    run_flat_disk_kh_term_audit,
    run_flat_disk_kh_term_audit_refine_sweep,
)


@pytest.mark.regression
def test_flat_disk_kh_term_audit_reports_finite_rows() -> None:
    report = run_flat_disk_kh_term_audit(
        refine_level=1,
        outer_mode="disabled",
        smoothness_model="splay_twist",
        theta_values=(0.0, 6.366e-4, 0.004),
    )

    assert report["meta"]["parameterization"] == "kh_physical"
    rows = report["rows"]
    assert len(rows) == 3

    for row in rows:
        assert np.isfinite(float(row["theta"]))
        assert np.isfinite(float(row["mesh_total"]))
        assert np.isfinite(float(row["mesh_contact"]))
        assert np.isfinite(float(row["mesh_internal"]))
        assert np.isfinite(float(row["theory_total"]))
        assert np.isfinite(float(row["theory_contact"]))
        assert np.isfinite(float(row["theory_internal"]))
        assert np.isfinite(float(row["total_error"]))
        assert np.isfinite(float(row["contact_error"]))
        assert np.isfinite(float(row["internal_error"]))
        assert np.isfinite(float(row["mesh_internal_disk"]))
        assert np.isfinite(float(row["mesh_internal_outer"]))
        assert np.isfinite(float(row["theory_internal_disk"]))
        assert np.isfinite(float(row["theory_internal_outer"]))
        assert np.isfinite(float(row["internal_disk_error"]))
        assert np.isfinite(float(row["internal_outer_error"]))
        assert np.isfinite(float(row["inner_tphi_over_trad_median"]))
        assert np.isfinite(float(row["outer_tphi_over_trad_median"]))

    theta0 = rows[0]
    assert float(theta0["theta"]) == pytest.approx(0.0, abs=1e-18)
    assert float(theta0["theory_contact"]) == pytest.approx(0.0, abs=1e-18)
    assert float(theta0["mesh_contact"]) == pytest.approx(0.0, abs=1e-15)
    assert int(theta0["rim_samples"]) > 0

    resolution = report["resolution"]
    assert int(resolution["rim_edge_count"]) > 0
    assert np.isfinite(float(resolution["rim_h_over_lambda_median"]))


@pytest.mark.regression
def test_flat_disk_kh_term_audit_refine_sweep_shape() -> None:
    report = run_flat_disk_kh_term_audit_refine_sweep(
        refine_levels=(1, 2),
        theta_values=(0.0, 6.366e-4),
    )
    assert report["meta"]["mode"] == "refine_sweep"
    runs = report["runs"]
    assert len(runs) == 2
    for idx, run in enumerate(runs):
        assert int(run["meta"]["refine_level"]) == (idx + 1)
        assert run["meta"]["theory_model"] == "kh_physical_strict_kh"
        assert len(run["rows"]) == 2


@pytest.mark.regression
def test_flat_disk_kh_term_audit_local_rim_refine_changes_resolution() -> None:
    theta = 0.004
    base = run_flat_disk_kh_term_audit(
        refine_level=1,
        theta_values=(theta,),
        rim_local_refine_steps=0,
        rim_local_refine_band_lambda=0.0,
    )
    local = run_flat_disk_kh_term_audit(
        refine_level=1,
        theta_values=(theta,),
        rim_local_refine_steps=1,
        rim_local_refine_band_lambda=4.0,
    )

    base_row = base["rows"][0]
    local_row = local["rows"][0]
    assert np.isfinite(float(base_row["internal_outer_ratio_mesh_over_theory"]))
    assert np.isfinite(float(local_row["internal_outer_ratio_mesh_over_theory"]))
    base_h = float(base["resolution"]["rim_h_over_lambda_median"])
    local_h = float(local["resolution"]["rim_h_over_lambda_median"])
    assert local_h < base_h
