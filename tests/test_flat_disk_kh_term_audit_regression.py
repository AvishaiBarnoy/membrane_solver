import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_kh_term_audit import (
    run_flat_disk_kh_strict_preset_characterization,
    run_flat_disk_kh_strict_refinement_characterization,
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
        tilt_mass_mode_in="lumped",
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
        assert float(row["mesh_internal_total_from_regions"]) == pytest.approx(
            float(row["mesh_internal"]), rel=0.0, abs=1e-12
        )

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


@pytest.mark.regression
def test_flat_disk_kh_strict_refinement_characterization_emits_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_term_audit as audit_mod

    def _fake_run_flat_disk_one_leaflet_benchmark(**kwargs):
        refine_level = int(kwargs.get("refine_level", 1))
        rim_steps = int(kwargs.get("rim_local_refine_steps", 0))
        if refine_level == 1 and rim_steps == 1:
            return {
                "parity": {
                    "theta_factor": 1.10,
                    "energy_factor": 1.10,
                    "meets_factor_2": True,
                },
                "optimize": {"optimize_steps": 120, "optimize_inner_steps": 20},
            }
        return {
            "parity": {
                "theta_factor": 1.30,
                "energy_factor": 1.20,
                "meets_factor_2": True,
            },
            "optimize": {"optimize_steps": 120, "optimize_inner_steps": 20},
        }

    def _fake_run_flat_disk_kh_term_audit(**kwargs):
        rim_steps = int(kwargs.get("rim_local_refine_steps", 0))
        rim_h = 0.6 if rim_steps > 0 else 1.2
        return {"resolution": {"rim_h_over_lambda_median": rim_h}}

    monkeypatch.setattr(
        "tools.reproduce_flat_disk_one_leaflet.run_flat_disk_one_leaflet_benchmark",
        _fake_run_flat_disk_one_leaflet_benchmark,
    )
    monkeypatch.setattr(
        audit_mod, "run_flat_disk_kh_term_audit", _fake_run_flat_disk_kh_term_audit
    )
    monkeypatch.setattr(audit_mod, "perf_counter", lambda: 1.0)

    report = run_flat_disk_kh_strict_refinement_characterization(
        optimize_preset="kh_wide",
        global_refine_levels=(1,),
        rim_local_steps=(0, 1),
    )
    rows = report["rows"]
    assert len(rows) == 2
    for row in rows:
        assert int(row["refine_level"]) >= 1
        assert int(row["rim_local_refine_steps"]) >= 0
        assert np.isfinite(float(row["theta_factor"]))
        assert np.isfinite(float(row["energy_factor"]))
        assert np.isfinite(float(row["balanced_parity_score"]))
        assert np.isfinite(float(row["rim_h_over_lambda_median"]))
    best = report["selected_best"]
    assert np.isfinite(float(best["theta_factor"]))
    assert int(best["rim_local_refine_steps"]) == 1


@pytest.mark.regression
def test_flat_disk_kh_strict_preset_characterization_emits_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_term_audit as audit_mod

    def _fake_run_flat_disk_one_leaflet_benchmark(**kwargs):
        preset = str(kwargs.get("optimize_preset"))
        if preset == "kh_strict_fast":
            return {
                "parity": {
                    "theta_factor": 1.12,
                    "energy_factor": 1.10,
                    "meets_factor_2": True,
                },
                "optimize": {"optimize_steps": 30, "optimize_inner_steps": 14},
            }
        return {
            "parity": {
                "theta_factor": 1.10,
                "energy_factor": 1.10,
                "meets_factor_2": True,
            },
            "optimize": {"optimize_steps": 120, "optimize_inner_steps": 20},
        }

    monkeypatch.setattr(
        "tools.reproduce_flat_disk_one_leaflet.run_flat_disk_one_leaflet_benchmark",
        _fake_run_flat_disk_one_leaflet_benchmark,
    )
    monkeypatch.setattr(audit_mod, "perf_counter", lambda: 1.0)

    report = run_flat_disk_kh_strict_preset_characterization(
        optimize_presets=("kh_strict_refine", "kh_strict_fast"),
        refine_level=1,
        rim_local_refine_steps=1,
        rim_local_refine_band_lambda=4.0,
    )
    assert report["meta"]["mode"] == "strict_preset_characterization"
    rows = report["rows"]
    assert len(rows) == 2
    for row in rows:
        assert np.isfinite(float(row["theta_factor"]))
        assert np.isfinite(float(row["energy_factor"]))
        assert np.isfinite(float(row["balanced_parity_score"]))
        assert np.isfinite(float(row["runtime_seconds"]))
    best = report["selected_best"]
    assert best["optimize_preset"] == "kh_strict_refine"
