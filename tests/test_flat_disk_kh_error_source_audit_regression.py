import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_kh_error_source_audit import (
    _rank_effects,
    run_flat_disk_kh_error_source_audit,
    run_flat_disk_kh_error_source_candidate_bakeoff,
)


@pytest.mark.unit
def test_flat_disk_kh_error_source_rank_effects_orders_descending() -> None:
    ranked = _rank_effects(
        {
            "partition_effect": 0.3,
            "mass_effect": 0.1,
            "resolution_effect": 0.2,
            "operator_effect": 0.05,
        }
    )
    assert ranked["dominant_source"] == "partition_effect"
    assert ranked["ranking"][0] == "partition_effect"
    assert 0.0 < float(ranked["confidence"]) <= 1.0


@pytest.mark.regression
def test_flat_disk_kh_error_source_audit_emits_finite_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_error_source_audit as mod

    def _fake_benchmark(**kwargs):
        refine = int(kwargs["refine_level"])
        preset = str(kwargs["optimize_preset"])
        return {
            "mesh": {
                "theta_star": 0.10
                + 0.01 * refine
                + (0.01 if "outerband" in preset else 0.0)
            }
        }

    def _fake_term_audit(**kwargs):
        refine = int(kwargs["refine_level"])
        mass = str(kwargs["tilt_mass_mode_in"])
        part = str(kwargs.get("partition_mode", "centroid"))
        near = 1.10 + 0.02 * (refine - 2)
        far = 0.95 - (0.01 if mass == "lumped" else 0.0)
        score = float(np.hypot(np.log(near), np.log(max(far, 1e-18))))
        if part == "fractional":
            score *= 0.9
        return {
            "rows": [
                {
                    "internal_disk_ratio_mesh_over_theory": 1.1,
                    "internal_outer_near_ratio_mesh_over_theory_finite": near,
                    "internal_outer_far_ratio_mesh_over_theory_finite": far,
                    "section_score_internal_bands_finite_outer_l2_log": score,
                    "proj_radial_internal_outer_near_abs_error_delta_vs_unprojected": 0.03,
                    "proj_radial_internal_outer_far_abs_error_delta_vs_unprojected": 0.02,
                }
            ]
        }

    monkeypatch.setattr(mod, "run_flat_disk_one_leaflet_benchmark", _fake_benchmark)
    monkeypatch.setattr(mod, "run_flat_disk_kh_term_audit", _fake_term_audit)

    report = run_flat_disk_kh_error_source_audit(
        refine_levels=(2, 3),
        mass_modes=("consistent", "lumped"),
        partition_modes=("centroid", "fractional"),
    )
    assert report["meta"]["mode"] == "kh_error_source_audit"
    assert len(report["runs"]) == 16
    assert report["meta"]["partition_modes"] == ["centroid", "fractional"]
    for row in report["runs"]:
        assert np.isfinite(
            float(row["section_score_internal_bands_finite_outer_l2_log"])
        )
        assert row["partition_mode"] in {"centroid", "fractional"}
    assert report["attribution"]["dominant_source"] in {
        "partition_effect",
        "mass_effect",
        "resolution_effect",
        "operator_effect",
    }
    assert float(report["attribution"]["effect_sizes"]["partition_effect"]) > 0.0


@pytest.mark.regression
def test_flat_disk_kh_error_source_candidate_bakeoff_selects_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_error_source_audit as mod

    def _fake_benchmark(**kwargs):
        preset = str(kwargs["optimize_preset"])
        if "outerfield_averaged" in preset:
            return {"parity": {"theta_factor": 1.06, "energy_factor": 1.10}}
        if "outerband" in preset:
            return {"parity": {"theta_factor": 1.09, "energy_factor": 1.14}}
        return {"parity": {"theta_factor": 1.12, "energy_factor": 1.18}}

    def _fake_error_source_audit(**kwargs):
        preset = str(kwargs["primary_preset"])
        if "outerfield_averaged" in preset:
            near, far, part = 1.04, 0.98, 0.02
        elif "outerband" in preset:
            near, far, part = 1.08, 0.96, 0.03
        else:
            near, far, part = 1.12, 0.92, 0.04
        return {
            "runs": [
                {"outer_near_ratio": near, "outer_far_ratio": far},
                {"outer_near_ratio": near, "outer_far_ratio": far},
            ],
            "attribution": {
                "dominant_source": "partition_effect",
                "effect_sizes": {
                    "partition_effect": part,
                    "mass_effect": 0.01,
                    "resolution_effect": 0.02,
                    "operator_effect": 0.03,
                },
            },
        }

    monkeypatch.setattr(mod, "run_flat_disk_one_leaflet_benchmark", _fake_benchmark)
    monkeypatch.setattr(
        mod, "run_flat_disk_kh_error_source_audit", _fake_error_source_audit
    )

    report = run_flat_disk_kh_error_source_candidate_bakeoff(
        optimize_presets=(
            "kh_strict_outerfield_tight",
            "kh_strict_outerband_tight",
            "kh_strict_outerfield_averaged",
        ),
        refine_level=2,
    )
    assert report["meta"]["mode"] == "kh_error_source_candidate_bakeoff"
    rows = report["candidates"]
    assert len(rows) == 3
    for row in rows:
        assert np.isfinite(float(row["outer_section_score"]))
        assert np.isfinite(float(row["balanced_parity_score"]))
        assert np.isfinite(float(row["runtime_seconds"]))
    assert report["selected_best"]["optimize_preset"] == "kh_strict_outerfield_averaged"
