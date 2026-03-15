import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_kh_term_audit import (
    run_flat_disk_kh_outerfield_averaged_sweep,
)


@pytest.mark.regression
def test_flat_disk_kh_outerfield_averaged_sweep_v2_uses_v2_ratios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.diagnostics.flat_disk_kh_term_audit as audit_mod

    def _fake_run_flat_disk_kh_term_audit(**kwargs):
        refine_rmax = float(kwargs["outer_local_refine_rmax_lambda"])
        if refine_rmax >= 9.0:
            disk_ratio = 1.01
            near_ratio = 1.02
            far_ratio = 0.99
        else:
            disk_ratio = 1.20
            near_ratio = 1.30
            far_ratio = 1.10
        return {
            "rows": [
                {
                    "internal_disk_ratio_mesh_over_theory_v2": disk_ratio,
                    "internal_outer_near_ratio_mesh_over_theory_v2": near_ratio,
                    "internal_outer_far_ratio_mesh_over_theory_v2": far_ratio,
                    "internal_disk_ratio_mesh_over_theory": 1.5,
                    "internal_outer_near_ratio_mesh_over_theory_finite": 1.5,
                    "internal_outer_far_ratio_mesh_over_theory_finite": 1.5,
                    "section_score_internal_bands_finite_outer_l2_log": 10.0,
                }
            ]
        }

    monkeypatch.setattr(
        audit_mod, "run_flat_disk_kh_term_audit", _fake_run_flat_disk_kh_term_audit
    )

    report = run_flat_disk_kh_outerfield_averaged_sweep(
        ratio_version="v2",
        outer_local_refine_steps_values=(1,),
        outer_local_refine_rmin_lambda_values=(1.0,),
        outer_local_refine_rmax_lambda_values=(8.0, 9.0),
        outer_local_vertex_average_steps_values=(2,),
        outer_local_vertex_average_rmin_lambda_values=(4.0,),
        outer_local_vertex_average_rmax_lambda_values=(12.0,),
    )

    assert report["meta"]["ratio_version"] == "v2"
    best = report["selected_best"]
    assert float(best["outer_local_refine_rmax_lambda"]) == pytest.approx(9.0)
    assert float(best["internal_disk_ratio_mesh_over_theory_v2"]) == pytest.approx(1.01)
