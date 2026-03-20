import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.diagnostics.flat_disk_curved_3d_audit import run_flat_disk_curved_3d_audit


@pytest.mark.acceptance
def test_flat_disk_curved_3d_audit_smoke_report_is_finite() -> None:
    cfg = yaml.safe_load(
        (
            ROOT / "tests" / "fixtures" / "flat_disk_curved_3d_audit_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    meta = cfg["meta"]
    expected = cfg["expected"]

    report = run_flat_disk_curved_3d_audit(
        fixture=ROOT / str(meta["fixture"]),
        refine_level=int(meta["refine_level"]),
        outer_mode=str(meta["outer_mode"]),
        smoothness_model=str(meta["smoothness_model"]),
        theta_mode=str(meta["theta_mode"]),
        theta_initial=float(meta["theta_initial"]),
        theta_optimize_steps=int(meta["theta_optimize_steps"]),
        theta_optimize_every=int(meta["theta_optimize_every"]),
        theta_optimize_delta=float(meta["theta_optimize_delta"]),
        theta_optimize_inner_steps=int(meta["theta_optimize_inner_steps"]),
        kappa_physical=float(meta["kappa_physical"]),
        kappa_t_physical=float(meta["kappa_t_physical"]),
        length_scale_nm=float(meta["length_scale_nm"]),
        radius_nm=float(meta["radius_nm"]),
        drive_physical=float(meta["drive_physical"]),
        z_gauge=str(meta["z_gauge"]),
        curved_acceptance_profile=str(meta["curved_acceptance_profile"]),
        curved_theta_objective_ablation_mode=str(
            meta.get("curved_theta_objective_ablation_mode", "off")
        ),
        curved_theta_objective_ablation_inner_scale=float(
            meta.get("curved_theta_objective_ablation_inner_scale", 1.0)
        ),
        curved_theta_objective_ablation_outer_scale=float(
            meta.get("curved_theta_objective_ablation_outer_scale", 1.0)
        ),
        curved_theta_objective_ablation_contact_scale=float(
            meta.get("curved_theta_objective_ablation_contact_scale", 1.0)
        ),
        include_sections=bool(meta.get("include_sections", True)),
    )

    assert report["meta"]["mode"] == "curved_3d_audit_smoke"
    assert report["meta"]["theory_source"] == str(expected["theory_source"])
    assert report["meta"]["geometry_lane"] == str(expected["geometry_lane"])
    assert bool(report["meta"]["sections_requested"]) is True
    assert bool(report["meta"]["sections_available"]) is False

    for field in (
        "theta_star_mesh",
        "theta_star_theory",
        "theta_factor",
        "total_energy_mesh",
        "total_energy_theory",
        "energy_factor",
    ):
        assert np.isfinite(float(report["parity"][field]))

    for field in ("h_mean", "h_p95", "h_max"):
        assert np.isfinite(float(report["curvature"][field]))

    ablation = report["ablation"]
    assert bool(ablation["available"]) is True
    assert bool(ablation["applied"]) is True
    assert ablation["reason"] == "ok"
    assert ablation["mode"] == "inner_outer_rescaled"
    assert float(ablation["inner_scale"]) == pytest.approx(
        float(meta["curved_theta_objective_ablation_inner_scale"])
    )
    assert float(ablation["outer_scale"]) == pytest.approx(
        float(meta["curved_theta_objective_ablation_outer_scale"])
    )
    assert float(ablation["contact_scale"]) == pytest.approx(
        float(meta["curved_theta_objective_ablation_contact_scale"])
    )

    for field in (
        "theta_star_pred",
        "total_energy_pred",
        "theta_factor_pred",
        "energy_factor_pred",
        "coeff_a_inner_raw",
        "coeff_a_outer_raw",
        "coeff_b_contact_raw",
        "coeff_a_effective",
        "coeff_b_effective",
        "theta_fit_local",
    ):
        assert np.isfinite(float(ablation[field]))

    boundary = report["boundary_at_R"]
    assert boundary is not None
    assert boundary["theory_model"] == "small_slope_half_split_proxy"
    assert bool(boundary["available"]) is True
    assert int(boundary["sample_count"]) > 0
    assert boundary["disk_source"] == "disk_boundary_group"
    assert boundary["rim_source"] == "first_shell_outside_disk"
    assert boundary["outer_source"] == "second_shell_outside_disk"
    assert int(boundary["disk_count"]) > 0
    assert int(boundary["rim_count"]) > 0
    assert int(boundary["outer_count"]) > 0
    assert np.isfinite(float(boundary["disk_radius"]))
    assert np.isfinite(float(boundary["rim_radius"]))
    assert np.isfinite(float(boundary["outer_radius"]))
    assert float(boundary["rim_radius"]) > float(boundary["disk_radius"])
    assert float(boundary["outer_radius"]) > float(boundary["rim_radius"])

    for field in (
        "kink_angle_mesh",
        "kink_angle_theory",
        "tilt_in_mesh",
        "tilt_in_theory",
        "tilt_out_mesh",
        "tilt_out_theory",
    ):
        assert np.isfinite(float(boundary[field]))

    for field in ("kink_angle_factor", "tilt_in_factor", "tilt_out_factor"):
        value = float(boundary[field])
        assert value >= 1.0 or np.isinf(value)
