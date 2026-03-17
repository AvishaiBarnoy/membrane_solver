import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.reproduce_flat_disk_one_leaflet import run_flat_disk_one_leaflet_benchmark


def _run_curved_lane_from_fixture(fixture_name: str) -> dict:
    cfg = yaml.safe_load(
        (ROOT / "tests" / "fixtures" / fixture_name).read_text(encoding="utf-8")
    )
    meta = cfg["meta"]
    expected = cfg["expected"]
    observability = cfg["observability"]

    report = run_flat_disk_one_leaflet_benchmark(
        fixture=ROOT / str(meta["fixture"]),
        refine_level=int(meta["refine_level"]),
        outer_mode=str(meta["outer_mode"]),
        smoothness_model=str(meta["smoothness_model"]),
        parameterization=str(meta["parameterization"]),
        theta_mode=str(meta["theta_mode"]),
        theta_initial=float(meta["theta_initial"]),
        theta_optimize_steps=int(meta["theta_optimize_steps"]),
        theta_optimize_every=int(meta["theta_optimize_every"]),
        theta_optimize_delta=float(meta["theta_optimize_delta"]),
        theta_optimize_inner_steps=int(meta["theta_optimize_inner_steps"]),
        theta_optimize_mode=str(meta.get("theta_optimize_mode", "scalar_local")),
        kappa_physical=float(meta["kappa_physical"]),
        kappa_t_physical=float(meta["kappa_t_physical"]),
        length_scale_nm=float(meta["length_scale_nm"]),
        radius_nm=float(meta["radius_nm"]),
        drive_physical=float(meta["drive_physical"]),
        geometry_lane=str(meta["geometry_lane"]),
        z_gauge=str(meta["z_gauge"]),
        curved_acceptance_profile=str(meta.get("curved_acceptance_profile", "full")),
        curved_theta_calibration_mode=str(
            meta.get("curved_theta_calibration_mode", "off")
        ),
        curved_theta_calibration_inner_scale=float(
            meta.get("curved_theta_calibration_inner_scale", 1.0)
        ),
        curved_theta_calibration_outer_scale=float(
            meta.get("curved_theta_calibration_outer_scale", 1.0)
        ),
        curved_theta_calibration_contact_scale=float(
            meta.get("curved_theta_calibration_contact_scale", 1.0)
        ),
    )

    assert report["meta"]["theory_source"] == str(meta["theory_reference"])
    for field in observability["required_meta_fields"]:
        assert field in report["meta"]
    for field in observability["required_curvature_fields"]:
        assert field in report["diagnostics"]["curvature"]
    for field in observability["required_transfer_fields"]:
        assert field in report["diagnostics"]["tilt_transfer"]

    boundary = report["parity"]["boundary_at_R"]
    assert boundary["theory_model"] == "small_slope_half_split_proxy"
    assert bool(boundary["available"])
    assert int(boundary["sample_count"]) > 0
    assert boundary["disk_source"] == "disk_boundary_group"
    assert boundary["rim_source"] == "first_shell_outside_disk"
    assert boundary["outer_source"] == "second_shell_outside_disk"
    assert int(boundary["disk_count"]) > 0
    assert int(boundary["rim_count"]) > 0
    assert int(boundary["outer_count"]) > 0
    assert float(boundary["rim_radius"]) > float(boundary["disk_radius"])
    assert float(boundary["outer_radius"]) > float(boundary["rim_radius"])
    assert float(boundary["kink_angle_theory"]) > 0.0
    assert float(boundary["tilt_in_theory"]) > 0.0
    assert float(boundary["tilt_out_theory"]) > 0.0
    assert float(boundary["kink_angle_factor"]) > 0.0
    assert float(boundary["tilt_in_factor"]) > 0.0
    assert float(boundary["tilt_out_factor"]) > 0.0

    calibration = report["diagnostics"]["curved_theta_calibration"]
    assert float(calibration["theta_star_raw"]) == pytest.approx(
        float(expected["theta_star"]), abs=float(expected["theta_star_abs_tol"])
    )
    assert float(calibration["total_energy_raw"]) == pytest.approx(
        float(expected["total_energy"]), abs=float(expected["total_energy_abs_tol"])
    )
    assert float(calibration["theta_factor_raw"]) <= float(expected["theta_factor_max"])
    assert float(calibration["energy_factor_raw"]) <= float(
        expected["energy_factor_max"]
    )
    return report


@pytest.mark.acceptance
def test_flat_disk_curved_lane_matches_3d_updated_signs_p10_target() -> None:
    report = _run_curved_lane_from_fixture(
        "flat_disk_one_leaflet_kh_physical_curved_p10_target.yaml"
    )
    expected = yaml.safe_load(
        (
            ROOT
            / "tests"
            / "fixtures"
            / "flat_disk_one_leaflet_kh_physical_curved_p10_target.yaml"
        ).read_text(encoding="utf-8")
    )["expected"]
    calibration = report["diagnostics"]["curved_theta_calibration"]
    assert float(calibration["theta_factor_raw"]) <= float(expected["theta_factor_max"])
    assert float(calibration["energy_factor_raw"]) <= float(
        expected["energy_factor_max"]
    )
