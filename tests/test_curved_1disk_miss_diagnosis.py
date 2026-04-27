import pytest

from tools.diagnostics.curved_1disk_miss_diagnosis import (
    run_curved_1disk_miss_diagnosis,
)


def _benchmark_report() -> dict[str, object]:
    return {
        "benchmark_lock_passed": False,
        "benchmark_lock_failures": [
            "theta_B_opt",
            "theta_out_half_split",
            "inner_i1_fit",
            "outer_k1_fit",
            "outer_height_log_fit",
            "total_energy",
            "inner_elastic",
            "outer_elastic",
            "contact_energy",
        ],
        "theta_B_selected": 0.06,
        "theory": {
            "radius": 7.0 / 15.0,
            "lambda_theory": 15.0,
            "theta_B_opt": 0.1845693593,
            "F_in_el": 0.8094230804,
            "F_out_el": 0.3503377181,
            "F_cont": -2.3195215970,
            "F_tot": -1.1597607985,
        },
        "energies": {
            "inner_elastic_numeric": 0.0490795294,
            "outer_elastic_numeric": 1.2871398562,
            "contact_numeric": -0.7508070761,
            "total_numeric": 0.0196715731,
        },
        "near_rim": {
            "theta_out_over_half_theta_B": 0.5576009832,
        },
        "fits": {
            "inner_i1": {
                "lambda_fit": 115.0452828735,
                "lambda_ratio": 7.6696855249,
                "rel_rmse": 0.7052065017,
            },
            "outer_k1": {
                "lambda_fit": 10.0016445247,
                "lambda_ratio": 0.6667763016,
                "leaflet_mismatch_median": 0.0001856232,
            },
            "outer_height_log": {
                "slope_fit": 0.0,
                "slope_ratio": 0.0,
                "rel_rmse": 0.0,
            },
        },
        "shell_rows": [
            {"radius": 0.519902, "z": 0.0016},
            {"radius": 0.524333, "z": 0.0},
            {"radius": 1.491794, "z": 0.0},
            {"radius": 2.312, "z": 0.0},
            {"radius": 4.618667, "z": 0.0},
        ],
    }


def test_curved_1disk_miss_diagnosis_explains_failure_groups() -> None:
    report = run_curved_1disk_miss_diagnosis(
        benchmark_report=_benchmark_report(),
        phi_target_audit={
            "diagnosis": {"call": "another specific target-construction defect"},
            "shell_target_construction": {
                "target_direction": {
                    "r_dir_cos_global_radial_median": -0.98,
                },
            },
        },
        shell2_audit={
            "first_material_departure": {
                "call": "theta_out_radial",
                "shell_radius": 0.524333,
            },
        },
        control_volume_rows=[
            {
                "theta_b": 0.14,
                "outer_control_area": 0.12,
                "outer_annulus_area": 0.02,
                "rim_control_area": 0.08,
                "rim_annulus_area": 0.03,
                "outer_shell_area": 0.12,
                "rim_shell_area": 0.08,
            }
        ],
        include_target_audits=False,
        include_control_volume=False,
    )

    assert report["scope"]["diagnosis_only"] is True
    assert report["scope"]["runtime_physics_changed"] is False
    assert report["benchmark_summary"]["theta_B_selected"] == pytest.approx(0.06)

    failures = report["failure_explanations"]
    assert set(failures) == {
        "theta_B_opt",
        "outer_height_log_fit",
        "outer_k1_fit",
        "inner_i1_fit",
        "energy_split",
    }
    assert failures["outer_height_log_fit"]["shape_propagation_call"] == (
        "height confined to local support shell"
    )
    assert (
        failures["energy_split"]["outer_elastic_numeric_over_tex_at_selected_theta"]
        > 30.0
    )
    assert failures["energy_split"][
        "contact_numeric_over_tex_at_selected_theta"
    ] == pytest.approx(1.0, rel=0.02)

    ranked = report["candidate_causes_ranked"]
    assert {row["cause"] for row in ranked} == {
        "curvature generation does not propagate",
        "excess shared-rim/local-shell elastic cost",
        "wrong rim/shell target direction or shell-2 continuation",
    }
    assert ranked[0]["rank_score"] >= ranked[-1]["rank_score"]
    assert any(
        row["evidence"].get("call") == "target radial direction points inward"
        for row in ranked
    )
    assert report["recommended_next_pr"]["feature_contract_required"] is True
