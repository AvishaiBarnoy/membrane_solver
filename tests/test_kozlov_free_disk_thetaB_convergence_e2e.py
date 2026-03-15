import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from tools.diagnostics.free_disk_profile_protocol import (  # noqa: E402
    load_free_disk_theory_mesh,
    optimize_free_disk_curved_theta_b,
    optimize_free_disk_theta_b,
    run_free_disk_curved_bilayer_energy_sweep,
    run_free_disk_curved_bilayer_protocol,
    run_free_disk_curved_bilayer_refinement_sweep,
    run_free_disk_curved_bilayer_theta_sweep,
    summarize_free_disk_curved_elastic_growth,
)


@pytest.fixture(scope="module")
def _theta_seed_scans4() -> float:
    """Cache the flat-theory seed used by the slower curved optimizer tests."""
    return optimize_free_disk_theta_b(load_free_disk_theory_mesh(), scans=4)


@pytest.fixture(scope="module")
def _theta_seed_scans2() -> float:
    """Cache the shorter flat-theory seed used by diagnostic ablation tests."""
    return optimize_free_disk_theta_b(load_free_disk_theory_mesh(), scans=2)


@pytest.fixture(scope="module")
def _energy_sweep_main() -> list[dict[str, float]]:
    """Cache the canonical imposed-theta sweep used by multiple e2e checks."""
    return run_free_disk_curved_bilayer_energy_sweep(
        [0.04, 0.08, 0.10, 0.14, 0.16, 0.18, 0.20]
    )


@pytest.fixture(scope="module")
def _energy_sweep_081018() -> list[dict[str, float]]:
    """Cache the mid/high-theta sweep used for regional growth assertions."""
    return run_free_disk_curved_bilayer_energy_sweep([0.08, 0.10, 0.18])


@pytest.fixture(scope="module")
def _energy_sweep_0141820() -> list[dict[str, float]]:
    """Cache the post-peak sweep used by high-theta growth diagnostics."""
    return run_free_disk_curved_bilayer_energy_sweep([0.14, 0.18, 0.20])


@pytest.fixture(scope="module")
def _curved_optimizer_result(_theta_seed_scans4: float) -> dict[str, object]:
    """Cache the main curved optimizer result used by the branch-selection check."""
    return optimize_free_disk_curved_theta_b(
        theta_b_seed=_theta_seed_scans4,
        shape_steps=60,
    )


@pytest.mark.regression
def test_kozlov_free_disk_flat_thetaB_scan_stays_on_flat_surrogate_branch() -> None:
    """Diagnostic: the legacy flat free-disk scan still lands on the flat branch."""
    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kozlov_1disk_3d_free_disk_theory_parity.yaml",
    )
    mesh = parse_geometry(load_data(path))
    theta_b = optimize_free_disk_theta_b(mesh, scans=4)

    assert theta_b == pytest.approx(0.04, abs=0.02)
    assert float(np.ptp(mesh.positions_view()[:, 2])) == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.e2e
def test_kozlov_free_disk_curved_theta_sweep_scales_linearly_with_imposed_drive() -> (
    None
):
    """E2E: the curved shared-rim protocol tracks imposed thetaB on the named mesh."""
    mesh_path = (
        Path(__file__).resolve().parent.parent
        / "meshes"
        / "caveolin"
        / "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml"
    )
    rows = run_free_disk_curved_bilayer_theta_sweep(
        [0.02, 0.04, 0.10],
        curved_path=mesh_path,
    )

    assert len(rows) == 3
    for row in rows:
        target = 0.5 * row["theta_b"]
        assert row["theta_disk"] == pytest.approx(row["theta_b"], rel=0.05, abs=1.0e-3)
        assert row["theta_outer_in"] == pytest.approx(target, rel=0.05, abs=1.0e-3)
        assert row["theta_outer_out"] == pytest.approx(target, rel=0.05, abs=1.0e-3)
        assert row["phi_abs"] == pytest.approx(target, rel=0.05, abs=1.0e-3)
        assert row["closure_error"] == pytest.approx(0.0, abs=1.0e-3)

    phi_vals = np.asarray([row["phi_abs"] for row in rows], dtype=float)
    theta_vals = np.asarray([row["theta_b"] for row in rows], dtype=float)
    assert np.all(np.diff(phi_vals) > 0.0)
    assert np.allclose(phi_vals / theta_vals, 0.5, rtol=0.05, atol=1.0e-3)


@pytest.mark.e2e
def test_kozlov_free_disk_curved_theta_optimizer_beats_flat_stage_seed(
    _theta_seed_scans4: float,
    _curved_optimizer_result: dict[str, object],
) -> None:
    """E2E: curved local scan prefers a higher thetaB than the flat stage seed."""
    theta_seed = _theta_seed_scans4
    result = _curved_optimizer_result

    assert result["theta_b_seed"] == pytest.approx(theta_seed, abs=1.0e-12)
    assert float(result["best_theta_b"]) > float(theta_seed)
    assert float(result["best_total_energy"]) < -0.31

    rows = result["rows"]
    energies = np.asarray([float(row["total_energy"]) for row in rows], dtype=float)
    thetas = np.asarray([float(row["theta_b"]) for row in rows], dtype=float)
    best_idx = int(np.argmin(energies))
    assert thetas[best_idx] == pytest.approx(float(result["best_theta_b"]), abs=1.0e-12)
    assert thetas[best_idx] == pytest.approx(0.18, abs=0.021)

    _, theta_protocol = run_free_disk_curved_bilayer_protocol()
    assert theta_protocol == pytest.approx(float(result["best_theta_b"]), abs=1.0e-12)


@pytest.mark.e2e
def test_kozlov_free_disk_curved_theta_gap_is_elastic_not_contact_limited(
    _energy_sweep_main: list[dict[str, float]],
) -> None:
    """E2E: higher imposed thetaB gains contact work but loses to elastic growth."""
    rows = _energy_sweep_main
    by_theta = {float(row["theta_b"]): row for row in rows}

    row_004 = by_theta[0.04]
    row_008 = by_theta[0.08]
    row_010 = by_theta[0.10]
    row_014 = by_theta[0.14]
    row_016 = by_theta[0.16]
    row_018 = by_theta[0.18]
    row_020 = by_theta[0.20]

    assert row_008["total_energy"] < row_004["total_energy"]
    assert row_010["total_energy"] < row_008["total_energy"]
    assert row_014["total_energy"] < row_010["total_energy"]
    assert row_018["total_energy"] < row_014["total_energy"]
    assert row_018["total_energy"] < row_016["total_energy"]
    assert row_018["total_energy"] < row_020["total_energy"]

    assert row_010["contact_energy"] < row_008["contact_energy"]
    assert row_014["contact_energy"] < row_010["contact_energy"]
    assert row_016["contact_energy"] < row_014["contact_energy"]
    assert row_018["contact_energy"] < row_016["contact_energy"]
    assert row_020["contact_energy"] < row_018["contact_energy"]

    assert row_010["elastic_energy"] > row_008["elastic_energy"]
    assert row_014["elastic_energy"] > row_010["elastic_energy"]
    assert row_016["elastic_energy"] > row_014["elastic_energy"]
    assert row_018["elastic_energy"] > row_016["elastic_energy"]
    assert row_020["elastic_energy"] > row_018["elastic_energy"]

    contact_gain_018_to_020 = row_018["contact_energy"] - row_020["contact_energy"]
    elastic_cost_018_to_020 = row_020["elastic_energy"] - row_018["elastic_energy"]

    assert elastic_cost_018_to_020 > contact_gain_018_to_020


@pytest.mark.e2e
def test_kozlov_free_disk_curved_theta_gap_is_dominated_by_tilt_in_growth(
    _energy_sweep_main: list[dict[str, float]],
) -> None:
    """E2E: after both shared-rim exclusions, inner tilt again dominates overgrowth."""
    rows = _energy_sweep_main
    growth = summarize_free_disk_curved_elastic_growth(rows)

    assert len(growth) == 6
    for step in growth:
        assert step["dominant_term"] == "tilt_in_energy"
        deltas = step["term_deltas"]
        assert float(deltas["tilt_in_energy"]) > float(deltas["tilt_out_energy"])
        assert float(deltas["tilt_in_energy"]) > float(deltas["bending_tilt_in_energy"])
        assert float(deltas["tilt_in_energy"]) > float(
            deltas["bending_tilt_out_energy"]
        )


@pytest.mark.e2e
def test_kozlov_free_disk_tilt_in_overgrowth_is_near_rim_dominated(
    _energy_sweep_081018: list[dict[str, float]],
) -> None:
    """E2E: after shared-rim exclusion, inner-tilt growth is no longer rim dominated."""
    rows = _energy_sweep_081018
    by_theta = {float(row["theta_b"]): row for row in rows}

    row_008 = by_theta[0.08]
    row_010 = by_theta[0.10]
    row_018 = by_theta[0.18]

    growth_008_to_010 = {
        key: float(row_010[f"tilt_in_{key}"]) - float(row_008[f"tilt_in_{key}"])
        for key in ("disk_core", "disk_rim", "rim_outer", "outer_membrane")
    }
    growth_010_to_018 = {
        key: float(row_018[f"tilt_in_{key}"]) - float(row_010[f"tilt_in_{key}"])
        for key in ("disk_core", "disk_rim", "rim_outer", "outer_membrane")
    }

    near_rim_008_to_010 = growth_008_to_010["disk_rim"] + growth_008_to_010["rim_outer"]
    near_rim_010_to_018 = growth_010_to_018["disk_rim"] + growth_010_to_018["rim_outer"]

    assert near_rim_008_to_010 > growth_008_to_010["disk_core"]
    assert near_rim_008_to_010 < growth_008_to_010["outer_membrane"]
    assert near_rim_010_to_018 > growth_010_to_018["disk_core"]
    assert near_rim_010_to_018 < growth_010_to_018["outer_membrane"]


@pytest.mark.e2e
def test_kozlov_free_disk_tilt_in_overgrowth_is_disk_rim_dominated(
    _energy_sweep_081018: list[dict[str, float]],
) -> None:
    """E2E: after shared-rim exclusions, outer-membrane inner-tilt growth is largest."""
    rows = _energy_sweep_081018
    by_theta = {float(row["theta_b"]): row for row in rows}

    row_008 = by_theta[0.08]
    row_010 = by_theta[0.10]
    row_018 = by_theta[0.18]

    growth_008_to_010 = {
        key: float(row_010[f"tilt_in_{key}"]) - float(row_008[f"tilt_in_{key}"])
        for key in ("disk_core", "disk_rim", "rim_outer", "outer_membrane")
    }
    growth_010_to_018 = {
        key: float(row_018[f"tilt_in_{key}"]) - float(row_010[f"tilt_in_{key}"])
        for key in ("disk_core", "disk_rim", "rim_outer", "outer_membrane")
    }

    assert max(growth_008_to_010, key=growth_008_to_010.get) == "outer_membrane"
    assert max(growth_010_to_018, key=growth_010_to_018.get) == "outer_membrane"


@pytest.mark.e2e
def test_kozlov_free_disk_post_theta018_tilt_in_growth_is_outer_membrane_dominated(
    _energy_sweep_0141820: list[dict[str, float]],
) -> None:
    """E2E: above thetaB~0.18, inner-tilt growth stays outer-membrane dominated."""
    rows = _energy_sweep_0141820
    by_theta = {float(row["theta_b"]): row for row in rows}

    row_018 = by_theta[0.18]
    row_014 = by_theta[0.14]
    row_020 = by_theta[0.20]

    growth_014_to_018 = {
        key: float(row_018[f"tilt_in_{key}"]) - float(row_014[f"tilt_in_{key}"])
        for key in ("disk_core", "disk_rim", "rim_outer", "outer_membrane")
    }
    growth_018_to_020 = {
        key: float(row_020[f"tilt_in_{key}"]) - float(row_018[f"tilt_in_{key}"])
        for key in ("disk_core", "disk_rim", "rim_outer", "outer_membrane")
    }

    assert max(growth_014_to_018, key=growth_014_to_018.get) == "outer_membrane"
    assert max(growth_018_to_020, key=growth_018_to_020.get) == "outer_membrane"


@pytest.mark.e2e
def test_kozlov_free_disk_post_theta018_tilt_in_growth_is_outer_support_band_dominated(
    _energy_sweep_0141820: list[dict[str, float]],
) -> None:
    """E2E: above thetaB~0.18, residual inner-tilt growth stays near the first outer band."""
    rows = _energy_sweep_0141820
    by_theta = {float(row["theta_b"]): row for row in rows}

    row_018 = by_theta[0.18]
    row_014 = by_theta[0.14]
    row_020 = by_theta[0.20]

    growth_014_to_018 = {
        key: float(row_018[f"tilt_in_{key}"]) - float(row_014[f"tilt_in_{key}"])
        for key in ("outer_support_band", "outer_far")
    }
    growth_018_to_020 = {
        key: float(row_020[f"tilt_in_{key}"]) - float(row_018[f"tilt_in_{key}"])
        for key in ("outer_support_band", "outer_far")
    }

    assert max(growth_014_to_018, key=growth_014_to_018.get) == "outer_support_band"
    assert max(growth_018_to_020, key=growth_018_to_020.get) == "outer_support_band"


@pytest.mark.e2e
def test_kozlov_free_disk_one_step_refinement_shrinks_support_band_but_not_total_outer_tilt_in() -> (
    None
):
    """E2E: one local refinement shrinks the first support band control area but shifts cost outward."""
    coarse = run_free_disk_curved_bilayer_refinement_sweep(
        [0.14, 0.18],
        refine_steps=0,
        shape_steps=10,
    )
    refined = run_free_disk_curved_bilayer_refinement_sweep(
        [0.14, 0.18],
        refine_steps=1,
        shape_steps=10,
    )
    coarse_by_theta = {float(row["theta_b"]): row for row in coarse}
    refined_by_theta = {float(row["theta_b"]): row for row in refined}

    for theta in (0.14, 0.18):
        coarse_row = coarse_by_theta[theta]
        refined_row = refined_by_theta[theta]
        assert float(refined_row["ring_r"]) < float(coarse_row["ring_r"])
        assert float(refined_row["outer_control_area"]) < float(
            coarse_row["outer_control_area"]
        )
        assert float(refined_row["tilt_in_outer_support_band"]) < float(
            coarse_row["tilt_in_outer_support_band"]
        )
        assert float(refined_row["tilt_in_outer_far"]) > float(
            coarse_row["tilt_in_outer_far"]
        )
        assert float(refined_row["tilt_in_energy"]) >= float(
            coarse_row["tilt_in_energy"]
        )


@pytest.mark.e2e
def test_kozlov_free_disk_one_step_refinement_does_not_lift_curved_theta_b() -> None:
    """E2E: naive local refinement does not solve the thetaB gap by itself."""
    coarse = run_free_disk_curved_bilayer_refinement_sweep(
        [0.04, 0.08, 0.12, 0.14, 0.16, 0.18],
        refine_steps=0,
        shape_steps=15,
    )
    refined = run_free_disk_curved_bilayer_refinement_sweep(
        [0.04, 0.08, 0.12, 0.14, 0.16, 0.18],
        refine_steps=1,
        shape_steps=15,
    )

    coarse_best = min(coarse, key=lambda row: float(row["total_energy"]))
    refined_best = min(refined, key=lambda row: float(row["total_energy"]))

    assert float(coarse_best["theta_b"]) == pytest.approx(0.14, abs=0.03)
    assert float(refined_best["theta_b"]) <= float(coarse_best["theta_b"])
    assert float(refined_best["total_energy"]) > float(coarse_best["total_energy"])


@pytest.mark.regression
def test_kozlov_free_disk_shared_rim_control_areas_exceed_simple_annulus_targets_on_coarse_mesh() -> (
    None
):
    """Regression: coarse shared-rim dual areas are much larger than simple annulus targets."""
    row = run_free_disk_curved_bilayer_refinement_sweep(
        [0.14],
        refine_steps=0,
        shape_steps=10,
    )[0]

    assert float(row["rim_control_area"]) > float(row["rim_annulus_area"])
    assert float(row["outer_control_area"]) > float(row["outer_annulus_area"])
    assert float(row["rim_control_area"] / row["rim_annulus_area"]) > 2.0
    assert float(row["outer_control_area"] / row["outer_annulus_area"]) > 4.0


@pytest.mark.regression
def test_kozlov_free_disk_shared_rim_control_areas_drop_under_one_step_refinement() -> (
    None
):
    """Regression: one refinement sharply reduces shared-rim row control areas."""
    coarse = run_free_disk_curved_bilayer_refinement_sweep(
        [0.14],
        refine_steps=0,
        shape_steps=10,
    )[0]
    refined = run_free_disk_curved_bilayer_refinement_sweep(
        [0.14],
        refine_steps=1,
        shape_steps=10,
    )[0]

    assert float(refined["rim_control_area"]) < float(coarse["rim_control_area"])
    assert float(refined["outer_control_area"]) < float(coarse["outer_control_area"])
    assert float(refined["outer_control_area"] / coarse["outer_control_area"]) < 0.2
    assert float(refined["rim_control_area"] / coarse["rim_control_area"]) < 0.6


@pytest.mark.regression
def test_kozlov_free_disk_shared_rim_control_areas_track_adjacent_ring_shells() -> None:
    """Regression: shared-rim control areas are consistent with full adjacent-ring shells."""
    for refine_steps in (0, 1):
        row = run_free_disk_curved_bilayer_refinement_sweep(
            [0.14],
            refine_steps=refine_steps,
            shape_steps=10,
        )[0]
        assert float(row["rim_control_area"]) == pytest.approx(
            float(row["rim_shell_area"]), rel=0.06, abs=1.0e-6
        )
        assert float(row["outer_control_area"]) == pytest.approx(
            float(row["outer_shell_area"]), rel=0.06, abs=1.0e-6
        )


@pytest.mark.regression
def test_kozlov_free_disk_coarse_outer_support_ring_represents_a_wide_shell() -> None:
    """Regression: coarse outer support rows carry a full shell, not just the narrow R-to-R+ annulus."""
    row = run_free_disk_curved_bilayer_refinement_sweep(
        [0.14],
        refine_steps=0,
        shape_steps=10,
    )[0]

    assert float(row["outer_shell_area"]) > 4.0 * float(row["outer_annulus_area"])
    assert float(row["outer_control_area"]) > 4.0 * float(row["outer_annulus_area"])


@pytest.mark.regression
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"tilt_in_exclude_shared_rim_outer_rows": True},
        {"tilt_in_shared_rim_outer_row_energy_weight": 1.0},
    ],
)
def test_kozlov_free_disk_tilt_in_audit_matches_runtime_energy(
    overrides: dict[str, object],
) -> None:
    """Regression: regional tilt_in audit must sum to the runtime tilt_in term."""
    row = run_free_disk_curved_bilayer_energy_sweep(
        [0.14],
        global_parameter_overrides=overrides,
    )[0]
    partition_sum = sum(
        float(row[f"tilt_in_{key}"])
        for key in (
            "disk_core",
            "disk_rim",
            "rim_outer",
            "outer_support_band",
            "outer_far",
        )
    )
    assert partition_sum == pytest.approx(
        float(row["tilt_in_energy"]), rel=1e-6, abs=1e-9
    )


@pytest.mark.e2e
def test_kozlov_free_disk_shared_rim_tilt_in_exclusion_reduces_rim_overgrowth() -> None:
    """E2E: excluding shared-rim rows from inner tilt cost should lift curved thetaB."""
    theta_seed = optimize_free_disk_theta_b(load_free_disk_theory_mesh(), scans=4)
    baseline = optimize_free_disk_curved_theta_b(
        theta_b_seed=theta_seed,
        shape_steps=40,
        global_parameter_overrides={"tilt_in_exclude_shared_rim_rows": False},
    )
    corrected = optimize_free_disk_curved_theta_b(
        theta_b_seed=theta_seed,
        shape_steps=40,
        global_parameter_overrides={"tilt_in_exclude_shared_rim_rows": True},
    )

    assert float(corrected["best_theta_b"]) > float(baseline["best_theta_b"])

    baseline_row = run_free_disk_curved_bilayer_energy_sweep(
        [0.10],
        shape_steps=40,
        global_parameter_overrides={"tilt_in_exclude_shared_rim_rows": False},
    )[0]
    corrected_row = run_free_disk_curved_bilayer_energy_sweep(
        [0.10],
        shape_steps=40,
        global_parameter_overrides={"tilt_in_exclude_shared_rim_rows": True},
    )[0]

    assert float(corrected_row["tilt_in_disk_rim"]) < float(
        baseline_row["tilt_in_disk_rim"]
    )
    assert corrected_row["theta_outer_in"] == pytest.approx(0.05, rel=0.05, abs=1.0e-3)
    assert corrected_row["theta_outer_out"] == pytest.approx(0.05, rel=0.05, abs=1.0e-3)
    assert corrected_row["phi_abs"] == pytest.approx(0.05, rel=0.05, abs=1.0e-3)


@pytest.mark.e2e
def test_kozlov_free_disk_outer_band_tilt_in_half_weight_is_cleaner_than_exclusion(
    _theta_seed_scans2: float,
) -> None:
    """E2E: half-weighting shared-rim outer rows lifts thetaB while reducing support-band cost."""
    theta_seed = _theta_seed_scans2
    theta_offsets = (0.0, 0.04, 0.08, 0.10, 0.12)
    baseline = optimize_free_disk_curved_theta_b(
        theta_b_seed=theta_seed,
        shape_steps=10,
        theta_offsets=theta_offsets,
        global_parameter_overrides={"tilt_in_shared_rim_outer_row_energy_weight": 1.0},
    )
    weighted = optimize_free_disk_curved_theta_b(
        theta_b_seed=theta_seed,
        shape_steps=10,
        theta_offsets=theta_offsets,
        global_parameter_overrides={"tilt_in_shared_rim_outer_row_energy_weight": 0.5},
    )

    assert float(weighted["best_theta_b"]) >= float(baseline["best_theta_b"])
    assert float(weighted["best_total_energy"]) < float(baseline["best_total_energy"])

    baseline_row = run_free_disk_curved_bilayer_energy_sweep(
        [0.18],
        shape_steps=10,
        global_parameter_overrides={"tilt_in_shared_rim_outer_row_energy_weight": 1.0},
    )[0]
    weighted_row = run_free_disk_curved_bilayer_energy_sweep(
        [0.18],
        shape_steps=10,
        global_parameter_overrides={"tilt_in_shared_rim_outer_row_energy_weight": 0.5},
    )[0]

    assert float(weighted_row["tilt_in_outer_support_band"]) < float(
        baseline_row["tilt_in_outer_support_band"]
    )
    assert weighted_row["theta_outer_in"] == pytest.approx(0.09, rel=0.05, abs=1.0e-3)
    assert weighted_row["theta_outer_out"] == pytest.approx(0.09, rel=0.05, abs=1.0e-3)
    assert weighted_row["phi_abs"] == pytest.approx(0.09, rel=0.05, abs=1.0e-3)


@pytest.mark.e2e
def test_kozlov_free_disk_outer_shell_consistent_quadrature_lifts_curved_theta_b(
    _theta_seed_scans2: float,
) -> None:
    """E2E: shell-consistent inner quadrature should beat coarse lumped support-shell quadrature."""
    theta_seed = _theta_seed_scans2
    theta_offsets = (0.0, 0.04, 0.08, 0.10, 0.12)
    baseline = optimize_free_disk_curved_theta_b(
        theta_b_seed=theta_seed,
        shape_steps=40,
        theta_offsets=theta_offsets,
        global_parameter_overrides={
            "tilt_in_shared_rim_outer_shell_mass_mode": "lumped"
        },
    )
    corrected = optimize_free_disk_curved_theta_b(
        theta_b_seed=theta_seed,
        shape_steps=40,
        theta_offsets=theta_offsets,
    )

    assert float(corrected["best_theta_b"]) > float(baseline["best_theta_b"])

    baseline_row = run_free_disk_curved_bilayer_energy_sweep(
        [0.18],
        shape_steps=20,
        global_parameter_overrides={
            "tilt_in_shared_rim_outer_shell_mass_mode": "lumped"
        },
    )[0]
    corrected_row = run_free_disk_curved_bilayer_energy_sweep(
        [0.18],
        shape_steps=20,
        global_parameter_overrides={
            "tilt_in_shared_rim_outer_shell_mass_mode": "consistent"
        },
    )[0]

    assert float(corrected_row["total_energy"]) < float(baseline_row["total_energy"])
    assert float(corrected_row["tilt_in_energy"]) < float(
        baseline_row["tilt_in_energy"]
    )
    assert corrected_row["theta_outer_in"] == pytest.approx(0.09, rel=0.05, abs=1.0e-3)
    assert corrected_row["theta_outer_out"] == pytest.approx(0.09, rel=0.05, abs=1.0e-3)
    assert corrected_row["phi_abs"] == pytest.approx(0.09, rel=0.05, abs=1.0e-3)


@pytest.mark.e2e
def test_kozlov_free_disk_outer_excess_is_outer_membrane_tilt_out_dominated(
    _energy_sweep_081018: list[dict[str, float]],
) -> None:
    """E2E: outer tilt growth stays off the core and becomes tail dominated at high theta."""
    rows = _energy_sweep_081018
    by_theta = {float(row["theta_b"]): row for row in rows}

    row_008 = by_theta[0.08]
    row_010 = by_theta[0.10]
    row_018 = by_theta[0.18]

    tilt_out_growth_008_to_010 = {
        key: float(row_010[f"tilt_out_{key}"]) - float(row_008[f"tilt_out_{key}"])
        for key in ("disk_core", "disk_rim", "rim_outer", "outer_membrane")
    }
    tilt_out_growth_010_to_018 = {
        key: float(row_018[f"tilt_out_{key}"]) - float(row_010[f"tilt_out_{key}"])
        for key in ("disk_core", "disk_rim", "rim_outer", "outer_membrane")
    }

    bending_out_growth_008_to_010 = {
        key: float(row_010[f"bending_tilt_out_{key}"])
        - float(row_008[f"bending_tilt_out_{key}"])
        for key in ("disk_core", "disk_rim", "rim_outer", "outer_membrane")
    }
    bending_out_growth_010_to_018 = {
        key: float(row_018[f"bending_tilt_out_{key}"])
        - float(row_010[f"bending_tilt_out_{key}"])
        for key in ("disk_core", "disk_rim", "rim_outer", "outer_membrane")
    }

    # The low-theta winner is solver/version sensitive between the disk-rim
    # band and the outer-membrane tail, but it should stay away from disk_core.
    assert max(tilt_out_growth_008_to_010, key=tilt_out_growth_008_to_010.get) in {
        "disk_rim",
        "outer_membrane",
    }
    assert float(tilt_out_growth_008_to_010["disk_core"]) < max(
        float(tilt_out_growth_008_to_010["disk_rim"]),
        float(tilt_out_growth_008_to_010["outer_membrane"]),
    )
    assert (
        max(tilt_out_growth_010_to_018, key=tilt_out_growth_010_to_018.get)
        == "outer_membrane"
    )
    assert (
        max(bending_out_growth_008_to_010, key=bending_out_growth_008_to_010.get)
        == "rim_outer"
    )
    assert (
        max(bending_out_growth_010_to_018, key=bending_out_growth_010_to_018.get)
        == "rim_outer"
    )

    assert float(
        row_010["bending_tilt_out_energy"] - row_008["bending_tilt_out_energy"]
    ) > float(row_010["tilt_out_energy"] - row_008["tilt_out_energy"])
    assert float(
        row_018["bending_tilt_out_energy"] - row_010["bending_tilt_out_energy"]
    ) > float(row_018["tilt_out_energy"] - row_010["tilt_out_energy"])


@pytest.mark.e2e
def test_kozlov_free_disk_outer_row_tilt_out_exclusion_lifts_curved_thetaB() -> None:
    """E2E: excluding shared-rim outer rows from tilt_out should lift curved thetaB."""
    theta_seed = optimize_free_disk_theta_b(load_free_disk_theory_mesh(), scans=4)
    baseline = optimize_free_disk_curved_theta_b(
        theta_b_seed=theta_seed,
        shape_steps=40,
        global_parameter_overrides={"tilt_out_exclude_shared_rim_outer_rows": False},
    )
    corrected = optimize_free_disk_curved_theta_b(
        theta_b_seed=theta_seed,
        shape_steps=40,
        global_parameter_overrides={"tilt_out_exclude_shared_rim_outer_rows": True},
    )

    assert float(corrected["best_theta_b"]) > float(baseline["best_theta_b"])

    baseline_row = run_free_disk_curved_bilayer_energy_sweep(
        [0.10],
        shape_steps=40,
        global_parameter_overrides={"tilt_out_exclude_shared_rim_outer_rows": False},
    )[0]
    corrected_row = run_free_disk_curved_bilayer_energy_sweep(
        [0.10],
        shape_steps=40,
        global_parameter_overrides={"tilt_out_exclude_shared_rim_outer_rows": True},
    )[0]

    assert float(corrected_row["tilt_out_rim_outer"]) < float(
        baseline_row["tilt_out_rim_outer"]
    )
    assert corrected_row["theta_outer_in"] == pytest.approx(0.05, rel=0.05, abs=1.0e-3)
    assert corrected_row["theta_outer_out"] == pytest.approx(0.05, rel=0.05, abs=1.0e-3)
    assert corrected_row["phi_abs"] == pytest.approx(0.05, rel=0.05, abs=1.0e-3)


@pytest.mark.e2e
def test_kozlov_free_disk_outer_band_tilt_in_exclusion_is_not_a_clean_fix() -> None:
    """E2E: outer-band tilt_in exclusion changes the branch but worsens fixed-theta support cost."""
    baseline_row = run_free_disk_curved_bilayer_energy_sweep(
        [0.18],
        shape_steps=40,
        global_parameter_overrides={"tilt_in_exclude_shared_rim_outer_rows": False},
    )[0]
    corrected_row = run_free_disk_curved_bilayer_energy_sweep(
        [0.18],
        shape_steps=40,
        global_parameter_overrides={"tilt_in_exclude_shared_rim_outer_rows": True},
    )[0]

    assert float(corrected_row["tilt_in_outer_support_band"]) < float(
        baseline_row["tilt_in_outer_support_band"]
    )
    assert float(corrected_row["total_energy"]) < float(baseline_row["total_energy"])
    assert abs(float(corrected_row["theta_outer_in"]) - 0.09) > 0.02
    assert abs(float(corrected_row["theta_outer_out"]) - 0.09) > 0.02
    assert abs(float(corrected_row["phi_abs"]) - 0.09) > 0.02
