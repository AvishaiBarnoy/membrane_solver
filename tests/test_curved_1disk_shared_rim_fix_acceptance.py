import numpy as np
import pytest

from tools.diagnostics.curved_1disk_shared_rim_phi_target_audit import (
    THEORY_THETA_B,
    run_curved_1disk_shared_rim_phi_target_audit,
)
from tools.diagnostics.curved_1disk_theory_benchmark import (
    OUTER_K1_WINDOW,
    OUTER_LOG_WINDOW,
    _fit_outer_k1,
    _fit_outer_log_height,
    _run_curved_theta_candidate,
    _shell_profile,
    run_curved_1disk_theory_benchmark,
)


@pytest.fixture(scope="module")
def fixed_theta_result() -> dict[str, object]:
    return _run_curved_theta_candidate(THEORY_THETA_B)


@pytest.mark.benchmark
@pytest.mark.slow
def test_fixed_theta_shell2_target_direction_is_outward() -> None:
    report = run_curved_1disk_shared_rim_phi_target_audit()

    direction = report["shell_target_construction"]["target_direction"]
    assert float(direction["r_dir_cos_global_radial_median"]) > 0.50
    assert float(direction["r_dir_cos_global_radial_min"]) > 0.0


@pytest.mark.benchmark
@pytest.mark.slow
def test_fixed_theta_outer_height_propagates_past_first_shell(
    fixed_theta_result: dict[str, object],
) -> None:
    mesh = fixed_theta_result["mesh"]
    shell_rows = _shell_profile(mesh)
    radius = 7.0 / 15.0
    outer_rows = [row for row in shell_rows if float(row["radius"]) > radius + 1.0e-6]
    nonzero_outer = [row for row in outer_rows if abs(float(row["z"])) > 1.0e-12]
    last_free_shell_radius = max(float(row["radius"]) for row in outer_rows[:-1])
    support_radius = min(float(row["radius"]) for row in outer_rows)
    propagated_rows = [
        row for row in outer_rows if float(row["radius"]) > support_radius + 1.0e-4
    ]

    assert len(nonzero_outer) >= 2
    assert max(float(row["z"]) for row in propagated_rows) > 0.0
    fit = _fit_outer_log_height(
        shell_rows,
        radius=radius,
        slope_theory=0.5 * THEORY_THETA_B * radius,
        last_free_shell_radius=last_free_shell_radius,
        window=OUTER_LOG_WINDOW,
    )
    assert abs(float(fit["slope_fit"])) > 1.0e-10


@pytest.mark.xfail(
    reason=(
        "Post-PR503 diagnostics show energy attribution reconciles, but the "
        "fixed-theta outer trumpet still does not match the TeX log profile."
    ),
)
@pytest.mark.benchmark
@pytest.mark.slow
def test_fixed_theta_outer_height_matches_tex_log_profile(
    fixed_theta_result: dict[str, object],
) -> None:
    mesh = fixed_theta_result["mesh"]
    shell_rows = _shell_profile(mesh)
    radius = 7.0 / 15.0
    outer_rows = [row for row in shell_rows if float(row["radius"]) > radius + 1.0e-6]
    last_free_shell_radius = max(float(row["radius"]) for row in outer_rows[:-1])

    fit = _fit_outer_log_height(
        shell_rows,
        radius=radius,
        slope_theory=0.5 * THEORY_THETA_B * radius,
        last_free_shell_radius=last_free_shell_radius,
        window=OUTER_LOG_WINDOW,
    )

    assert float(fit["slope_ratio"]) == pytest.approx(1.0, rel=0.25)
    assert float(fit["rel_rmse"]) < 0.20


@pytest.mark.xfail(
    reason=(
        "The remaining fixed-theta miss is profile propagation: the outer "
        "leaflets have not converged to the TeX K1 branch."
    ),
)
@pytest.mark.benchmark
@pytest.mark.slow
def test_fixed_theta_outer_tilt_matches_tex_k1_profile(
    fixed_theta_result: dict[str, object],
) -> None:
    mesh = fixed_theta_result["mesh"]
    shell_rows = _shell_profile(mesh)
    radius = 7.0 / 15.0
    outer_rows = [row for row in shell_rows if float(row["radius"]) > radius + 1.0e-6]
    last_free_shell_radius = max(float(row["radius"]) for row in outer_rows[:-1])

    fit = _fit_outer_k1(
        shell_rows,
        radius=radius,
        lambda_theory=15.0,
        last_free_shell_radius=last_free_shell_radius,
        window=OUTER_K1_WINDOW,
    )

    assert float(fit["lambda_ratio"]) == pytest.approx(1.0, rel=0.25)
    assert float(fit["rel_rmse"]) < 0.20
    assert float(fit["leaflet_mismatch_median"]) < 0.10


@pytest.mark.benchmark
@pytest.mark.slow
def test_fixed_theta_near_rim_split_is_preserved(
    fixed_theta_result: dict[str, object],
) -> None:
    near_rim = fixed_theta_result["near_rim"]
    half_theta = 0.5 * THEORY_THETA_B

    assert abs(float(near_rim["phi"])) / THEORY_THETA_B == pytest.approx(0.5, rel=0.10)
    shell_rows = _shell_profile(fixed_theta_result["mesh"])
    support_radius = min(
        float(row["radius"])
        for row in shell_rows
        if float(row["radius"]) > float(near_rim["rim_radius"]) + 1.0e-6
    )
    target_radius = min(
        float(row["radius"])
        for row in shell_rows
        if float(row["radius"]) > support_radius + 1.0e-4
    )
    target_rows = [
        row for row in shell_rows if abs(float(row["radius"]) - target_radius) < 1.0e-6
    ]
    theta_in = float(np.median([float(row["theta_in"]) for row in target_rows]))
    theta_out = float(np.median([float(row["theta_out"]) for row in target_rows]))

    assert theta_in / half_theta == pytest.approx(1.0, rel=0.15)
    assert theta_out / half_theta == pytest.approx(1.0, rel=0.15)


@pytest.mark.benchmark
@pytest.mark.slow
def test_selected_theta_moves_above_current_low_branch() -> None:
    report = run_curved_1disk_theory_benchmark()

    assert float(report["theta_B_selected"]) > 0.06
