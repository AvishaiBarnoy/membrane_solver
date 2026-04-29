import numpy as np
import pytest

from geometry.geom_io import parse_geometry
from modules.constraints import rim_slope_match_out
from tests.test_rim_slope_match_out_constraint import (
    _build_rim_match_geometry,
    _collect_group_rows,
)
from tools.diagnostics.curved_1disk_shape_propagation_blocker import (
    run_curved_1disk_shape_propagation_blocker,
)


def test_shared_rim_minimize_shape_enforcement_does_not_mutate_tilt_out() -> None:
    data = _build_rim_match_geometry(z_bump=0.12)
    data["global_parameters"]["rim_slope_match_mode"] = "shared_rim_staggered_v1"
    data["global_parameters"]["rim_slope_match_thetaB_param"] = "tilt_thetaB_value"
    data["global_parameters"]["tilt_thetaB_value"] = 0.2
    mesh = parse_geometry(data)
    mesh.build_position_cache()

    outer_rows = _collect_group_rows(mesh, "outer")
    tilts_out = mesh.tilts_out_view().copy(order="F")
    positions = mesh.positions_view()
    r_hat = positions[outer_rows].copy()
    r_hat[:, 2] = 0.0
    r_hat /= np.linalg.norm(r_hat, axis=1)[:, None]
    tilts_out[outer_rows] = 0.03 * r_hat
    mesh.set_tilts_out_from_array(tilts_out)
    before = mesh.tilts_out_view().copy(order="F")

    rim_slope_match_out.enforce_constraint(
        mesh,
        global_params=mesh.global_parameters,
        context="minimize",
    )

    np.testing.assert_allclose(mesh.tilts_out_view(), before, atol=0.0)


@pytest.mark.benchmark
@pytest.mark.slow
def test_fixed_theta_shape_blocker_report_identifies_backtracking_budget() -> None:
    report = run_curved_1disk_shape_propagation_blocker()

    assert report["classification"] == "shape_update_accepted"
    alpha0 = report["line_search_probe"]["alpha0_enforcement"]
    assert float(alpha0["energy_delta"]) <= 0.0
    assert float(alpha0["tilt_out_delta_max"]) < 1.0e-3
    trials = report["line_search_probe"]["trial_alphas"]
    assert any(bool(row["accepted_by_decrease"]) for row in trials)
    assert bool(report["one_step_default_backtracking"]["step_success"]) is True
    assert bool(report["one_step_extended_backtracking"]["step_success"]) is True
    assert float(report["one_step_extended_backtracking"]["xy_delta_abs_sum"]) < 1.0e-12
    assert float(report["one_step_extended_backtracking"]["z_delta_abs_sum"]) > 0.0
