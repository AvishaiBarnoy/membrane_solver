import math

import numpy as np
import pytest

from tools.diagnostics.curved_1disk_shape_direction_audit import (
    ALLOWED_CLASSIFICATIONS,
    _direction_catalog,
    _prepare_minimizer,
    run_curved_1disk_shape_direction_audit,
)
from tools.diagnostics.curved_1disk_trumpet_descent_audit import _apply_z_mode
from tools.diagnostics.utils import capture_state, restore_state

pytestmark = pytest.mark.slow


def test_curved_1disk_leaflet_shape_gradients_match_log_direction() -> None:
    minim = _prepare_minimizer(0.1845693593)
    mesh = minim.mesh
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    minim._sync_evaluation_manager()
    evaluator = minim._evaluation_manager

    _energy, full_gradient = minim.compute_energy_and_gradient_array()
    direction = _direction_catalog(mesh, full_gradient[:, 2])["outer_log_trumpet"]
    state = capture_state(mesh)
    epsilon = 1.0e-6

    for name in ("bending_tilt_in", "bending_tilt_out"):
        module = minim.energy_modules[minim.energy_module_names.index(name)]
        gradient = np.zeros_like(positions)
        evaluator._call_module_array(
            module, positions=positions, index_map=index_map, grad_arr=gradient
        )
        energies = []
        for sign in (1.0, -1.0):
            restore_state(mesh, *state)
            _apply_z_mode(mesh, direction, sign * epsilon)
            energies.append(
                evaluator._call_module_array(
                    module,
                    positions=mesh.positions_view(),
                    index_map=index_map,
                    grad_arr=None,
                )
            )
        analytic = float(np.dot(gradient[:, 2], direction))
        central = float((energies[0] - energies[1]) / (2.0 * epsilon))
        assert analytic == pytest.approx(central, rel=0.02, abs=2.0e-3)

    restore_state(mesh, *state)


def test_curved_1disk_shape_direction_audit_reports_required_schema() -> None:
    report = run_curved_1disk_shape_direction_audit(horizons=(1,))

    assert report["diagnosis"]["classification"] in ALLOWED_CLASSIFICATIONS
    assert report["diagnosis"]["no_energy_rescaling"] is True
    assert "Feature Contract" in report["diagnosis"]["summary"]
    assert "do not rescale" in report["diagnosis"]["recommended_next_stream"]

    directions = {row["name"]: row for row in report["direction_summaries"]}
    assert {
        "outer_log_trumpet",
        "projected_gradient_descent",
        "log_residual_gradient",
        "near_support_gradient",
        "far_field_gradient",
        "high_frequency_gradient",
        "area_weighted_gradient_probe",
        "shell_normalized_gradient_probe",
        "support_suppressed_gradient_probe",
    } <= set(directions)
    assert directions["outer_log_trumpet"]["norm"] == pytest.approx(1.0)
    assert directions["outer_log_trumpet"]["nonzero_rows"] > 0

    log_probe = next(
        row
        for row in report["directional_probes"]
        if row["name"] == "outer_log_trumpet" and not row["relax_tilts"]
    )
    assert bool(log_probe["accepted_by_decrease"]) is True
    assert float(log_probe["total_delta"]) < 0.0


def test_curved_1disk_shape_direction_audit_reconciles_and_replays_updates() -> None:
    report = run_curved_1disk_shape_direction_audit(horizons=(1,))

    for probe in report["directional_probes"]:
        assert math.isfinite(float(probe["total_delta"]))
        assert math.isfinite(float(probe["module_delta_sum"]))
        assert abs(float(probe["module_residual"])) < 1.0e-10
        assert probe["top_module_deltas"]
        assert float(probe["direction_norm"]) == pytest.approx(0.0) or float(
            probe["direction_norm"]
        ) == pytest.approx(1.0)

    replay = report["accepted_update_replay"]
    assert len(replay) == 1
    row = replay[0]
    assert int(row["n_steps"]) == 1
    assert bool(row["step_success"]) is True
    assert float(row["xy_delta_abs_sum"]) < 1.0e-10
    assert float(row["z_delta_abs_sum"]) > 0.0
    assert "outer_log_trumpet" in row["mode_alignment"]
    assert row["z_delta_by_shell"]
