import numpy as np

from geometry.geom_io import load_data, parse_geometry
from runtime.steppers.line_search import backtracking_line_search_array


def test_line_search_failure_restores_state():
    data = load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    mesh = parse_geometry(data)

    positions = mesh.positions_view().copy()
    gradient = np.ones_like(positions)
    direction = -gradient

    baseline_tilt_in = np.full_like(mesh.tilts_in_view(), 0.123)
    baseline_tilt_out = np.full_like(mesh.tilts_out_view(), -0.234)
    later_tilt_in = np.full_like(mesh.tilts_in_view(), 0.456)
    later_tilt_out = np.full_like(mesh.tilts_out_view(), -0.567)

    state = {"first": True}

    def energy_fn():
        if state["first"]:
            mesh.set_tilts_in_from_array(baseline_tilt_in)
            mesh.set_tilts_out_from_array(baseline_tilt_out)
            mesh.global_parameters.set("tilt_thetaB_value", 0.1)
            state["first"] = False
        else:
            mesh.set_tilts_in_from_array(later_tilt_in)
            mesh.set_tilts_out_from_array(later_tilt_out)
            mesh.global_parameters.set("tilt_thetaB_value", 0.5)
        return 1.0

    setattr(mesh, "_line_search_reduced_energy", True)
    try:
        success, _, _ = backtracking_line_search_array(
            mesh,
            direction,
            gradient,
            step_size=1e-3,
            energy_fn=energy_fn,
            vertex_ids=mesh.vertex_ids,
            max_iter=2,
            beta=0.1,
            c=1e-4,
            gamma=1.5,
            alpha_max_factor=1.0,
            constraint_enforcer=None,
        )
    finally:
        delattr(mesh, "_line_search_reduced_energy")

    assert not success
    np.testing.assert_allclose(mesh.positions_view(), positions, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        mesh.tilts_in_view(), baseline_tilt_in, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        mesh.tilts_out_view(), baseline_tilt_out, rtol=0, atol=1e-12
    )
    assert mesh.global_parameters.get("tilt_thetaB_value") == 0.1
