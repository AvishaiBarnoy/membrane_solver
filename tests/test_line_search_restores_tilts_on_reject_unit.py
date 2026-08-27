import numpy as np

from runtime.steppers.line_search import backtracking_line_search_array
from tests.sample_meshes import single_triangle_mesh


def test_line_search_restores_tilts_on_reject_when_reduced_energy_enabled() -> None:
    mesh = single_triangle_mesh()
    vertex_ids = tuple(int(v) for v in mesh.vertex_ids.tolist())

    # Set an explicit initial tilt state.
    rng = np.random.default_rng(0)
    mesh.set_tilts_in_from_array(rng.normal(size=(len(vertex_ids), 3)))
    mesh.set_tilts_out_from_array(rng.normal(size=(len(vertex_ids), 3)))
    mesh.set_tilts_from_array(rng.normal(size=(len(vertex_ids), 3)))

    # Signal reduced-energy mode so line_search snapshots tilts after energy0.
    mesh._line_search_reduced_energy = True

    x0 = mesh.positions_view().copy()

    def energy_fn() -> float:
        # Mutate tilts based on current positions to emulate reduced-energy
        # evaluation that relaxes/updates tilts during trial energy calls.
        pos = mesh.positions_view()
        marker = float(np.sum(pos))
        mesh.tilts_in_view()[:] = marker
        mesh.tilts_out_view()[:] = -marker
        mesh.tilts_view()[:] = 0.5 * marker
        # Energy increases when vertex 0 moves in +x.
        return float(pos[0, 0])

    # Direction moves vertex 0 in +x, so trial energy will increase.
    direction = np.zeros_like(x0)
    direction[0, 0] = 1.0

    # Provide a gradient that makes this look like a descent direction.
    gradient = -direction.copy()

    success, _new_step, _accepted = backtracking_line_search_array(
        mesh,
        direction,
        gradient,
        step_size=0.5,
        energy_fn=energy_fn,
        vertex_ids=vertex_ids,
        max_iter=1,
        beta=0.5,
        c=1e-4,
        gamma=1.0,
        alpha_max_factor=1.0,
        constraint_enforcer=None,
    )
    assert success is False

    # Positions restored.
    assert np.allclose(mesh.positions_view(), x0)

    # Tilts restored to the baseline state used for energy0 comparisons.
    baseline_marker = float(np.sum(x0))
    assert np.allclose(mesh.tilts_in_view(), baseline_marker)
    assert np.allclose(mesh.tilts_out_view(), -baseline_marker)
    assert np.allclose(mesh.tilts_view(), 0.5 * baseline_marker)
