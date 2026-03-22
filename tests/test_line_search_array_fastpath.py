import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.steppers.line_search import backtracking_line_search_array


def _build_triangle_mesh() -> Mesh:
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([0.0, 1.0, 0.0])),
    }
    mesh.edges = {1: Edge(1, 0, 1), 2: Edge(2, 1, 2), 3: Edge(3, 2, 0)}
    mesh.facets = {0: Facet(0, edge_indices=[1, 2, 3])}
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def test_array_line_search_fast_path_uses_trial_positions_without_mutating_mesh() -> (
    None
):
    mesh = _build_triangle_mesh()
    vertex_ids = tuple(int(v) for v in mesh.vertex_ids.tolist())
    x0 = mesh.positions_view().copy()

    direction = np.zeros_like(x0)
    direction[1, 0] = -1.0
    gradient = np.zeros_like(x0)
    gradient[1, 0] = 1.0

    energy_calls = {"count": 0}
    trial_positions_seen: list[np.ndarray] = []

    def energy_fn() -> float:
        energy_calls["count"] += 1
        np.testing.assert_allclose(mesh.positions_view(), x0, rtol=0, atol=1e-12)
        return float(mesh.positions_view()[1, 0] ** 2)

    def trial_energy_fn(trial_positions: np.ndarray) -> float:
        np.testing.assert_allclose(mesh.positions_view(), x0, rtol=0, atol=1e-12)
        trial_positions_seen.append(np.array(trial_positions, copy=True))
        return float(trial_positions[1, 0] ** 2)

    success, _new_step, accepted_energy = backtracking_line_search_array(
        mesh,
        direction,
        gradient,
        step_size=0.5,
        energy_fn=energy_fn,
        trial_energy_fn=trial_energy_fn,
        vertex_ids=vertex_ids,
        max_iter=1,
        beta=0.5,
        c=1e-4,
        gamma=1.0,
        alpha_max_factor=1.0,
        constraint_enforcer=None,
    )

    assert success is True
    assert energy_calls["count"] == 1
    assert len(trial_positions_seen) == 1
    assert accepted_energy == 0.25
    np.testing.assert_allclose(mesh.positions_view()[1], np.array([0.5, 0.0, 0.0]))


def test_array_line_search_fast_path_failure_keeps_mesh_state_pristine() -> None:
    mesh = _build_triangle_mesh()
    vertex_ids = tuple(int(v) for v in mesh.vertex_ids.tolist())
    x0 = mesh.positions_view().copy()

    direction = np.zeros_like(x0)
    direction[1, 0] = 1.0
    gradient = np.zeros_like(x0)
    gradient[1, 0] = -1.0

    trial_energies: list[float] = []

    def energy_fn() -> float:
        np.testing.assert_allclose(mesh.positions_view(), x0, rtol=0, atol=1e-12)
        return float(mesh.positions_view()[1, 0])

    def trial_energy_fn(trial_positions: np.ndarray) -> float:
        np.testing.assert_allclose(mesh.positions_view(), x0, rtol=0, atol=1e-12)
        energy = float(trial_positions[1, 0])
        trial_energies.append(energy)
        return energy

    success, _new_step, accepted_energy = backtracking_line_search_array(
        mesh,
        direction,
        gradient,
        step_size=1.0,
        energy_fn=energy_fn,
        trial_energy_fn=trial_energy_fn,
        vertex_ids=vertex_ids,
        max_iter=1,
        beta=0.5,
        c=1e-4,
        gamma=1.0,
        alpha_max_factor=1.0,
        constraint_enforcer=None,
    )

    assert success is False
    assert trial_energies == [2.0]
    assert accepted_energy == 1.0
    np.testing.assert_allclose(mesh.positions_view(), x0, rtol=0, atol=1e-12)


def test_array_line_search_reduced_fast_path_accepts_trial_state() -> None:
    mesh = _build_triangle_mesh()
    mesh.global_parameters = GlobalParameters()
    vertex_ids = tuple(int(v) for v in mesh.vertex_ids.tolist())
    x0 = mesh.positions_view().copy()
    mesh._line_search_reduced_energy = True
    mesh._line_search_reduced_accept_rule = "decrease_only"

    direction = np.zeros((len(vertex_ids), 3), dtype=float)
    direction[1, 0] = -1.0
    gradient = direction.copy()
    trial_markers: list[float] = []

    def energy_fn() -> float:
        np.testing.assert_allclose(mesh.positions_view(), x0, rtol=0, atol=1e-12)
        marker = float(np.sum(mesh.positions_view()))
        mesh.tilts_in_view()[:] = marker
        mesh.tilts_out_view()[:] = -marker
        mesh.tilts_view()[:] = 0.5 * marker
        mesh.global_parameters.set("tilt_thetaB_value", marker)
        return float(mesh.positions_view()[1, 0])

    def trial_energy_fn(trial_positions: np.ndarray) -> float:
        np.testing.assert_allclose(mesh.positions_view(), x0, rtol=0, atol=1e-12)
        marker = float(np.sum(trial_positions))
        trial_markers.append(marker)
        mesh.tilts_in_view()[:] = marker
        mesh.tilts_out_view()[:] = -marker
        mesh.tilts_view()[:] = 0.5 * marker
        mesh.global_parameters.set("tilt_thetaB_value", marker)
        return float(trial_positions[1, 0])

    try:
        success, _new_step, accepted_energy = backtracking_line_search_array(
            mesh,
            direction,
            gradient,
            step_size=0.5,
            energy_fn=energy_fn,
            trial_energy_fn=trial_energy_fn,
            vertex_ids=vertex_ids,
            max_iter=1,
            beta=0.5,
            c=1e-4,
            gamma=1.0,
            alpha_max_factor=1.0,
            constraint_enforcer=None,
        )
    finally:
        delattr(mesh, "_line_search_reduced_energy")
        delattr(mesh, "_line_search_reduced_accept_rule")

    assert success is True
    assert accepted_energy == 0.5
    assert trial_markers == [1.5]
    np.testing.assert_allclose(mesh.positions_view()[1], np.array([0.5, 0.0, 0.0]))
    assert np.allclose(mesh.tilts_in_view(), 1.5)
    assert np.allclose(mesh.tilts_out_view(), -1.5)
    assert np.allclose(mesh.tilts_view(), 0.75)
    assert float(mesh.global_parameters.get("tilt_thetaB_value")) == 1.5


def test_array_line_search_reduced_fast_path_restores_baseline_on_reject() -> None:
    mesh = _build_triangle_mesh()
    mesh.global_parameters = GlobalParameters()
    vertex_ids = tuple(int(v) for v in mesh.vertex_ids.tolist())
    x0 = mesh.positions_view().copy()
    mesh._line_search_reduced_energy = True
    mesh._line_search_reduced_accept_rule = "decrease_only"

    direction = np.zeros((len(vertex_ids), 3), dtype=float)
    direction[1, 0] = 1.0
    gradient = direction.copy()
    trial_markers: list[float] = []

    def energy_fn() -> float:
        np.testing.assert_allclose(mesh.positions_view(), x0, rtol=0, atol=1e-12)
        marker = float(np.sum(mesh.positions_view()))
        mesh.tilts_in_view()[:] = marker
        mesh.tilts_out_view()[:] = -marker
        mesh.tilts_view()[:] = 0.5 * marker
        mesh.global_parameters.set("tilt_thetaB_value", marker)
        return float(mesh.positions_view()[1, 0])

    def trial_energy_fn(trial_positions: np.ndarray) -> float:
        np.testing.assert_allclose(mesh.positions_view(), x0, rtol=0, atol=1e-12)
        marker = float(np.sum(trial_positions))
        trial_markers.append(marker)
        mesh.tilts_in_view()[:] = marker
        mesh.tilts_out_view()[:] = -marker
        mesh.tilts_view()[:] = 0.5 * marker
        mesh.global_parameters.set("tilt_thetaB_value", marker)
        return float(trial_positions[1, 0])

    try:
        success, _new_step, accepted_energy = backtracking_line_search_array(
            mesh,
            direction,
            gradient,
            step_size=1.0,
            energy_fn=energy_fn,
            trial_energy_fn=trial_energy_fn,
            vertex_ids=vertex_ids,
            max_iter=1,
            beta=0.5,
            c=1e-4,
            gamma=1.0,
            alpha_max_factor=1.0,
            constraint_enforcer=None,
        )
    finally:
        delattr(mesh, "_line_search_reduced_energy")
        delattr(mesh, "_line_search_reduced_accept_rule")

    assert success is False
    assert accepted_energy == 1.0
    assert trial_markers == [3.0]
    np.testing.assert_allclose(mesh.positions_view(), x0, rtol=0, atol=1e-12)
    assert np.allclose(mesh.tilts_in_view(), 2.0)
    assert np.allclose(mesh.tilts_out_view(), -2.0)
    assert np.allclose(mesh.tilts_view(), 1.0)
    assert float(mesh.global_parameters.get("tilt_thetaB_value")) == 2.0


def test_array_line_search_reduced_mode_with_constraints_falls_back() -> None:
    mesh = _build_triangle_mesh()
    vertex_ids = tuple(int(v) for v in mesh.vertex_ids.tolist())
    mesh._line_search_reduced_energy = True
    mesh._line_search_reduced_accept_rule = "decrease_only"

    direction = np.zeros((len(vertex_ids), 3), dtype=float)
    direction[1, 0] = -1.0
    gradient = direction.copy()

    def energy_fn() -> float:
        return float(mesh.positions_view()[1, 0])

    def trial_energy_fn(_trial_positions: np.ndarray) -> float:
        raise AssertionError(
            "constraint-backed reduced line search should use fallback"
        )

    try:
        success, _new_step, accepted_energy = backtracking_line_search_array(
            mesh,
            direction,
            gradient,
            step_size=0.5,
            energy_fn=energy_fn,
            trial_energy_fn=trial_energy_fn,
            vertex_ids=vertex_ids,
            max_iter=1,
            beta=0.5,
            c=1e-4,
            gamma=1.0,
            alpha_max_factor=1.0,
            constraint_enforcer=lambda _mesh: None,
        )
    finally:
        delattr(mesh, "_line_search_reduced_energy")
        delattr(mesh, "_line_search_reduced_accept_rule")

    assert success is True
    assert accepted_energy == 0.5
    np.testing.assert_allclose(mesh.positions_view()[1], np.array([0.5, 0.0, 0.0]))
