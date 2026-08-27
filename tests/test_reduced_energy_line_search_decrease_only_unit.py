import numpy as np
import pytest

from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.steppers.line_search import (
    backtracking_line_search,
    backtracking_line_search_array,
)


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


@pytest.mark.parametrize("frontend", ["array", "dict"])
def test_reduced_energy_decrease_only_accepts_even_if_not_descent(
    frontend: str,
) -> None:
    mesh = _build_triangle_mesh()

    rng = np.random.default_rng(0 if frontend == "array" else 1)
    n = len(mesh.vertex_ids)
    mesh.set_tilts_in_from_array(rng.normal(size=(n, 3)))
    mesh.set_tilts_out_from_array(rng.normal(size=(n, 3)))
    mesh.set_tilts_from_array(rng.normal(size=(n, 3)))

    mesh._line_search_reduced_energy = True
    mesh._line_search_reduced_accept_rule = "decrease_only"

    def energy_fn() -> float:
        pos = mesh.positions_view()
        marker = float(np.sum(pos))
        mesh.tilts_in_view()[:] = marker
        mesh.tilts_out_view()[:] = -marker
        mesh.tilts_view()[:] = 0.5 * marker
        return float(pos[0, 0])

    initial_energy = float(mesh.positions_view()[0, 0])
    if frontend == "array":
        direction = np.zeros_like(mesh.positions_view())
        direction[0, 0] = -1.0
        success, _new_step, accepted_energy = backtracking_line_search_array(
            mesh,
            direction,
            direction.copy(),  # Make the direction look non-descent (g·d > 0).
            step_size=0.5,
            energy_fn=energy_fn,
            vertex_ids=tuple(int(v) for v in mesh.vertex_ids.tolist()),
            max_iter=1,
            beta=0.5,
            c=1e-4,
            gamma=1.0,
            alpha_max_factor=1.0,
            constraint_enforcer=None,
        )
    else:
        direction = {0: np.array([-1.0, 0.0, 0.0])}
        gradient = {0: np.array([-1.0, 0.0, 0.0])}
        success, _new_step, accepted_energy = backtracking_line_search(
            mesh,
            direction,
            gradient,  # Make the direction look non-descent (g·d > 0).
            step_size=0.25,
            energy_fn=energy_fn,
            max_iter=1,
            beta=0.5,
            c=1e-4,
            gamma=1.0,
            alpha_max_factor=1.0,
            constraint_enforcer=None,
        )

    assert success is True
    assert accepted_energy < initial_energy

    expected_marker = float(np.sum(mesh.positions_view()))
    assert np.allclose(mesh.tilts_in_view(), expected_marker)
    assert np.allclose(mesh.tilts_out_view(), -expected_marker)
    assert np.allclose(mesh.tilts_view(), 0.5 * expected_marker)
