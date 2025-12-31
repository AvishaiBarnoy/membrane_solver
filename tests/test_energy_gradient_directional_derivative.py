import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _tetra_mesh_with_body() -> Mesh:
    mesh = Mesh()
    points = np.array(
        [
            [0.1, 0.2, 0.05],
            [1.1, -0.1, 0.3],
            [0.4, 1.2, -0.2],
            [0.5, 0.4, 1.5],
        ],
        dtype=float,
    )
    for i, p in enumerate(points):
        mesh.vertices[i] = Vertex(i, p)

    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    edge_map: dict[tuple[int, int], int] = {}
    next_eid = 1
    for fid, (a, b, c) in enumerate(faces):
        e_ids = []
        for tail, head in ((a, b), (b, c), (c, a)):
            key = (min(tail, head), max(tail, head))
            if key not in edge_map:
                edge_map[key] = next_eid
                mesh.edges[next_eid] = Edge(next_eid, tail, head)
                next_eid += 1
            eid = edge_map[key]
            edge = mesh.edges[eid]
            e_ids.append(eid if edge.tail_index == tail else -eid)
        mesh.facets[fid] = Facet(fid, e_ids)

    mesh.bodies[0] = Body(0, list(mesh.facets.keys()), target_volume=0.5)
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def _set_mesh_positions(mesh: Mesh, positions: np.ndarray) -> None:
    mesh.build_position_cache()
    if positions.shape != (len(mesh.vertex_ids), 3):
        raise ValueError("positions must have shape (N_vertices, 3)")

    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = positions[row]
    mesh.increment_version()


def _directional_derivative_error(
    minimizer: Minimizer,
    mesh: Mesh,
    *,
    eps: float = 1e-6,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    x0 = mesh.positions_view().copy()
    direction = rng.normal(size=x0.shape)
    direction /= float(np.linalg.norm(direction))

    _set_mesh_positions(mesh, x0)
    energy0 = minimizer.compute_energy()
    energy1, grad_dict = minimizer.compute_energy_and_gradient()
    assert float(energy1) == pytest.approx(float(energy0), rel=1e-12, abs=1e-12)

    idx_map = mesh.vertex_index_to_row
    analytic = 0.0
    for vidx, gvec in grad_dict.items():
        row = idx_map.get(vidx)
        if row is None:
            continue
        analytic += float(np.dot(gvec, direction[row]))

    _set_mesh_positions(mesh, x0 + eps * direction)
    e_plus = float(minimizer.compute_energy())
    _set_mesh_positions(mesh, x0 - eps * direction)
    e_minus = float(minimizer.compute_energy())
    numeric = (e_plus - e_minus) / (2.0 * eps)

    scale = max(1.0, abs(analytic), abs(numeric))
    return abs(analytic - numeric) / scale


@pytest.mark.parametrize(
    ("modules", "global_params"),
    [
        (
            ["surface", "volume"],
            {
                "surface_tension": 1.0,
                "volume_constraint_mode": "penalty",
                "volume_stiffness": 2.5,
            },
        ),
        (
            ["surface", "line_tension"],
            {
                "surface_tension": 0.7,
                "line_tension": 0.3,
            },
        ),
        (
            ["surface", "body_area_penalty"],
            {
                "surface_tension": 0.5,
                "area_stiffness": 3.0,
            },
        ),
    ],
)
def test_total_energy_gradient_matches_directional_derivative(modules, global_params):
    mesh = _tetra_mesh_with_body()

    if "line_tension" in modules:
        for edge in mesh.edges.values():
            edge.options.setdefault("energy", [])
            if "line_tension" not in edge.options["energy"]:
                edge.options["energy"].append("line_tension")

    if "body_area_penalty" in modules:
        mesh.bodies[0].options["area_target"] = 1.0

    gp = GlobalParameters(global_params)
    stepper = GradientDescent(max_iter=2)
    energy_manager = EnergyModuleManager(modules)
    constraint_manager = ConstraintModuleManager([])
    minimizer = Minimizer(
        mesh=mesh,
        global_params=gp,
        stepper=stepper,
        energy_manager=energy_manager,
        constraint_manager=constraint_manager,
        energy_modules=modules,
        constraint_modules=[],
        quiet=True,
    )

    rel_err = _directional_derivative_error(minimizer, mesh, eps=1e-6, seed=0)
    assert rel_err < 3e-4, f"relative directional-derivative error={rel_err:.3e}"
