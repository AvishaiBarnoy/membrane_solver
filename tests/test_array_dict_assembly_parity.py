import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from parameters.global_parameters import GlobalParameters
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
    return mesh


def test_array_and_dict_pipelines_match_directional_derivative():
    mesh = _tetra_mesh_with_body()
    gp = GlobalParameters(
        {
            "surface_tension": 1.0,
            "volume_constraint_mode": "lagrange",
            "volume_projection_during_minimization": False,
        }
    )

    energy_modules = ["surface"]
    constraint_modules = ["volume"]
    minimizer = Minimizer(
        mesh=mesh,
        global_params=gp,
        stepper=GradientDescent(max_iter=2),
        energy_manager=EnergyModuleManager(energy_modules),
        constraint_manager=ConstraintModuleManager(constraint_modules),
        energy_modules=energy_modules,
        constraint_modules=constraint_modules,
        quiet=True,
    )

    energy_arr, grad_arr = minimizer.compute_energy_and_gradient_array()
    energy_dict, grad_dict = minimizer.compute_energy_and_gradient_dict()
    assert float(energy_arr) == pytest.approx(float(energy_dict), rel=1e-12, abs=1e-12)

    positions = mesh.positions_view()
    rng = np.random.default_rng(0)
    direction = rng.normal(size=positions.shape)
    direction /= float(np.linalg.norm(direction))

    dot_arr = float(np.sum(grad_arr * direction))

    idx_map = mesh.vertex_index_to_row
    dot_dict = 0.0
    for vidx, gvec in grad_dict.items():
        row = idx_map.get(vidx)
        if row is None:
            continue
        dot_dict += float(np.dot(gvec, direction[row]))

    assert dot_arr == pytest.approx(dot_dict, rel=1e-10, abs=1e-10)
