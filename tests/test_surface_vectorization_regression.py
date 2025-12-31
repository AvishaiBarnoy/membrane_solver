import os
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from modules.energy import surface
from parameters.global_parameters import GlobalParameters
from parameters.resolver import ParameterResolver


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


def test_surface_energy_avoids_per_facet_area_calls_for_triangle_meshes():
    mesh = _tetra_mesh_with_body()
    gp = GlobalParameters({"surface_tension": 1.0})

    with patch(
        "geometry.entities.Facet.compute_area",
        autospec=True,
        side_effect=AssertionError("surface energy regressed to per-facet loop"),
    ):
        energy = surface.calculate_surface_energy(mesh, gp)

    assert float(energy) > 0.0


def test_surface_gradient_avoids_per_facet_gradient_calls_for_triangle_meshes():
    mesh = _tetra_mesh_with_body()
    gp = GlobalParameters({"surface_tension": 1.0})
    resolver = ParameterResolver(gp)

    with patch(
        "geometry.entities.Facet.compute_area_and_gradient",
        autospec=True,
        side_effect=AssertionError(
            "surface gradient regressed to per-facet compute_area_and_gradient"
        ),
    ):
        energy, grad = surface.compute_energy_and_gradient(mesh, gp, resolver)

    assert float(energy) > 0.0
    assert set(grad.keys()) == set(mesh.vertices.keys())
