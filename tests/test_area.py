"""Consolidated tests for area constraints (global, body, and facet)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from modules.constraints import body_area as body_area_constraint
from modules.constraints import fix_facet_area as facet_area_constraint
from modules.constraints import global_area as global_area_constraint

# --- Helpers ---


def build_tetra_mesh() -> Mesh:
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2 = Vertex(2, np.array([0.0, 1.0, 0.0]))
    v3 = Vertex(3, np.array([0.0, 0.0, 1.0]))
    vertices = {0: v0, 1: v1, 2: v2, 3: v3}
    edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 0),
        4: Edge(4, 0, 3),
        5: Edge(5, 1, 3),
        6: Edge(6, 2, 3),
    }
    facets = {
        0: Facet(0, [1, 2, 3]),
        1: Facet(1, [1, 5, -4]),
        2: Facet(2, [3, 6, -4]),
        3: Facet(3, [2, 6, -5]),
    }
    body = Body(0, [0, 1, 2, 3], target_volume=None, options={})
    mesh = Mesh(vertices=vertices, edges=edges, facets=facets, bodies={0: body})
    mesh.build_connectivity_maps()
    return mesh


def build_square_patch(scale: float = 1.0) -> Mesh:
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([scale, 0.0, 0.0]))
    v2 = Vertex(2, np.array([scale, scale, 0.0]))
    v3 = Vertex(3, np.array([0.0, scale, 0.0]))
    e0 = Edge(1, 0, 1)
    e1 = Edge(2, 1, 2)
    e2 = Edge(3, 2, 3)
    e3 = Edge(4, 3, 0)
    facet = Facet(0, [1, 2, 3, 4])
    body = Body(0, [0])
    mesh = Mesh()
    mesh.vertices = {0: v0, 1: v1, 2: v2, 3: v3}
    mesh.edges = {1: e0, 2: e1, 3: e2, 4: e3}
    mesh.facets = {0: facet}
    mesh.bodies = {0: body}
    mesh.build_connectivity_maps()
    return mesh


def build_single_facet_mesh() -> Mesh:
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2 = Vertex(2, np.array([0.0, 1.0, 0.0]))
    vertices = {0: v0, 1: v1, 2: v2}
    edges = {1: Edge(1, 0, 1), 2: Edge(2, 1, 2), 3: Edge(3, 2, 0)}
    facet = Facet(0, [1, 2, 3], options={})
    mesh = Mesh(vertices=vertices, edges=edges, facets={0: facet}, bodies={})
    mesh.build_connectivity_maps()
    return mesh


# --- Tests ---


def test_global_area_constraint_enforces_target():
    mesh = build_tetra_mesh()
    initial_area = mesh.compute_total_surface_area()
    target_area = initial_area * 1.1
    mesh.global_parameters.set("target_surface_area", target_area)
    global_area_constraint.enforce_constraint(mesh, tol=1e-10, max_iter=20)
    final_area = mesh.compute_total_surface_area()
    assert abs(final_area - target_area) < 1e-5


def test_body_area_constraint_enforces_target_on_tetrahedron():
    mesh = build_tetra_mesh()
    body = mesh.bodies[0]
    initial_area = body.compute_surface_area(mesh)
    target_area = initial_area * 0.8
    body.options["target_area"] = target_area
    body_area_constraint.enforce_constraint(mesh, tol=1e-10, max_iter=20)
    final_area = body.compute_surface_area(mesh)
    assert abs(final_area - target_area) < 1e-6


def test_facet_area_constraint_enforces_target():
    mesh = build_single_facet_mesh()
    facet = mesh.facets[0]
    initial_area = facet.compute_area(mesh)
    target_area = initial_area * 0.5
    facet.options["target_area"] = target_area
    facet_area_constraint.enforce_constraint(mesh, tol=1e-10, max_iter=50)
    final_area = facet.compute_area(mesh)
    assert abs(final_area - target_area) < 1e-6


def test_body_area_constraint_noop_at_target():
    mesh = build_square_patch(scale=1.0)
    body = mesh.bodies[0]
    area0 = body.compute_surface_area(mesh)
    body.options["target_area"] = area0
    positions_before = {vidx: v.position.copy() for vidx, v in mesh.vertices.items()}
    body_area_constraint.enforce_constraint(mesh)
    area_after = body.compute_surface_area(mesh)
    assert np.isclose(area_after, area0, rtol=1e-12, atol=1e-12)
    for vidx, v in mesh.vertices.items():
        assert np.allclose(v.position, positions_before[vidx])


def test_body_area_constraint_converges_to_target_on_square():
    mesh = build_square_patch(scale=1.1)
    body = mesh.bodies[0]
    target_area = 1.0
    body.options["target_area"] = target_area
    area_before = body.compute_surface_area(mesh)
    assert not np.isclose(area_before, target_area, rtol=1e-6)
    for _ in range(5):
        body_area_constraint.enforce_constraint(mesh)
    area_after = body.compute_surface_area(mesh)
    assert abs(area_after - target_area) < 1e-3
