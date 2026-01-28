import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry.entities import Facet, Mesh, Vertex


def create_simple_mesh():
    # Create a simple tetrahedron
    mesh = Mesh()
    # Vertices
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[2] = Vertex(2, np.array([0.0, 1.0, 0.0]))
    mesh.vertices[3] = Vertex(3, np.array([0.0, 0.0, 1.0]))

    # Facets (just one body)
    # Using simple indices for this test
    # Facet 0: 0-2-1 (bottom)
    mesh.facets[0] = Facet(
        0, edge_indices=[]
    )  # Dummy edges for now, won't use edge-based computation if I mock it or construct carefully
    # Actually, Body.compute_volume relies on edges/facets being correct.
    # It's easier to mock the compute_volume_and_gradient method or just test the logic mechanism on Body directly
    # if I trust the math.
    # But let's build a minimal valid mesh structure to test the real integration.

    from geometry.entities import Edge

    # 0->1
    mesh.edges[1] = Edge(1, 0, 1)
    # 1->2
    mesh.edges[2] = Edge(2, 1, 2)
    # 2->0
    mesh.edges[3] = Edge(3, 2, 0)

    mesh.facets[0] = Facet(0, edge_indices=[1, 2, 3])

    # 0->3
    mesh.edges[4] = Edge(4, 0, 3)
    # 3->1
    mesh.edges[5] = Edge(5, 3, 1)
    # 1->0 (reversed 1) - wait, Body logic needs closed volume.

    # Let's just create a Body and Mock the compute logic?
    # No, integration test is better.
    # Let's use a sample mesh function if available.
    return mesh


from sample_meshes import cube_soft_volume_input

from geometry.geom_io import parse_geometry


def test_body_cache_hit_and_miss():
    # 1. Setup
    data = cube_soft_volume_input("lagrange")
    mesh = parse_geometry(data)
    body = mesh.bodies[0]

    # Initial state: geometry validation may pre-warm volume cache.
    if body._cached_volume is None:
        assert body._cached_version == -1
    else:
        assert body._cached_version == mesh._version
        assert body._cached_volume_grad is None
    initial_version = mesh._version

    # 2. First Computation (Populate Cache)
    vol1, grad1 = body.compute_volume_and_gradient(mesh)

    assert body._cached_volume is not None
    assert body._cached_version == initial_version
    # Ensure it cached the exact objects
    assert body._cached_volume == vol1
    assert body._cached_volume_grad is grad1

    # 3. Second Computation (Cache Hit)
    vol2, grad2 = body.compute_volume_and_gradient(mesh)

    # Should be identical objects
    assert vol2 == vol1
    assert grad2 is grad1

    # 4. Modify Mesh (Invalidate)
    mesh.vertices[0].position += np.array([0.1, 0.0, 0.0])
    mesh.increment_version()

    assert mesh._version == initial_version + 1

    # 5. Third Computation (Cache Miss / Recompute)
    vol3, grad3 = body.compute_volume_and_gradient(mesh)

    assert body._cached_version == mesh._version
    assert vol3 != vol1  # Volume should change
    assert grad3 is not grad1  # Should be a new gradient dict


def test_manual_version_increment():
    mesh = Mesh()
    v = mesh._version
    mesh.increment_version()
    assert mesh._version == v + 1


def test_stepper_increments_version():
    """Verify that the line search (used by steppers) increments mesh version."""
    from runtime.steppers.line_search import backtracking_line_search

    data = cube_soft_volume_input("lagrange")
    mesh = parse_geometry(data)
    # Ensure vertex 0 is not fixed so it can move
    for v in mesh.vertices.values():
        v.fixed = False

    initial_version = mesh._version

    # Create a simple energy function: E = (x - 1)^2 for vertex 0
    # x0 = 0 initially. E = 1.
    def energy_fn():
        x = mesh.vertices[0].position[0]
        return (x - 1.0) ** 2

    # Gradient at x=0 is -2.
    # Direction +1 (towards minimum).
    direction = {0: np.array([1.0, 0.0, 0.0])}
    gradient = {0: np.array([-2.0, 0.0, 0.0])}

    # Run line search
    # This should succeed and modify the mesh
    success, _ = backtracking_line_search(
        mesh, direction, gradient, step_size=0.1, energy_fn=energy_fn
    )

    assert success
    # Version should have incremented (at least once)
    assert mesh._version > initial_version


def test_triangle_row_cache_reused_across_position_updates():
    data = cube_soft_volume_input("lagrange")
    mesh = parse_geometry(data)
    mesh.build_facet_vertex_loops()

    tri_rows_1, facets_1 = mesh.triangle_row_cache()
    assert tri_rows_1 is not None

    mesh.vertices[0].position += np.array([0.1, 0.0, 0.0])
    mesh.increment_version()

    tri_rows_2, facets_2 = mesh.triangle_row_cache()
    assert tri_rows_2 is tri_rows_1
    assert facets_2 == facets_1


def test_curvature_cache_reused_during_geometry_freeze():
    data = cube_soft_volume_input("lagrange")
    mesh = parse_geometry(data)
    mesh.build_facet_vertex_loops()

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    from geometry.curvature import compute_curvature_data

    with mesh.geometry_freeze(positions):
        k0, a0, w0, t0 = compute_curvature_data(mesh, positions, index_map)
        k1, a1, w1, t1 = compute_curvature_data(mesh, positions, index_map)

    assert k1 is k0
    assert a1 is a0
    assert w1 is w0
    assert t1 is t0

    mesh.vertices[0].position += np.array([0.1, 0.0, 0.0])
    mesh.increment_version()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    k2, a2, w2, t2 = compute_curvature_data(mesh, positions, index_map)
    assert k2 is not k0
    assert a2 is not a0
    assert w2 is not w0
    assert t2 is t0


def test_vertex_average_increments_version():
    from runtime.vertex_average import vertex_average

    data = cube_soft_volume_input("lagrange")
    mesh = parse_geometry(data)
    initial_version = mesh._version

    vertex_average(mesh)

    assert mesh._version > initial_version


def test_constraint_enforcement_increments_version():
    from modules.constraints import volume

    data = cube_soft_volume_input("lagrange")
    mesh = parse_geometry(data)

    # Set up a target volume that requires adjustment
    body = mesh.bodies[0]
    current_vol = body.compute_volume(mesh)
    body.target_volume = current_vol * 1.1  # Force a change

    initial_version = mesh._version

    # Enforce
    volume.enforce_constraint(mesh)

    assert mesh._version > initial_version
