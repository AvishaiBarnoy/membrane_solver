import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.topology import (
    check_max_normal_change,
    detect_vertex_edge_collisions,
    get_min_edge_length,
)


def create_triangle_mesh():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[2] = Vertex(2, np.array([0.0, 1.0, 0.0]))

    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)

    mesh.facets[0] = Facet(0, [1, 2, 3])

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh

def test_get_min_edge_length():
    mesh = create_triangle_mesh()
    # Edges lengths: 1.0, sqrt(2)=1.414, 1.0. Min is 1.0.
    min_len = get_min_edge_length(mesh)
    assert np.isclose(min_len, 1.0)

    # Modify vertex to shorten an edge
    mesh.vertices[1].position = np.array([0.1, 0.0, 0.0])
    # Edge 0 length is now 0.1
    min_len = get_min_edge_length(mesh)
    assert np.isclose(min_len, 0.1)

def test_check_max_normal_change_safe():
    mesh = create_triangle_mesh()
    original_positions = {v.index: v.position.copy() for v in mesh.vertices.values()}

    # Slight perturbation (z-shift) -> small rotation
    mesh.vertices[2].position[2] += 0.1

    is_safe = check_max_normal_change(mesh, original_positions, limit_radians=0.5)
    assert is_safe

def test_check_max_normal_change_unsafe():
    mesh = create_triangle_mesh()
    original_positions = {v.index: v.position.copy() for v in mesh.vertices.values()}

    # Large perturbation (flip normal almost 90 degrees or more)
    # Move vertex 2 deep into negative Z?
    # Original normal is (0,0,1).
    # If V2 moves to (0, 1, -100), normal will point mostly sideways/down.
    mesh.vertices[2].position[2] = -10.0

    is_safe = check_max_normal_change(mesh, original_positions, limit_radians=0.5)
    assert not is_safe

def test_detect_vertex_edge_collisions():
    mesh = Mesh()
    # Edge along X axis
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([10.0, 0.0, 0.0]))
    mesh.edges[1] = Edge(1, 0, 1)

    # Vertex colliding (midpoint, very close in Y)
    mesh.vertices[2] = Vertex(2, np.array([5.0, 0.0001, 0.0]))

    # Vertex safe (far away)
    mesh.vertices[3] = Vertex(3, np.array([5.0, 1.0, 0.0]))

    # Update cache
    mesh.build_position_cache()

    collisions = detect_vertex_edge_collisions(mesh, threshold=0.01)

    # Should detect collision between Vertex 2 and Edge 0
    assert (2, 1) in collisions

    # Should NOT detect Vertex 3
    assert (3, 1) not in collisions
