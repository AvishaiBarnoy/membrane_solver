import numpy as np
import pytest

from geometry.entities import Edge, Facet, Mesh, Vertex
from modules.energy import bending
from parameters.global_parameters import GlobalParameters
from parameters.resolver import ParameterResolver


def create_sphere_mesh(subdivisions=3):
    """Creates a simple sphere by refining a cube."""
    from runtime.refinement import refine_triangle_mesh

    mesh = Mesh()

    # Unit cube vertices
    points = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )

    for i, p in enumerate(points):
        mesh.vertices[i] = Vertex(i, p)

    # Triangulated faces
    tri_indices = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]

    edge_map = {}
    next_eid = 1
    for i, (a, b, c) in enumerate(tri_indices):
        e_ids = []
        for pair in [(a, b), (b, c), (c, a)]:
            key = tuple(sorted(pair))
            if key not in edge_map:
                edge_map[key] = next_eid
                mesh.edges[next_eid] = Edge(next_eid, pair[0], pair[1])
                next_eid += 1
            eid = edge_map[key]
            e_ids.append(eid if mesh.edges[eid].tail_index == pair[0] else -eid)
        mesh.facets[i] = Facet(i, e_ids)

    # Project to unit sphere and refine
    for _ in range(subdivisions):
        for v in mesh.vertices.values():
            v.position /= np.linalg.norm(v.position)
        mesh = refine_triangle_mesh(mesh)

    for v in mesh.vertices.values():
        v.position /= np.linalg.norm(v.position)

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def test_sphere_willmore_energy():
    """Verify that the bending energy of a sphere is approx 4*pi."""
    mesh = create_sphere_mesh(subdivisions=3)  # ~1.5k facets
    gp = GlobalParameters()
    gp.bending_modulus = 1.0
    gp.bending_energy_model = "willmore"
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)

    energy = bending.compute_energy_and_gradient_array(
        mesh, gp, resolver, positions=positions, index_map=index_map, grad_arr=grad_arr
    )

    # Expected Willmore energy of a sphere is 4*pi
    expected = 4 * np.pi
    print(f"Sphere Energy: {energy:.4f}, Expected: {expected:.4f}")

    # Allow some error due to discretization
    assert energy == pytest.approx(expected, rel=0.05)


def test_flat_plane_energy_is_zero():
    """Verify that a flat plane has zero bending energy."""
    mesh = Mesh()
    # Simple grid
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            mesh.vertices[idx] = Vertex(idx, np.array([i, j, 0.0], dtype=float))

    # Two triangles
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 4)
    mesh.edges[3] = Edge(3, 4, 0)
    mesh.facets[0] = Facet(0, [1, 2, 3])

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()

    gp = GlobalParameters()
    resolver = ParameterResolver(gp)
    energy, _ = bending.compute_energy_and_gradient(mesh, gp, resolver)

    assert energy == pytest.approx(0.0, abs=1e-10)
