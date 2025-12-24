import os
import sys
import time

import numpy as np

# Ensure we can import from the project root
sys.path.append(os.getcwd())

from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from modules.energy import volume
from parameters.global_parameters import GlobalParameters
from runtime.refinement import refine_triangle_mesh


def create_sphere_mesh(subdivisions=4):
    """Creates a simple icosphere-like mesh by refining a cube projected to a sphere."""
    mesh = Mesh()

    # Vertices of a cube
    # Ensure not at origin
    points = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
    for i, p in enumerate(points):
        mesh.vertices[i] = Vertex(i, np.array(p, dtype=float))

    # Facets (triangles)
    # 0-1-2-3 is bottom (z=-1), 4-5-6-7 is top (z=1)
    # Proper orientation (CCW outside)
    tri_indices = [
        [0, 2, 1],
        [0, 3, 2],  # Bottom
        [4, 5, 6],
        [4, 6, 7],  # Top
        [0, 1, 5],
        [0, 5, 4],  # Front
        [1, 2, 6],
        [1, 6, 5],  # Right
        [2, 3, 7],
        [2, 7, 6],  # Back
        [3, 0, 4],
        [3, 4, 7],  # Left
    ]

    # Create Edges
    # We need a map to avoid duplicating edges: (min, max) -> edge_index
    edge_map = {}
    next_edge_id = 1

    def get_edge(u, v):
        nonlocal next_edge_id
        key = tuple(sorted((u, v)))
        if key not in edge_map:
            edge_map[key] = next_edge_id
            mesh.edges[next_edge_id] = Edge(next_edge_id, u, v)
            next_edge_id += 1
        return edge_map[key]

    # Create Facets and link edges
    for i, (a, b, c) in enumerate(tri_indices):
        e1 = get_edge(a, b)
        e2 = get_edge(b, c)
        e3 = get_edge(c, a)

        # Determine signs based on orientation
        # e1: u->v. If (a,b) matches (u,v), sign +1, else -1
        def sign(idx, start, end):
            e = mesh.edges[idx]
            return idx if e.tail_index == start else -idx

        signed_edges = [sign(e1, a, b), sign(e2, b, c), sign(e3, c, a)]

        mesh.facets[i] = Facet(i, signed_edges)

    # Create Body
    mesh.bodies[0] = Body(0, list(mesh.facets.keys()), target_volume=1.0)

    # Project to sphere
    for v in mesh.vertices.values():
        n = np.linalg.norm(v.position)
        if n > 1e-9:
            v.position /= n

    # Refine
    # Ensure connectivity for refiner
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()

    for _ in range(subdivisions):
        mesh = refine_triangle_mesh(mesh)
        # Project to sphere again
        for v in mesh.vertices.values():
            n = np.linalg.norm(v.position)
            if n > 1e-9:
                v.position /= n

    # Final setup
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    mesh.triangle_row_cache()

    return mesh


def benchmark():
    print("Preparing mesh...")
    # 4 subdivisions on a cube ~ 12 * 4^4 = 3072 facets.
    # 5 subdivisions ~ 12 * 4^5 = 12288 facets. Good for benchmarking.
    mesh = create_sphere_mesh(subdivisions=5)

    print(f"Mesh stats: {len(mesh.vertices)} vertices, {len(mesh.facets)} facets.")

    # Setup parameters
    gp = GlobalParameters()
    gp.volume_stiffness = 10.0
    gp.set("volume_constraint_mode", "penalty")

    # Pre-computation setup
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)

    class MockResolver:
        def get(self, obj, name):
            return None

    resolver = MockResolver()

    # --- Warmup ---
    volume.compute_energy_and_gradient_array(
        mesh, gp, resolver, positions=positions, index_map=index_map, grad_arr=grad_arr
    )
    volume.compute_energy_and_gradient(mesh, gp, resolver, compute_gradient=True)

    iterations = 100

    # --- Benchmark New (Array) Path ---
    start_time = time.perf_counter()
    for _ in range(iterations):
        grad_arr.fill(0.0)
        volume.compute_energy_and_gradient_array(
            mesh,
            gp,
            resolver,
            positions=positions,
            index_map=index_map,
            grad_arr=grad_arr,
        )
    end_time = time.perf_counter()
    avg_new = (end_time - start_time) / iterations
    print(f"\nNew Array-based Path: {avg_new * 1000:.4f} ms per iteration")

    # --- Benchmark Old (Dict) Path ---
    # This calls compute_energy_and_gradient which returns a dictionary
    # mimicking the overhead of the old system.
    start_time = time.perf_counter()
    for _ in range(iterations):
        volume.compute_energy_and_gradient(mesh, gp, resolver, compute_gradient=True)
    end_time = time.perf_counter()
    avg_old = (end_time - start_time) / iterations
    print(f"Old Dict-based Path:  {avg_old * 1000:.4f} ms per iteration")

    speedup = avg_old / avg_new
    print(f"\nSpeedup: {speedup:.2f}x")


if __name__ == "__main__":
    benchmark()
