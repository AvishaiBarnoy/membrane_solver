import time

import numpy as np

from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from parameters.global_parameters import GlobalParameters
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.refinement import refine_triangle_mesh
from runtime.steppers.gradient_descent import GradientDescent


def create_dented_sphere(subdivisions=3):
    """Creates a sphere with a significant 'dent' to test bending relaxation."""
    mesh = Mesh()
    # Cube base
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

    for _ in range(subdivisions):
        for v in mesh.vertices.values():
            v.position /= np.linalg.norm(v.position)
        mesh = refine_triangle_mesh(mesh)

    # Project to unit sphere
    for v in mesh.vertices.values():
        v.position /= np.linalg.norm(v.position)

    # Introduce a DENT: push all vertices near the top pole (0,0,1) inwards
    for v in mesh.vertices.values():
        if v.position[2] > 0.5:
            # Scale the radius down for the top section
            dist_to_pole = np.sqrt(v.position[0] ** 2 + v.position[1] ** 2)
            if dist_to_pole < 0.5:
                v.position *= 0.7  # Significant dent

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    # Add a body for volume constraint (optional but keeps physics consistent)
    mesh.bodies[0] = Body(0, list(mesh.facets.keys()), target_volume=None)
    return mesh


def benchmark():
    mesh = create_dented_sphere(subdivisions=3)

    gp = GlobalParameters()
    gp.set("bending_modulus", 1.0)
    gp.set("surface_tension", 0.0)  # ONLY Bending
    gp.set("volume_constraint_mode", "lagrange")  # Keep volume fixed
    gp.set("volume_projection_during_minimization", True)
    gp.set("bending_gradient_mode", "analytic")

    # Target volume is the current (dented) volume
    mesh.bodies[0].target_volume = mesh.bodies[0].compute_volume(mesh)

    em = EnergyModuleManager(["bending"])
    cm = ConstraintModuleManager(["volume"])
    stepper = GradientDescent()

    minimizer = Minimizer(mesh, gp, stepper, em, cm)

    n_steps = 20
    start_time = time.perf_counter()
    minimizer.minimize(n_steps=n_steps)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / n_steps
    return avg_time


if __name__ == "__main__":
    t = benchmark()
    print(f"Average step time (Bending): {t:.4f}s")
