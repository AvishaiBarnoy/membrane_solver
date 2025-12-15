import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex
from modules.constraints import perimeter as perimeter_constraint


def build_square_loop() -> Mesh:
    v0 = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1 = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2 = Vertex(2, np.array([1.0, 1.0, 0.0]))
    v3 = Vertex(3, np.array([0.0, 1.0, 0.0]))
    vertices = {0: v0, 1: v1, 2: v2, 3: v3}

    edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 3),
        4: Edge(4, 3, 0),
    }
    facets = {0: Facet(0, [1, 2, 3, 4])}
    mesh = Mesh(vertices=vertices, edges=edges, facets=facets, bodies={})
    mesh.global_parameters.set(
        "perimeter_constraints",
        [
            {
                "edges": [1, 2, 3, 4],
                "target_perimeter": 4.0,
            }
        ],
    )
    return mesh


def test_perimeter_constraint_enforces_length():
    mesh = build_square_loop()
    # Distort the square
    mesh.vertices[2].position += np.array([0.2, 0.5, 0.0])
    mesh.vertices[3].position += np.array([-0.1, 0.3, 0.0])

    initial_perimeter = 0.0
    for idx in [1, 2, 3, 4]:
        e = mesh.edges[idx]
        tail = mesh.vertices[e.tail_index].position
        head = mesh.vertices[e.head_index].position
        initial_perimeter += np.linalg.norm(head - tail)
    assert not np.isclose(initial_perimeter, 4.0)

    perimeter_constraint.enforce_constraint(mesh, tol=1e-6, max_iter=50)

    final_perimeter = 0.0
    for idx in [1, 2, 3, 4]:
        e = mesh.edges[idx]
        tail = mesh.vertices[e.tail_index].position
        head = mesh.vertices[e.head_index].position
        final_perimeter += np.linalg.norm(head - tail)

    assert abs(final_perimeter - 4.0) < 1e-4
