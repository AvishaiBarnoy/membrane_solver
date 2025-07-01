from geometry.entities import Mesh, Vertex, Facet
from runtime.vertex_average import vertex_average
import numpy as np

def test_vertex_averaging_smooths_mesh():
    mesh = Mesh()

    # Make a simple pyramid with base in XY, top at z=1
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([1.0, 1.0, 0.0])),
        3: Vertex(3, np.array([0.0, 1.0, 0.0])),
        4: Vertex(4, np.array([0.5, 0.5, 1.0]))  # apex
    }

    # Link facets (sides of pyramid)
    mesh.facets = {
        0: Facet(0, [0, 1, 4]),
        1: Facet(1, [1, 2, 4]),
        2: Facet(2, [2, 3, 4]),
        3: Facet(3, [3, 0, 4])
    }

    for f_id, facet in mesh.facets.items():
        for v_id in facet.vertex_ids:
            mesh.vertex_to_facets.setdefault(v_id, set()).add(f_id)

    original_z = mesh.vertices[4].position[2]
    vertex_average(mesh)
    new_z = mesh.vertices[4].position[2]

    assert new_z <= original_z, "Apex vertex should not move upward"
