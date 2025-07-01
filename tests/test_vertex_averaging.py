import numpy as np
import sys
import os

# Add root project directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Mesh, Vertex, Edge, Facet
from runtime.vertex_average import vertex_average

def test_vertex_averaging_smooths_mesh():
    mesh = Mesh()

    # Define vertices
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, -0.05])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([1.0, 1.0, 0.0])),
        3: Vertex(3, np.array([0.0, 1.0, 0.0])),
        4: Vertex(4, np.array([0.5, 0.5, 1.0]))  # Apex vertex
    }

    # Define edges for each triangle face of the pyramid
    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 4),
        3: Edge(3, 4, 0),
        4: Edge(4, 1, 2),
        5: Edge(5, 2, 4),
        6: Edge(6, 4, 1),
        7: Edge(7, 2, 3),
        8: Edge(8, 3, 4),
        9: Edge(9, 4, 2),
        10: Edge(10, 3, 0),
        11: Edge(11, 0, 4),
        12: Edge(12, 4, 3),
    }

    # Define facets using signed edge indices (counterclockwise)
    mesh.facets = {
        0: Facet(0, [1, 2, 3]),
        1: Facet(1, [4, 5, 6]),
        2: Facet(2, [7, 8, 9]),
        3: Facet(3, [10, 11, 12]),
    }

    # Build connectivity maps for vertex averaging
    mesh.build_connectivity_maps()

    # Record original z position of apex
    original_z = mesh.vertices[4].position[2]

    # Apply vertex averaging
    vertex_average(mesh)

    # Record new z position of apex
    new_z = mesh.vertices[4].position[2]

    # Check that apex moved downward
    assert new_z < original_z, "Apex vertex should move downward to smooth the surface"
