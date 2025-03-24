# geometry.py
import numpy as np

class Vertex:
    def __init__(self, position, fixed=False):
        self.position = np.array(position, dtype=float)
        self.fixed = fixed  # Handle the fixed attribute
        self.force = np.zeros(3, dtype=float)
        # For conjugate gradient updates:
        self.prev_force = np.zeros(3, dtype=float)
        self.search_direction = np.zeros(3, dtype=float)
        self.initialized_cg = False  # Flag to initialize the search direction

class Edge:
    def __init__(self, v1, v2, options=None):
        """
        An edge is a one-dimensional geometric element
        """
        # Store vertex indices (or references) for the edge endpoints.
        # TODO: implement orientability of edge
        self.v1 = v1
        self.v2 = v2
        self.options = options if options is not None else {}

        def vector(self):
            """Compute the vector representing the edge."""
            return np.array(self.head_vertex.coordinates) - np.array(self.tail_vertex.coordinates)

        def length(self):
            """Compute the length of the edge."""
            return np.linalg.norm(self.vector())

class Facet:
    def __init__(self, edges, options=None):
        """
        A facet is defined by a set of oriented (for normal direction) edges

        Args:
            indices (list or tuple of int): Vertex indices defining the facet.
            options (dict, optional): Dictionary of facet-specific options.
        """
        self.edges = edges  # list of edges instances 
        self.area = None
        self.options = options if options is not None else {}

    def calculate_area(self, edges):
        """Calculates the area of the facet assuming it is a triangle"""
        # TODO: Tests: check calculation is the same no matter vector choice
        # Assume edges are cyclically ordered (e1: v0->v1, e2: v1->v2, e3: v2->v0)
        edge_vec1 = self.edges[0].vector()
        edge_vec2 = self.edges[1].vector()

        area = 0.5 * np.linalg.norm(np.cross(edge_vec1, edge_vec2))

        return area

class Body:
    def __init__(self, facets=None, target_volume=None):
        # A volume is defined by a collection of facets.
        self.facets = facets if facets is not None else []
        self.volume = None
        self.target_volume = None

    def calculate_volume(self, vertices):
        """
        Calculate the volume of the object from its surface information.

        For each triangular facet, the function computes the volume of the vertical
        prism between the facet and the z = 0 plane. This is done by:
          1. Projecting the facet onto the xy-plane to obtain a signed area.
          2. Calculating the average z-value of the facet's vertices.
          3. Multiplying the signed projected area by the average z-value.

        The sum of these contributions over all facets gives the total volume.
        Facets that are vertical or lie at z = 0 contribute zero, as desired.

        Args:
            vertices (list of Vertex): The list of vertex objects.

        Returns:
            float: The calculated volume.
        """
        volume = 0.0
        for facet in self.facets:
            # Assuming facets are triangular (after initial triangulation)
            if len(facet.indices) != 3:
                continue  # or raise an error if non-triangular facets are not allowed
            i1, i2, i3 = facet.indices
            v1, v2, v3 = vertices[i1].position, vertices[i2].position, vertices[i3].position

            # Compute the signed area of the triangle's projection onto the xy-plane.
            # The formula for a triangle with vertices (x1,y1), (x2,y2), (x3,y3) is:
            # A_proj = 0.5 * [(x1*y2 + x2*y3 + x3*y1) - (y1*x2 + y2*x3 + y3*x1)]
            A_proj = 0.5 * ((v1[0] * v2[1] + v2[0] * v3[1] + v3[0] * v1[1]) -
                            (v1[1] * v2[0] + v2[1] * v3[0] + v3[1] * v1[0]))

            # Compute the average z-coordinate of the facet's vertices.
            z_avg = (v1[2] + v2[2] + v3[2]) / 3.0

            # The volume contribution is the projected area times the average height.
            volume += A_proj * z_avg
        self.volume = volume

