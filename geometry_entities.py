# geometry.py
import numpy as np

class Vertex:
    def __init__(self, position, index, options=None):
        self.position = np.array(position, dtype=float)
        self.options = options if options is not None else {}
        self.index = index

        self.force = np.zeros(3, dtype=float)
        # For conjugate gradient updates:
        self.prev_force = np.zeros(3, dtype=float)
        self.search_direction = np.zeros(3, dtype=float)
        self.initialized_cg = False  # Flag to initialize the search direction

    def __repr__(self):
        return (f"Vertex(idx={self.index}, pos={self.position}), options={self.options})")

class Edge:
    def __init__(self, tail, head, vector=None, options=None):
        """
        An edge is a one-dimensional geometric element
        It has an orientation, but the orientation is only important in the
            sense of of defining a facet, for the facet normal.
        """
        # Store vertex indices (or references) for the edge endpoints.
        self.tail = tail
        self.head = head
        self.vector = vector
        self.options = options if options is not None else {}

    def compute_vector(self):
         """Compute the vector representing the edge."""
         return np.array(self.head.coordinates) - np.array(self.tail.coordinates)

    def length(self):
        """Compute the length of the edge."""
        return np.linalg.norm(self.vector())

    def __repr__(self):
        edge_repr = f"Edge({self.tail.position.tolist()}→{self.head.position.tolist()}, options={self.options})"
        return edge_repr

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

    def __repr__(self):
        # TODO: change to use the Edge.__repr__ instead of redfining it again
        edge_repr = ','.join([f"{e.tail.position.tolist()}→{e.head.position.tolist()}"
                               for e in self.edges])
        return f"Facet(edges=[{edge_repr}], options={self.options})"

    def calculate_area(self, edges):
        """Calculates the area of the facet assuming it is a triangle"""
        # TODO: Tests: check calculation is the same no matter vector choice
        # Assume edges are cyclically ordered (e1: v0->v1, e2: v1->v2, e3: v2->v0)
        edge_vec1 = self.edges[0].vector()
        edge_vec2 = self.edges[1].vector()

        area = 0.5 * np.linalg.norm(np.cross(edge_vec1, edge_vec2))

        return area

class Body:
    def __init__(self, facets, volume=None, target_volume=None,
                 surface_area=None, options=None):
        # A volume is defined by a collection of facets.
        self.facets = facets if facets is not None else []
        self.options = options if options is not None else {}
        self.volume = None          # TODO: should be read from option
        self.target_volume = None   # TODO: should be read from options

    def calculate_volume(self):
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

        # TODO: implement old version of volume calculation:
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
        """
        # TODO: change shape in geometry file and recalculate to make sure volume calculation is correct
        volume = 0.0
        for facet in self.facets:
            if len(facet.edges) < 3:
                continue

            # Reconstruct vertex loop from edges
            verts = []
            for edge in facet.edges:
                if not verts or verts[-1] != edge.tail:
                    verts.append(edge.tail)
            if verts[0] != facet.edges[-1].head:
                verts.append(facet.edges[-1].head)

            if len(verts) < 3:
                continue  # skip invalid facet

            # Compute signed area projected on xy-plane
            A_proj = 0.0
            for i in range(len(verts)):
                x0, y0 = verts[i].position[:2]
                x1, y1 = verts[(i + 1) % len(verts)].position[:2]
                A_proj += (x0 * y1 - x1 * y0)
            A_proj *= 0.5

            # Average z height of the polygon
            z_avg = sum(v.position[2] for v in verts) / len(verts)

            volume += A_proj * z_avg

        self.volume = volume
        return volume

        volume = 0.0
        for facet in self.facets:
            # Assuming facets are triangular (after initial triangulation)
            if len(facet.edges) != 3:
                continue  # or raise an error if non-triangular facets are not allowed

            # Get triangle vertices
            v1 = facet.edges[0].tail.position
            v2 = facet.edges[0].head.position
            v3 = facet.edges[1].head.position  # assume CCW orientation

            # Signed volume of the tetrahedron formed by (0, v1, v2, v3)
            v = np.dot(np.cross(v1, v2), v3) / 6.0

            volume += v

        self.volume = abs(volume)
        return self.volume

    def calculate_surface_area(self):
        for facet in self.facets:
            self.surface_area += facet.calculate_area()
        return self.surface_area
