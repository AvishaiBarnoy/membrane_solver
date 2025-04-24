# geometry.py
import numpy as np
import sys

class Vertex:
    def __init__(self, position, index, options=None):
        assert isinstance(index, int), f"Expected int index, got {index}"
        self.index = index
        self.position = np.array(position, dtype=float)
        self.options = options if options is not None else {}

        self.force = np.zeros(3, dtype=float)
        # For conjugate gradient updates:
        self.prev_force = np.zeros(3, dtype=float)
        self.search_direction = np.zeros(3, dtype=float)
        self.initialized_cg = False  # Flag to initialize the search direction

    def __repr__(self):
        return (f"Vertex(idx={self.index}, pos={self.position}), options={self.options})")

class Edge:
    def __init__(self, tail, head, index, reverse=False, vector=None, options=None):
        """
        An edge is a one-dimensional geometric element
        It has an orientation, but the orientation is only important in the
            sense of of defining a facet, for the facet normal.
        """
        # Store vertex indices (or references) for the edge endpoints.
        # TODO: add asserts that tail and head are Vertex instances
        self.reverse = reverse
        self.tail = tail if not reverse else head
        self.head = head if not reverse else tail
        self.index = index
        self.vector = vector
        self.options = options if options is not None else {}

    def compute_vector(self):
         """Compute the vector representing the edge."""
         return np.array(self.head.coordinates) - np.array(self.tail.coordinates)

    def length(self):
        """Compute the length of the edge."""
        return np.linalg.norm(self.vector())

    def __repr__(self):
        edge_repr = f"Edge(idx={self.index}, {self.tail.position.tolist()}→{self.head.position.tolist()}, options={self.options})"
        return edge_repr

class Facet:
    def __init__(self, edges, index, options=None):
        """
        A facet is defined by a set of oriented (for normal direction) edges

        Args:
            indices (list or tuple of int): Vertex indices defining the facet.
            options (dict, optional): Dictionary of facet-specific options.
        """
        # TODO: add asserts that all edges are Edge instances
        self.edges = edges  # list of edges instances
        self.index = index
        self.area = None
        self.options = options if options is not None else {}

    def __repr__(self):
        edge_repr = ','.join([f"{e.tail.position.tolist()}→{e.head.position.tolist()}"
                               for e in self.edges])
        edge_indices = ','.join([str(e.index) for e in self.edges])
        # sys.exit(1)
        return f"Facet(idx={self.index}, edges=[{edge_indices}],\nedges=[{edge_repr}],\noptions={self.options})"

    def calculate_area(self, edges):
        """Calculates the area of the facet assuming it is a triangle"""
        # TODO: Tests: check calculation is the same no matter vector choice
        # Assume edges are cyclically ordered (e1: v0->v1, e2: v1->v2, e3: v2->v0)
        edge_vec1 = self.edges[0].vector()
        edge_vec2 = self.edges[1].vector()

        area = 0.5 * np.linalg.norm(np.cross(edge_vec1, edge_vec2))

        return area

class Body:
    def __init__(self, facets, index, volume=None, target_volume=None,
                 surface_area=None, options=None):
        # A volume is defined by a collection of facets.
        # TODO: add asserts that all facets are Facet instances
        self.facets = facets if facets is not None else []
        self.index = index
        self.options = options if options is not None else {}
        self.volume = None
        self.target_volume = None

    def __repr__(self):
        body_repr = f"facets: {','.join([str(f.index) for f in self.facets])}"
        return f"Body(idx={self.index}, facets=[{body_repr}], volume={self.volume}, options={self.options})"

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

