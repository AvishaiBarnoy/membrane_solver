# entities.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from parameters.global_parameters import GlobalParameters
import numpy as np
import sys

# TODO: think about defining a class GeometryCalculationsMixin to keep 
#   Instead of putting .compute_area(), .compute_volume(), .compute_length() directly inside your core data classes (Vertex, Edge, Facet, Body...),
#   you could keep the core classes "pure" (only holding structure and basic behaviors)
#   and put all calculations into a separate Mixin class (or even in geometry_calculations.py).

# TODO: build a proper MeshError custom exception for better error reporting

@dataclass
class Vertex:
    index: int
    position: np.ndarray
    fixed: bool = False
    options: Dict[str, Any] = field(default_factory=dict)

    def project_position(self, pos: np.ndarray) -> np.ndarray:
        """
        Project the given position onto the constraint, if any.
        If no constraint is defined, return the position unchanged.
        """
        if 'constraint' in self.options:
            constraint = self.options['constraint']
            return constraint.project_position(pos)
        return pos

    def project_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        Project the given gradient into the tangent space of the constraint, if any.
        If no constraint is defined, return the gradient unchanged.
        """
        if 'constraint' in self.options:
            constraint = self.options['constraint']
            return constraint.project_gradient(grad)
        return grad

    def compute_distance(self, other: "Vertex") -> float:
        """
        Compute the distance to another vertex.
        """
        return np.linalg.norm(self.position - other.position)

@dataclass
class Edge:
    index: int
    tail_index: int
    head_index: int
    refine: bool = True
    fixed: bool = False
    options: Dict[str, Any] = field(default_factory=dict)

    def reversed(self) -> "Edge":
        return Edge(
            index=self.index,  # convention: reversed edge gets negative index
            tail_index=self.head_index,
            head_index=self.tail_index,
            refine=self.refine,
            fixed=self.fixed,
            options=self.options
        )

    def compute_length(self, mesh):
        tail = mesh.vertices[self.tail_index]
        head = mesh.vertices[self.head_index]
        return np.linalg.norm(head.position - tail.position)

@dataclass
class Facet:
    index: int
    edge_indices: List[int]  # Signed indices: +n = forward, -n = reversed (including -1 for "r0")
    options: Dict[str, Any] = field(default_factory=dict)

    def compute_normal(self, mesh) -> np.ndarray:
        """Compute (non-normalized) normal vector using right-hand rule from first three vertices."""
        if len(self.edge_indices) < 3:
            raise ValueError("Cannot compute normal with fewer than 3 edges.")

        verts = []
        for signed_index in self.edge_indices[:3]:
            edge = mesh.get_edge(signed_index)

            tail, head = edge.tail_index, edge.head_index
            if not verts:
                verts.append(tail)
            if head != verts[-1]:   # Prevent duplicates
                verts.append(head)

        # Only need first 3 vertices to define normal
        v0, v1, v2 = (mesh.vertices[i].position for i in verts[:3])
        u = v1 - v0
        v = v2 - v0

        return np.cross(u, v)

    def normal(self, mesh) -> np.ndarray:
        """Compute normalized normal vector"""
        n = self.compute_normal(mesh)
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError("Degenerate facet with zero normal.")
        return n / norm

    def compute_area(self, mesh: "Mesh") -> float:
        """
        Compute area by decomposing the polygon into triangles fan-based at vertex 0.
        Supports arbitrary n-gon.
        """
        verts = []
        for signed_index in self.edge_indices:
            edge = mesh.edges[abs(signed_index)]
            tail, head = (edge.tail_index, edge.head_index) if signed_index > 0 else (edge.head_index, edge.tail_index)
            if not verts:
                verts.append(tail)
            verts.append(head)

        verts = [mesh.vertices[i].position for i in verts[:-1]] # remove duplicate closing vertex
        v0, v1, v2 = verts
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        return area

    def compute_area_gradient(self, mesh: "Mesh") -> Dict[int, np.ndarray]:
        """
        Compute area gradient with respect to each vertex in the facet.
        Returns a dictionary where keys are vertex indices and values are gradient vectors.
        """
        # ordered vertex loop
        v_ids = []
        for signed_ei in self.edge_indices:
            edge = mesh.get_edge(signed_ei)
            tail = edge.tail_index
            if not v_ids or v_ids[-1] != tail:
                v_ids.append(tail)

        # triangulate facet around v0
        grad = {i: np.zeros(3) for i in v_ids}
        v0 = mesh.vertices[v_ids[0]].position

        for i in range(1, len(v_ids) - 1):
            a, b = v_ids[i], v_ids[i + 1]
            va, vb = mesh.vertices[a].position, mesh.vertices[b].position
            n = np.cross(va - v0, vb - v0)
            A = np.linalg.norm(n)
            if A < 1e-12:          # skip nearly-degenerate triangle
                continue
            n_hat = n / A          # unit normal
            grad[v_ids[0]] += 0.5 * np.cross(va - v0, n_hat)
            grad[a]                += 0.5 * np.cross(vb - va, n_hat)
            grad[b]                += 0.5 * np.cross(v0 - vb, n_hat)

        return grad

@dataclass
class Body:
    index: int
    facet_indices: List[int]
    target_volume: Optional[float] = 0.0
    options: Dict[str, Any] = field(default_factory=dict)

    def compute_volume(self, mesh) -> float:
        volume = 0.0
        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]
            # collect the true cyclic list of vertex‐indices around this facet:
            v_ids = []
            for signed_ei in facet.edge_indices:
                edge = mesh.edges[abs(signed_ei)]
                tail = edge.tail_index if signed_ei > 0 else edge.head_index
                if not v_ids or v_ids[-1] != tail:
                    v_ids.append(tail)
            # now v_ids is the ordered boundary of your facet (length >= 3)
            v_pos = [mesh.vertices[i].position for i in v_ids]
            v0 = v_pos[0]
            # triangulate into (v0, v_i, v_{i+1}) for i=1..len−2
            for i in range(1, len(v_pos)-1):
                v1 = v_pos[i]
                v2 = v_pos[i+1]
                volume += np.dot(v0, np.cross(v1, v2)) / 6.0
        return volume

    def compute_volume_gradient(self, mesh: "Mesh") -> Dict[int, np.ndarray]:
        """
        Compute the gradient of the volume with respect to each vertex in the body.
        Returns a dictionary mapping vertex indices to gradient vectors (np.ndarray).
        This version subtracts the body’s centroid so that the tetrahedron formula is applied
        relative to the body’s center rather than the origin.
        """
        # entities.py  – Body.compute_volume_gradient
        grad = {i: np.zeros(3) for i in mesh.vertices}

        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]

            # Re-create the ordered vertex loop exactly as you do in compute_volume
            v_ids = []
            for signed_ei in facet.edge_indices:
                edge = mesh.edges[abs(signed_ei)]
                tail = edge.tail_index if signed_ei > 0 else edge.head_index
                if not v_ids or v_ids[-1] != tail:
                    v_ids.append(tail)

            v0 = mesh.vertices[v_ids[0]].position
            for i in range(1, len(v_ids)-1):          # triangulate (v0,vi,vi+1)
                a, b = v_ids[i], v_ids[i+1]
                va, vb = mesh.vertices[a].position, mesh.vertices[b].position

                # ∂V/∂v equals cross-product of the opposite edge, divided by 6
                grad[v_ids[0]] += np.cross(va, vb) / 6
                grad[a]        += np.cross(vb, v0) / 6
                grad[b]        += np.cross(v0, va) / 6
        return grad

    def compute_surface_area(self, mesh) -> float:
        area = 0.0
        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]
            v_ids = []
            for signed_ei in facet.edge_indices:
                edge = mesh.edges[abs(signed_ei)]
                tail = edge.tail_index if signed_ei > 0 else edge.head_index
                if not v_ids or v_ids[-1] != tail:
                    v_ids.append(tail)
            v_pos = [mesh.vertices[i].position for i in v_ids]
            v0 = v_pos[0]
            for i in range(1, len(v_pos)-1):
                v1 = v_pos[i]
                v2 = v_pos[i+1]
                area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        return area

@dataclass
class Mesh:
    vertices: Dict[int, "Vertex"] = field(default_factory=dict)
    edges: Dict[int, "Edge"] = field(default_factory=dict)
    facets: Dict[int, "Facet"] = field(default_factory=dict)
    bodies: Dict[int, "Body"] = field(default_factory=dict)
    global_parameters: "GlobalParameters" = None  # Use the class here
    energy_modules: List[str] = field(default_factory=list)
    constraint_modules: List[str] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)

    def get_edge(self, index: int) -> "Edge":
        if index > 0:
            return self.edges[index]
        elif index < 0:
            return self.edges[-index].reversed()
        else:
            raise ValueError(f"Edge index {index} cannot be zero.")

    def compute_total_surface_area(self) -> float:
        return sum(facet.compute_area(self) for facet in self.facets.values())

    def compute_total_volume(self) -> float:
        return sum(body.compute_volume(self) for body in self.bodies.values())

    def validate_triangles(self):
        """Validate that all facets are triangles (have exactly 3 oriented edges).
        should only be called after initial triangulation."""
        for facet_idx in self.facets.keys():
            if len(self.facets[facet_idx].edge_indices) != 3:
                raise ValueError(f"Facet {facet_idx} does not have 3 edges. Found {len(self.facets[facet_idx].edge_indices)}.")
        return True

    def validate_edge_indices(self):
        for facet_idx in self.facets.keys():
            for signed_index in self.facets[facet_idx].edge_indices:
                edge_index = abs(signed_index)
                if not (1 <= edge_index <= len(self.edges)):
                    raise ValueError(f"Facet {facet_idx} uses invalid edge index {signed_index}.")
        return True

    def full_mesh_validate(self):
        """Perform full mesh validation."""
        # 1. Check all facets are triangles
        self.validate_triangles()

        # 2. Check all edge indices are valid
        self.validate_edge_indices()

        # 3. (optional future checks: vertex connectivity, bodies, etc.)
        # Pass for now
        return True
    
    def __post_init__(self):
        if self.global_parameters is None:
            self.global_parameters = GlobalParameters()

    def __str__(self):
        return f"Mesh with {len(self.vertices)} vertices, {len(self.edges)} edges, {len(self.facets)} facets, and {len(self.bodies)} bodies."
    def __repr__(self):
        return f"Mesh(vertices={self.vertices}, edges={self.edges}, facets={self.facets}, bodies={self.bodies})"
    def __len__(self):
        return len(self.vertices) + len(self.edges) + len(self.facets) + len(self.bodies)
    def __getitem__(self, index):
        if index in self.vertices:
            return self.vertices[index]
        elif index in self.edges:
            return self.edges[index]
        elif index in self.facets:
            return self.facets[index]
        elif index in self.bodies:
            return self.bodies[index]
        else:
            raise KeyError(f"Index {index} not found in mesh.")
    def __setitem__(self, index, value):
        if isinstance(value, Vertex):
            self.vertices[index] = value
        elif isinstance(value, Edge):
            self.edges[index] = value
        elif isinstance(value, Facet):
            self.facets[index] = value
        elif isinstance(value, Body):
            self.bodies[index] = value
        else:
            raise TypeError(f"Value {value} is not a valid mesh entity.")
    def __delitem__(self, index):
        if index in self.vertices:
            del self.vertices[index]
        elif index in self.edges:
            del self.edges[index]
        elif index in self.facets:
            del self.facets[index]
        elif index in self.bodies:
            del self.bodies[index]
        else:
            raise KeyError(f"Index {index} not found in mesh.")
    def __contains__(self, index):
        return index in self.vertices or index in self.edges or index in self.facets or index in self.bodies
