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
    options: Dict[str, Any] = field(default_factory=dict)

    def project_position(self, pos: np.ndarray) -> np.ndarray:
        ...

    def project_gradient(self, grad: np.ndarray) -> np.ndarray:
        ...

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

@dataclass
class Body:
    index: int
    facet_indices: List[int]
    target_volume: Optional[float] = None
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
        return abs(volume)

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
    #global_parameters: Dict[str, Any] = field(default_factory=dict)
    global_parameters: "GlobalParameters" = None  # Use the class here
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

