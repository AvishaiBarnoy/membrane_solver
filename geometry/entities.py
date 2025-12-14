# entities.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from parameters.global_parameters import GlobalParameters
import numpy as np
import sys
import logging

logger = logging.getLogger("membrane_solver")


class MeshError(Exception):
    """Custom exception for invalid mesh topology or geometry."""

@dataclass
class Vertex:
    index: int
    position: np.ndarray
    fixed: bool = False
    options: Dict[str, Any] = field(default_factory=dict)

    def copy(self):
        return Vertex(self.index, self.position.copy(), self.fixed,
                      self.options.copy())

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

    def copy(self):
        return Edge(self.index, self.tail_index, self.head_index, self.refine,
                    self.fixed, self.options.copy())

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
    refine: bool = True
    fixed: bool = False
    surface_tension: float = 1.0
    options: Dict[str, Any] = field(default_factory=dict)

    def copy(self):
        return Facet(index=self.index,
                     edge_indices=self.edge_indices[:],
                     refine=self.refine,
                     fixed=self.fixed,
                     surface_tension=self.surface_tension,
                     options=self.options.copy())

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
            tail, head = (
                edge.tail_index,
                edge.head_index,
            ) if signed_index > 0 else (
                edge.head_index,
                edge.tail_index,
            )
            if not verts:
                verts.append(tail)
            verts.append(head)

        v_pos = np.array([mesh.vertices[i].position for i in verts[:-1]])
        v0 = v_pos[0]
        v1 = v_pos[1:-1] - v0
        v2 = v_pos[2:] - v0
        cross = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(cross, axis=1).sum()
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

        if len(v_ids) < 3:
            return grad

        v_pos = np.array([mesh.vertices[i].position for i in v_ids])
        va = v_pos[1:-1] - v0
        vb = v_pos[2:] - v0

        n = np.cross(va, vb)
        A = np.linalg.norm(n, axis=1)
        mask = A >= 1e-12
        if not np.any(mask):
            return grad
        n_hat = n[mask] / A[mask][:, None]

        grad[v_ids[0]] += 0.5 * np.cross(va[mask], n_hat).sum(axis=0)

        cross_vb_va = np.cross(vb[mask] - va[mask], n_hat)
        cross_v0_vb = np.cross(-vb[mask], n_hat)  # v0 - vb since va,vb diff from v0
        for idx, (a, b) in enumerate(zip(np.array(v_ids[1:-1])[mask], np.array(v_ids[2:])[mask])):
            grad[a] += 0.5 * cross_vb_va[idx]
            grad[b] += 0.5 * cross_v0_vb[idx]

        return grad

    def compute_area_and_gradient(
        self,
        mesh: "Mesh",
        positions: "np.ndarray | None" = None,
        index_map: Dict[int, int] | None = None,
    ) -> tuple[float, Dict[int, np.ndarray]]:
        """
        Compute both the facet area and its gradient with respect to vertex positions.

        This combines the work of ``compute_area`` and ``compute_area_gradient`` so
        callers that need both can avoid redundant geometric computation.
        """
        # ordered vertex loop
        if getattr(mesh, "facet_vertex_loops", None) and self.index in mesh.facet_vertex_loops:
            v_ids_array = mesh.facet_vertex_loops[self.index]
            v_ids = v_ids_array.tolist()
        else:
            v_ids = []
            for signed_ei in self.edge_indices:
                edge = mesh.get_edge(signed_ei)
                tail = edge.tail_index
                if not v_ids or v_ids[-1] != tail:
                    v_ids.append(tail)

        grad: Dict[int, np.ndarray] = {i: np.zeros(3) for i in v_ids}
        if len(v_ids) < 3:
            return 0.0, grad

        if positions is not None and index_map is not None:
            rows = [index_map[i] for i in v_ids]
            v_pos = positions[rows]
        else:
            v_pos = np.array([mesh.vertices[i].position for i in v_ids])
        v0 = v_pos[0]
        va = v_pos[1:-1] - v0
        vb = v_pos[2:] - v0

        n = np.cross(va, vb)
        A = np.linalg.norm(n, axis=1)
        mask = A >= 1e-12
        if not np.any(mask):
            return 0.0, grad

        n_hat = n[mask] / A[mask][:, None]
        # total area is sum of triangle areas 0.5 * |n|
        area = float(0.5 * A[mask].sum())

        grad[v_ids[0]] += 0.5 * np.cross(va[mask], n_hat).sum(axis=0)

        cross_vb_va = np.cross(vb[mask] - va[mask], n_hat)
        cross_v0_vb = np.cross(-vb[mask], n_hat)
        inner_ids = np.array(v_ids[1:-1])
        next_ids = np.array(v_ids[2:])
        for idx, (a, b) in enumerate(zip(inner_ids[mask], next_ids[mask])):
            grad[a] += 0.5 * cross_vb_va[idx]
            grad[b] += 0.5 * cross_v0_vb[idx]

        return area, grad

@dataclass
class Body:
    index: int
    facet_indices: List[int]
    target_volume: Optional[float] = 0.0
    options: Dict[str, Any] = field(default_factory=dict)

    def copy(self):
        return Body(self.index, self.facet_indices[:],
                    self.target_volume, self.options.copy())

    def compute_volume(self, mesh) -> float:
        volume = 0.0
        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]
            # Reuse precomputed vertex loops when available to avoid
            # reconstructing the same connectivity on every call.
            if getattr(mesh, "facet_vertex_loops", None) and facet_idx in mesh.facet_vertex_loops:
                v_ids_array = mesh.facet_vertex_loops[facet_idx]
                v_ids = v_ids_array.tolist()
            else:
                # Fallback: collect the cyclic list of vertex indices.
                v_ids = []
                for signed_ei in facet.edge_indices:
                    edge = mesh.edges[abs(signed_ei)]
                    tail = edge.tail_index if signed_ei > 0 else edge.head_index
                    if not v_ids or v_ids[-1] != tail:
                        v_ids.append(tail)

            # ordered vertex positions
            v_pos = np.array([mesh.vertices[i].position for i in v_ids])
            v0 = v_pos[0]

            # triangulate into (v0, v_i, v_{i+1}) for i=1..len-2 using vectorized operations
            v1 = v_pos[1:-1]
            v2 = v_pos[2:]
            cross_prod = np.cross(v1, v2)
            volume += np.dot(cross_prod, v0).sum() / 6.0
        return volume

    def compute_volume_gradient(self, mesh: "Mesh") -> Dict[int, np.ndarray]:
        """
        Compute the gradient of the volume with respect to each vertex in the body.
        Returns a dictionary mapping vertex indices to gradient vectors (np.ndarray).
        This version subtracts the body’s centroid so that the tetrahedron formula is applied
        relative to the body’s center rather than the origin.
        """
        # entities.py  – Body.compute_volume_gradient
        vertex_indices: set[int] = set()
        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]
            for signed_ei in facet.edge_indices:
                edge = mesh.edges[abs(signed_ei)]
                tail = edge.tail_index if signed_ei > 0 else edge.head_index
                vertex_indices.add(tail)

        grad = {i: np.zeros(3) for i in vertex_indices}

        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]

            # Reuse the same ordered vertex loop as in compute_volume when
            # cached, otherwise reconstruct it from edges.
            if getattr(mesh, "facet_vertex_loops", None) and facet_idx in mesh.facet_vertex_loops:
                v_ids_array = mesh.facet_vertex_loops[facet_idx]
                v_ids = v_ids_array.tolist()
            else:
                v_ids = []
                for signed_ei in facet.edge_indices:
                    edge = mesh.edges[abs(signed_ei)]
                    tail = edge.tail_index if signed_ei > 0 else edge.head_index
                    if not v_ids or v_ids[-1] != tail:
                        v_ids.append(tail)

            if len(v_ids) < 3:
                continue

            v_pos = np.array([mesh.vertices[i].position for i in v_ids])
            v0 = v_pos[0]
            va = v_pos[1:-1]
            vb = v_pos[2:]

            cross_va_vb = np.cross(va, vb)
            grad[v_ids[0]] += cross_va_vb.sum(axis=0) / 6

            cross_vb_v0 = np.cross(vb, v0)
            cross_v0_va = np.cross(v0, va)

            for idx, (a, b) in enumerate(zip(v_ids[1:-1], v_ids[2:])):
                grad[a] += cross_vb_v0[idx] / 6
                grad[b] += cross_v0_va[idx] / 6
        return grad

    def compute_surface_area(self, mesh) -> float:
        area = 0.0
        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]
            if getattr(mesh, "facet_vertex_loops", None) and facet_idx in mesh.facet_vertex_loops:
                v_ids_array = mesh.facet_vertex_loops[facet_idx]
                v_ids = v_ids_array.tolist()
            else:
                v_ids = []
                for signed_ei in facet.edge_indices:
                    edge = mesh.edges[abs(signed_ei)]
                    tail = edge.tail_index if signed_ei > 0 else edge.head_index
                    if not v_ids or v_ids[-1] != tail:
                        v_ids.append(tail)

            v_pos = np.array([mesh.vertices[i].position for i in v_ids])
            v0 = v_pos[0]
            v1 = v_pos[1:-1] - v0
            v2 = v_pos[2:] - v0
            cross = np.cross(v1, v2)
            area += 0.5 * np.linalg.norm(cross, axis=1).sum()
        return area

    def compute_volume_and_gradient(self, mesh: "Mesh") -> tuple[float, Dict[int, np.ndarray]]:
        """
        Compute both the body volume and its gradient with respect to vertex positions.

        This combines the work of ``compute_volume`` and ``compute_volume_gradient``
        so callers that need both can avoid redundant geometric computation.
        """
        volume = 0.0
        grad: Dict[int, np.ndarray] = {}

        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]

            # Reuse cached vertex loop if available, otherwise reconstruct.
            if getattr(mesh, "facet_vertex_loops", None) and facet_idx in mesh.facet_vertex_loops:
                v_ids_array = mesh.facet_vertex_loops[facet_idx]
                v_ids = v_ids_array.tolist()
            else:
                v_ids = []
                for signed_ei in facet.edge_indices:
                    edge = mesh.edges[abs(signed_ei)]
                    tail = edge.tail_index if signed_ei > 0 else edge.head_index
                    if not v_ids or v_ids[-1] != tail:
                        v_ids.append(tail)

            if len(v_ids) < 3:
                continue

            v_pos = np.array([mesh.vertices[i].position for i in v_ids])
            v0 = v_pos[0]
            va = v_pos[1:-1]
            vb = v_pos[2:]

            # Volume contribution using vectorized tetrahedron fan at v0
            cross_prod = np.cross(va, vb)
            volume += float(np.dot(cross_prod, v0).sum() / 6.0)

            # Gradient contributions (same formula as compute_volume_gradient)
            cross_va_vb = cross_prod
            if v_ids[0] not in grad:
                grad[v_ids[0]] = np.zeros(3)
            grad[v_ids[0]] += cross_va_vb.sum(axis=0) / 6.0

            cross_vb_v0 = np.cross(vb, v0)
            cross_v0_va = np.cross(v0, va)

            for idx, (a, b) in enumerate(zip(v_ids[1:-1], v_ids[2:])):
                if a not in grad:
                    grad[a] = np.zeros(3)
                if b not in grad:
                    grad[b] = np.zeros(3)
                grad[a] += cross_vb_v0[idx] / 6.0
                grad[b] += cross_v0_va[idx] / 6.0

        return volume, grad

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

    vertex_to_facets: Dict[int, set] = field(default_factory=dict)
    vertex_to_edges: Dict[int, set] = field(default_factory=dict)
    edge_to_facets: Dict[int, set] = field(default_factory=dict)

    # Cached array views and facet vertex loops for performance.
    vertex_ids: "np.ndarray | None" = None
    vertex_index_to_row: Dict[int, int] = field(default_factory=dict)
    facet_vertex_loops: Dict[int, "np.ndarray"] = field(default_factory=dict)

    def copy(self):
        import copy
        new_mesh = Mesh()
        new_mesh.vertices = {vid: v.copy() for vid, v in self.vertices.items()}
        new_mesh.edges = {eid: e.copy() for eid, e in self.edges.items()}
        new_mesh.facets = {fid: f.copy() for fid, f in self.facets.items()}
        if hasattr(self, 'bodies'):
            new_mesh.bodies = {bid: b.copy() for bid, b in self.bodies.items()}
        if hasattr(self, 'global_parameters'):
            new_mesh.global_parameters = copy.deepcopy(self.global_parameters)
        return new_mesh

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

    def build_connectivity_maps(self):
        self.vertex_to_facets.clear()
        self.vertex_to_edges.clear()
        self.edge_to_facets.clear()

        for edge in self.edges.values():
            for v in (edge.tail_index, edge.head_index):
                if v not in self.vertex_to_edges:
                    self.vertex_to_edges[v] = set()
                self.vertex_to_edges[v].add(edge.index)

        for facet in self.facets.values():
            v_ids = set()
            for signed_ei in facet.edge_indices:
                ei = abs(signed_ei)

                if ei not in self.edge_to_facets:
                    self.edge_to_facets[ei] = set()
                self.edge_to_facets[ei].add(facet.index)

                edge = self.get_edge(signed_ei)
                v_ids.add(edge.tail_index)
                v_ids.add(edge.head_index)

            for v in v_ids:
                if v not in self.vertex_to_facets:
                    self.vertex_to_facets[v] = set()
                self.vertex_to_facets[v].add(facet.index)

    def build_position_cache(self):
        """Build or refresh the cached vertex ID order and index map.

        This lets downstream geometry routines construct position arrays
        efficiently without repeatedly iterating over dictionaries.
        """
        import numpy as np

        if self.vertex_ids is None or len(self.vertex_ids) != len(self.vertices):
            ids = np.array(sorted(self.vertices.keys()), dtype=int)
            self.vertex_ids = ids
            self.vertex_index_to_row = {vid: i for i, vid in enumerate(ids)}

    def positions_view(self) -> "np.ndarray":
        """Return a dense ``(N_vertices, 3)`` array of vertex positions.

        The ordering of rows is given by ``vertex_ids``; this is intended to
        be built once per geometry evaluation and reused across facets/bodies.
        """
        import numpy as np

        self.build_position_cache()
        return np.array([self.vertices[vid].position for vid in self.vertex_ids])

    def build_facet_vertex_loops(self):
        """Precompute ordered vertex loops for all facets.

        This avoids reconstructing the same vertex ordering from edges in
        every area/volume/gradient evaluation during minimization.
        """
        import numpy as np

        self.facet_vertex_loops.clear()
        for fid, facet in self.facets.items():
            v_ids: list[int] = []
            for signed_ei in facet.edge_indices:
                edge = self.edges[abs(signed_ei)]
                tail = edge.tail_index if signed_ei > 0 else edge.head_index
                if not v_ids or v_ids[-1] != tail:
                    v_ids.append(tail)
            if v_ids:
                self.facet_vertex_loops[fid] = np.array(v_ids, dtype=int)
    def get_facets_of_vertex(self, v_id: int) -> List["Facet"]:
        return [self.facets[fid] for fid in self.vertex_to_facets.get(v_id, [])]
    def get_edges_of_vertex(self, v_id: int) -> List["Edge"]:
        return [self.edges[eid] for eid in self.vertex_to_edges.get(v_id, [])]
    def get_facets_of_edge(self, e_id: int) -> List["Facet"]:
        return [self.facets[fid] for fid in self.edge_to_facets.get(e_id, [])]

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
