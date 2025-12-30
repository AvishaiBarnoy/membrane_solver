# entities.py

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from exceptions import InvalidEdgeIndexError
from parameters.global_parameters import GlobalParameters

logger = logging.getLogger("membrane_solver")


def _fast_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cross product of two arrays of 3D vectors along the last axis.
    Optimized to avoid np.cross overhead for small arrays or simple cases.
    Inputs must be shape (..., 3) or (3,).
    """
    x = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    y = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    z = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    out = np.empty(x.shape + (3,), dtype=x.dtype)
    out[..., 0] = x
    out[..., 1] = y
    out[..., 2] = z
    return out


class MeshError(Exception):
    """Custom exception for invalid mesh topology or geometry."""


@dataclass
class Vertex:
    index: int
    position: np.ndarray
    fixed: bool = False
    options: Dict[str, Any] = field(default_factory=dict)

    def copy(self):
        return Vertex(self.index, self.position.copy(), self.fixed, self.options.copy())

    def project_position(self, pos: np.ndarray) -> np.ndarray:
        """
        Project the given position onto the constraint, if any.
        If no constraint is defined, return the position unchanged.
        """
        if "constraint" in self.options:
            constraint = self.options["constraint"]
            return constraint.project_position(pos)
        return pos

    def project_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        Project the given gradient into the tangent space of the constraint, if any.
        If no constraint is defined, return the gradient unchanged.
        """
        if "constraint" in self.options:
            constraint = self.options["constraint"]
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
        return Edge(
            self.index,
            self.tail_index,
            self.head_index,
            self.refine,
            self.fixed,
            self.options.copy(),
        )

    def reversed(self) -> "Edge":
        return Edge(
            index=self.index,  # convention: reversed edge gets negative index
            tail_index=self.head_index,
            head_index=self.tail_index,
            refine=self.refine,
            fixed=self.fixed,
            options=self.options,
        )

    def compute_length(self, mesh):
        tail = mesh.vertices[self.tail_index]
        head = mesh.vertices[self.head_index]
        return np.linalg.norm(head.position - tail.position)


@dataclass
class Facet:
    index: int
    edge_indices: List[
        int
    ]  # Signed indices: +n = forward, -n = reversed (including -1 for "r0")
    refine: bool = True
    fixed: bool = False
    surface_tension: float = 1.0
    options: Dict[str, Any] = field(default_factory=dict)

    # TODO: Implement caching for area and gradient (similar to Body) to avoid
    # redundant calculations when multiple modules (energy, constraints) access
    # the same facet data within a single minimization step.

    def copy(self):
        return Facet(
            index=self.index,
            edge_indices=self.edge_indices[:],
            refine=self.refine,
            fixed=self.fixed,
            surface_tension=self.surface_tension,
            options=self.options.copy(),
        )

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
            if head != verts[-1]:  # Prevent duplicates
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
        Compute area using the standard shoelace formula (sum of cross products).
        This is robust for general polygons (convex or concave) in 3D,
        projected onto the plane defined by the facet normal.
        """
        verts = []
        for signed_index in self.edge_indices:
            edge = mesh.edges[abs(signed_index)]
            tail, head = (
                (
                    edge.tail_index,
                    edge.head_index,
                )
                if signed_index > 0
                else (
                    edge.head_index,
                    edge.tail_index,
                )
            )
            if not verts:
                verts.append(tail)
            verts.append(head)

        # Get unique ordered vertices for the loop
        v_ids = verts[:-1]
        v_pos = np.array([mesh.vertices[i].position for i in v_ids])

        if len(v_ids) < 3:
            return 0.0

        # Shoelace formula vector form: Area = 0.5 * |sum(v_i x v_{i+1})|
        # Note: cyclic cross products
        v_curr = v_pos
        v_next = np.roll(v_pos, -1, axis=0)
        cross_sum = _fast_cross(v_curr, v_next).sum(axis=0)
        return 0.5 * float(np.linalg.norm(cross_sum))

    def compute_area_gradient(self, mesh: "Mesh") -> Dict[int, np.ndarray]:
        """
        Compute area gradient with respect to each vertex using the shoelace formula.
        """
        # ordered vertex loop
        v_ids = []
        for signed_ei in self.edge_indices:
            edge = mesh.get_edge(signed_ei)
            tail = edge.tail_index
            if not v_ids or v_ids[-1] != tail:
                v_ids.append(tail)

        grad = {i: np.zeros(3) for i in v_ids}
        if len(v_ids) < 3:
            return grad

        v_pos = np.array([mesh.vertices[i].position for i in v_ids])

        # Area vector A_vec = 0.5 * sum(v_i x v_{i+1})
        # Area A = |A_vec|
        # Gradient of A w.r.t v_k is 0.5 * (v_{k-1} - v_{k+1}) x n_hat
        # where n_hat = A_vec / |A_vec|

        v_prev = np.roll(v_pos, 1, axis=0)
        v_next = np.roll(v_pos, -1, axis=0)

        # Calculate area vector first to get normal
        cross_sum = _fast_cross(v_pos, v_next).sum(axis=0)
        area_doubled = np.linalg.norm(cross_sum)

        if area_doubled < 1e-12:
            return grad

        n_hat = cross_sum / area_doubled

        # Term for each vertex v_i: 0.5 * n_hat x (v_{i-1} - v_{i+1})
        diff = v_prev - v_next
        grads = 0.5 * _fast_cross(n_hat, diff)

        for i, vid in enumerate(v_ids):
            grad[vid] += grads[i]

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
        Uses the Shoelace formula.
        """
        # ordered vertex loop
        if (
            getattr(mesh, "facet_vertex_loops", None)
            and self.index in mesh.facet_vertex_loops
        ):
            v_ids_array = mesh.facet_vertex_loops[self.index]
            v_ids = v_ids_array.tolist()
        else:
            v_ids = []
            for signed_ei in self.edge_indices:
                edge = mesh.edges[abs(signed_ei)]
                if signed_ei > 0:
                    tail, head = edge.tail_index, edge.head_index
                else:
                    tail, head = edge.head_index, edge.tail_index
                if not v_ids:
                    v_ids.append(tail)
                v_ids.append(head)
            if len(v_ids) > 1:
                v_ids = v_ids[:-1]

        grad: Dict[int, np.ndarray] = {i: np.zeros(3) for i in v_ids}
        if len(v_ids) < 3:
            return 0.0, grad

        if positions is not None and index_map is not None:
            rows = [index_map[i] for i in v_ids]
            v_pos = positions[rows]
        else:
            v_pos = np.array([mesh.vertices[i].position for i in v_ids])

        # Shoelace area vector: 0.5 * sum(v_i x v_{i+1})
        v_curr = v_pos
        v_next = np.roll(v_pos, -1, axis=0)

        cross_sum = _fast_cross(v_curr, v_next).sum(axis=0)
        area_doubled = float(np.linalg.norm(cross_sum))
        area = 0.5 * area_doubled

        if area_doubled < 1e-12:
            return 0.0, grad

        n_hat = cross_sum / area_doubled

        # Gradient term for each vertex v_i: 0.5 * n_hat x (v_{i-1} - v_{i+1})
        v_prev = np.roll(v_pos, 1, axis=0)
        diff = v_prev - v_next
        grads = 0.5 * _fast_cross(n_hat, diff)

        for i, vid in enumerate(v_ids):
            grad[vid] += grads[i]

        return area, grad


@dataclass
class Body:
    index: int
    facet_indices: List[int]
    target_volume: Optional[float] = 0.0
    options: Dict[str, Any] = field(default_factory=dict)

    _cached_volume: Optional[float] = field(init=False, default=None)
    _cached_volume_grad: Optional[Dict[int, np.ndarray]] = field(
        init=False, default=None
    )
    _cached_version: int = field(init=False, default=-1)

    # Cache for vectorized operations
    _cached_body_rows: Optional[np.ndarray] = field(init=False, default=None)
    _cached_body_rows_version: int = field(init=False, default=-1)

    def copy(self):
        return Body(
            self.index, self.facet_indices[:], self.target_volume, self.options.copy()
        )

    def _get_triangle_rows(self, mesh: "Mesh") -> Optional[np.ndarray]:
        """
        Get the indices of rows in mesh.triangle_row_cache that belong to this body.
        Returns None if not all facets are triangles or if cache is invalid.
        """
        if (
            self._cached_body_rows is not None
            and self._cached_body_rows_version == mesh._facet_loops_version
        ):
            return self._cached_body_rows

        # Ensure mesh cache is built
        tri_rows, _ = mesh.triangle_row_cache()
        if tri_rows is None:
            return None

        # Map our facets to rows
        facet_map = mesh.facet_to_triangle_row
        rows = []
        for fid in self.facet_indices:
            row_idx = facet_map.get(fid)
            if row_idx is None:
                # This facet is not a triangle or not in cache
                return None
            rows.append(row_idx)

        self._cached_body_rows = np.array(rows, dtype=int)
        self._cached_body_rows_version = mesh._facet_loops_version
        return self._cached_body_rows

    def compute_volume(
        self,
        mesh: "Mesh",
        positions: "np.ndarray | None" = None,
        index_map: Dict[int, int] | None = None,
    ) -> float:
        if (
            positions is None
            and self._cached_version == mesh._version
            and self._cached_volume is not None
        ):
            return self._cached_volume

        volume = 0.0

        # Try fully vectorized path first (fastest)
        # We need positions array and valid triangle rows for this body
        if positions is None and getattr(mesh, "facet_vertex_loops", None):
            positions = mesh.positions_view()

        if positions is not None:
            body_rows = self._get_triangle_rows(mesh)
            tri_rows, _ = mesh.triangle_row_cache()

            if body_rows is not None and tri_rows is not None:
                # Get triangle vertex indices: (N_tri, 3)
                # These are indices into the positions array (rows)
                # tri_rows is (Total_Tri, 3), body_rows is (N_body_tri,)
                # So we want tri_rows[body_rows] -> (N_body_tri, 3)
                indices = tri_rows[body_rows]

                v0 = positions[indices[:, 0]]
                v1 = positions[indices[:, 1]]
                v2 = positions[indices[:, 2]]

                cross = _fast_cross(v1, v2)
                vol_contrib = np.einsum("ij,ij->i", cross, v0)
                volume = float(vol_contrib.sum() / 6.0)

                self._cached_volume = volume
                self._cached_version = mesh._version
                return volume

        # Fallback path (polygonal or no batched inputs)
        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]
            if (
                getattr(mesh, "facet_vertex_loops", None)
                and facet_idx in mesh.facet_vertex_loops
            ):
                v_ids_array = mesh.facet_vertex_loops[facet_idx]
                v_ids = v_ids_array.tolist()
            else:
                v_ids = []
                for signed_ei in facet.edge_indices:
                    edge = mesh.edges[abs(signed_ei)]
                    if signed_ei > 0:
                        tail, head = edge.tail_index, edge.head_index
                    else:
                        tail, head = edge.head_index, edge.tail_index
                    if not v_ids:
                        v_ids.append(tail)
                    v_ids.append(head)
                if len(v_ids) > 1:
                    v_ids = v_ids[:-1]

            v_pos = np.array([mesh.vertices[i].position for i in v_ids])
            v0 = v_pos[0]
            v1 = v_pos[1:-1]
            v2 = v_pos[2:]
            cross_prod = _fast_cross(v1, v2)
            volume += np.dot(cross_prod, v0).sum() / 6.0

        self._cached_volume = volume
        self._cached_version = mesh._version
        return volume

    def accumulate_volume_gradient(
        self, mesh: "Mesh", positions: np.ndarray, grad_arr: np.ndarray, factor: float
    ):
        """
        Compute volume gradient and add (factor * grad) directly to grad_arr.
        Assumes positions and grad_arr are aligned with mesh.vertex_ids.
        """
        body_rows = self._get_triangle_rows(mesh)
        tri_rows, _ = mesh.triangle_row_cache()

        if body_rows is not None and tri_rows is not None:
            # Vectorized path
            indices = tri_rows[body_rows]  # (N_tri, 3)

            v0 = positions[indices[:, 0]]
            v1 = positions[indices[:, 1]]
            v2 = positions[indices[:, 2]]

            # Gradients for each vertex of the triangle
            # grad_v0 = (v1 x v2) / 6
            # grad_v1 = (v2 x v0) / 6
            # grad_v2 = (v0 x v1) / 6
            g0 = _fast_cross(v1, v2) * (factor / 6.0)
            g1 = _fast_cross(v2, v0) * (factor / 6.0)
            g2 = _fast_cross(v0, v1) * (factor / 6.0)

            # Scatter add
            np.add.at(grad_arr, indices[:, 0], g0)
            np.add.at(grad_arr, indices[:, 1], g1)
            np.add.at(grad_arr, indices[:, 2], g2)
        else:
            # Fallback path: Compute dictionary and scatter
            _, grad_dict = self.compute_volume_and_gradient(mesh)

            # We need to map vertex IDs to rows in grad_arr
            # mesh.vertex_index_to_row should be valid if positions is valid
            idx_map = mesh.vertex_index_to_row
            for vid, g in grad_dict.items():
                row = idx_map.get(vid)
                if row is not None:
                    grad_arr[row] += factor * g

    def compute_volume_and_accumulate_gradient(
        self,
        mesh: "Mesh",
        positions: np.ndarray,
        grad_arr: np.ndarray,
        stiffness_k: float,
        target_volume: float,
    ) -> tuple[float, float]:
        """
        Combined operation to compute volume, energy, and accumulate gradients.
        Avoids redundant memory gathers by doing everything in one pass.
        Returns (volume, energy).
        """
        body_rows = self._get_triangle_rows(mesh)
        tri_rows, _ = mesh.triangle_row_cache()

        if body_rows is not None and tri_rows is not None:
            # Vectorized path
            indices = tri_rows[body_rows]  # (N_tri, 3)

            # GATHER ONCE
            v0 = positions[indices[:, 0]]
            v1 = positions[indices[:, 1]]
            v2 = positions[indices[:, 2]]

            cross_v1_v2 = _fast_cross(v1, v2)

            # Volume
            vol_contrib = np.einsum("ij,ij->i", cross_v1_v2, v0)
            volume = float(vol_contrib.sum() / 6.0)

            # Energy
            delta = volume - target_volume
            energy = 0.5 * stiffness_k * delta**2

            # Gradient
            factor = stiffness_k * delta

            # Reuse cross_v1_v2 for g0
            g0 = cross_v1_v2 * (factor / 6.0)
            g1 = _fast_cross(v2, v0) * (factor / 6.0)
            g2 = _fast_cross(v0, v1) * (factor / 6.0)

            # Scatter add
            np.add.at(grad_arr, indices[:, 0], g0)
            np.add.at(grad_arr, indices[:, 1], g1)
            np.add.at(grad_arr, indices[:, 2], g2)

            # Cache the volume result
            self._cached_volume = volume
            self._cached_version = mesh._version

            return volume, energy
        else:
            # Fallback
            vol = self.compute_volume(mesh, positions)
            delta = vol - target_volume
            energy = 0.5 * stiffness_k * delta**2
            factor = stiffness_k * delta
            self.accumulate_volume_gradient(mesh, positions, grad_arr, factor)
            return vol, energy

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
            if (
                getattr(mesh, "facet_vertex_loops", None)
                and facet_idx in mesh.facet_vertex_loops
            ):
                v_ids_array = mesh.facet_vertex_loops[facet_idx]
                v_ids = v_ids_array.tolist()
            else:
                v_ids = []
                for signed_ei in facet.edge_indices:
                    edge = mesh.edges[abs(signed_ei)]
                    tail = edge.tail_index if signed_ei > 0 else edge.head_index
                    if not v_ids or v_ids[-1] != tail:
                        v_ids.append(tail)
                if len(v_ids) > 1:
                    v_ids = v_ids[:-1]

            if len(v_ids) < 3:
                continue

            v_pos = np.array([mesh.vertices[i].position for i in v_ids])
            v0 = v_pos[0]
            va = v_pos[1:-1]
            vb = v_pos[2:]

            cross_va_vb = _fast_cross(va, vb)
            grad[v_ids[0]] += cross_va_vb.sum(axis=0) / 6

            cross_vb_v0 = _fast_cross(vb, v0)
            cross_v0_va = _fast_cross(v0, va)

            for idx, (a, b) in enumerate(zip(v_ids[1:-1], v_ids[2:])):
                grad[a] += cross_vb_v0[idx] / 6
                grad[b] += cross_v0_va[idx] / 6
        return grad

    def compute_surface_area(self, mesh) -> float:
        area = 0.0
        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]
            if (
                getattr(mesh, "facet_vertex_loops", None)
                and facet_idx in mesh.facet_vertex_loops
            ):
                v_ids_array = mesh.facet_vertex_loops[facet_idx]
                v_ids = v_ids_array.tolist()
            else:
                v_ids = []
                for signed_ei in facet.edge_indices:
                    edge = mesh.edges[abs(signed_ei)]
                    if signed_ei > 0:
                        tail, head = edge.tail_index, edge.head_index
                    else:
                        tail, head = edge.head_index, edge.tail_index
                    if not v_ids:
                        v_ids.append(tail)
                    v_ids.append(head)
                if len(v_ids) > 1:
                    v_ids = v_ids[:-1]

            v_pos = np.array([mesh.vertices[i].position for i in v_ids])
            v0 = v_pos[0]
            v1 = v_pos[1:-1] - v0
            v2 = v_pos[2:] - v0
            cross = _fast_cross(v1, v2)
            area += 0.5 * np.linalg.norm(cross, axis=1).sum()
        return area

    def compute_volume_and_gradient(
        self,
        mesh: "Mesh",
        positions: "np.ndarray | None" = None,
        index_map: Dict[int, int] | None = None,
    ) -> tuple[float, Dict[int, np.ndarray]]:
        """
        Compute both the body volume and its gradient with respect to vertex positions.

        This combines the work of ``compute_volume`` and ``compute_volume_gradient``
        so callers that need both can avoid redundant geometric computation.
        """
        if (
            positions is None
            and self._cached_version == mesh._version
            and self._cached_volume is not None
            and self._cached_volume_grad is not None
        ):
            return self._cached_volume, self._cached_volume_grad

        volume = 0.0
        grad: Dict[int, np.ndarray] = {}

        # Batched path
        if positions is not None and index_map is not None:
            n_facets = len(self.facet_indices)
            tri_indices = np.empty((n_facets, 3), dtype=int)
            valid_count = 0
            all_triangles = True

            for facet_idx in self.facet_indices:
                loop = mesh.facet_vertex_loops.get(facet_idx)
                if loop is None or len(loop) != 3:
                    all_triangles = False
                    break
                tri_indices[valid_count, 0] = index_map[int(loop[0])]
                tri_indices[valid_count, 1] = index_map[int(loop[1])]
                tri_indices[valid_count, 2] = index_map[int(loop[2])]
                valid_count += 1

            if all_triangles:
                v0 = positions[tri_indices[:valid_count, 0]]
                v1 = positions[tri_indices[:valid_count, 1]]
                v2 = positions[tri_indices[:valid_count, 2]]

                # Volume
                cross_v1_v2 = _fast_cross(v1, v2)
                vol_contrib = np.einsum("ij,ij->i", cross_v1_v2, v0)
                volume = float(vol_contrib.sum() / 6.0)

                # Gradient
                g0 = cross_v1_v2 / 6.0
                g1 = _fast_cross(v2, v0) / 6.0
                g2 = _fast_cross(v0, v1) / 6.0

                n_vertices = len(mesh.vertex_ids)
                grad_arr = np.zeros((n_vertices, 3), dtype=float)

                i0 = tri_indices[:valid_count, 0]
                i1 = tri_indices[:valid_count, 1]
                i2 = tri_indices[:valid_count, 2]

                np.add.at(grad_arr, i0, g0)
                np.add.at(grad_arr, i1, g1)
                np.add.at(grad_arr, i2, g2)

                for vid, row in index_map.items():
                    if row < len(grad_arr):
                        g = grad_arr[row]
                        if np.any(g):
                            grad[vid] = g

                self._cached_volume = volume
                self._cached_volume_grad = grad
                self._cached_version = mesh._version
                return volume, grad

        # Fallback path
        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]

            # Reuse cached vertex loop if available, otherwise reconstruct.
            if (
                getattr(mesh, "facet_vertex_loops", None)
                and facet_idx in mesh.facet_vertex_loops
            ):
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
            cross_prod = _fast_cross(va, vb)
            volume += float(np.dot(cross_prod, v0).sum() / 6.0)

            # Gradient contributions (same formula as compute_volume_gradient)
            cross_va_vb = cross_prod
            if v_ids[0] not in grad:
                grad[v_ids[0]] = np.zeros(3)
            grad[v_ids[0]] += cross_va_vb.sum(axis=0) / 6.0

            cross_vb_v0 = _fast_cross(vb, v0)
            cross_v0_va = _fast_cross(v0, va)

            for idx, (a, b) in enumerate(zip(v_ids[1:-1], v_ids[2:])):
                if a not in grad:
                    grad[a] = np.zeros(3)
                if b not in grad:
                    grad[b] = np.zeros(3)
                grad[a] += cross_vb_v0[idx] / 6.0
                grad[b] += cross_v0_va[idx] / 6.0

        self._cached_volume = volume
        self._cached_volume_grad = grad
        self._cached_version = mesh._version

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
    macros: Dict[str, List[str]] = field(default_factory=dict)

    vertex_to_facets: Dict[int, set] = field(default_factory=dict)
    vertex_to_edges: Dict[int, set] = field(default_factory=dict)
    edge_to_facets: Dict[int, set] = field(default_factory=dict)

    # Cached array views and facet vertex loops for performance.
    vertex_ids: "np.ndarray | None" = None
    vertex_index_to_row: Dict[int, int] = field(default_factory=dict)
    facet_vertex_loops: Dict[int, "np.ndarray"] = field(default_factory=dict)

    _positions_cache: "np.ndarray | None" = None
    _positions_cache_version: int = -1
    _triangle_rows_cache: "np.ndarray | None" = None
    _triangle_rows_cache_version: int = -1
    _triangle_row_facets: list[int] = field(default_factory=list)
    _facet_loops_version: int = 0
    _facet_to_row_cache: Dict[int, int] = field(default_factory=dict)
    _facet_to_row_version: int = -1

    # Topology-only versioning for caches that depend on edges/facets adjacency.
    # This should only be incremented when discrete mesh operations change
    # connectivity (refinement, equiangulation, edge flips), not when positions move.
    _topology_version: int = 0
    _connectivity_cache_version: int = -1
    _connectivity_cache_counts: tuple[int, int, int] = (0, 0, 0)
    _boundary_vertex_cache_version: int = -1
    _boundary_vertex_cache: set[int] = field(default_factory=set)

    _fixed_mask_cache: "np.ndarray | None" = None
    _fixed_mask_version: int = -1

    _parameter_array_cache: Dict[str, "np.ndarray"] = field(default_factory=dict)
    _parameter_cache_version: int = -1

    _version: int = 0
    _vertex_ids_version: int = 0

    def increment_version(self):
        self._version += 1

    def increment_topology_version(self) -> None:
        """Invalidate caches that depend on mesh connectivity."""
        self._topology_version += 1
        self._connectivity_cache_version = -1
        self._boundary_vertex_cache_version = -1

    @property
    def fixed_mask(self) -> "np.ndarray":
        """Return a boolean array of fixed status for vertices."""
        import numpy as np

        if (
            self._fixed_mask_cache is not None
            and self._fixed_mask_version == self._version
            and len(self._fixed_mask_cache) == len(self.vertices)
        ):
            return self._fixed_mask_cache

        self.build_position_cache()
        n_verts = len(self.vertex_ids)
        mask = np.zeros(n_verts, dtype=bool)

        # Iterate over vertices in order or use mapping?
        # vertex_ids is sorted.
        for i, vid in enumerate(self.vertex_ids):
            if self.vertices[vid].fixed:
                mask[i] = True

        self._fixed_mask_cache = mask
        self._fixed_mask_version = self._version
        return mask

    def get_facet_parameter_array(
        self, param_name: str, default_val: float | None = None
    ) -> "np.ndarray":
        """Return a cached array of parameter values for all cached triangle rows."""
        import numpy as np

        # Ensure we have the triangle rows built, as this array corresponds to THAT ordering
        rows, facets = self.triangle_row_cache()
        if rows is None:
            # Fallback/Empty
            return np.array([])

        cache_key = f"facet_{param_name}"
        if (
            self._parameter_cache_version == self._version
            and cache_key in self._parameter_array_cache
        ):
            return self._parameter_array_cache[cache_key]

        # Invalidate cache if version changed
        if self._parameter_cache_version != self._version:
            self._parameter_array_cache.clear()
            self._parameter_cache_version = self._version

        # Build array
        n_facets = len(facets)
        if default_val is None:
            default_val = self.global_parameters.get(param_name) or 0.0

        arr = np.full(n_facets, default_val, dtype=float)

        # We need to scan only facets that have an override
        # Optimization: Most facets won't have overrides.
        # But we don't know which ones.
        # We iterate over the cached facets (which are ids)
        for i, fid in enumerate(facets):
            facet = self.facets[fid]
            if param_name in facet.options:
                arr[i] = float(facet.options[param_name])

        self._parameter_array_cache[cache_key] = arr
        return arr

    @property
    def facet_to_triangle_row(self) -> Dict[int, int]:
        """
        Return a map from facet_index to row index in the triangle_rows_cache.
        """
        if (
            self._facet_to_row_version == self._facet_loops_version
            and self._facet_to_row_cache
        ):
            return self._facet_to_row_cache

        # Ensure cache is built
        self.triangle_row_cache()

        self._facet_to_row_cache = {
            fid: i for i, fid in enumerate(self._triangle_row_facets)
        }
        self._facet_to_row_version = self._facet_loops_version
        return self._facet_to_row_cache

    def copy(self):
        import copy

        new_mesh = Mesh()
        new_mesh.vertices = {vid: v.copy() for vid, v in self.vertices.items()}
        new_mesh.edges = {eid: e.copy() for eid, e in self.edges.items()}
        new_mesh.facets = {fid: f.copy() for fid, f in self.facets.items()}
        if hasattr(self, "bodies"):
            new_mesh.bodies = {bid: b.copy() for bid, b in self.bodies.items()}
        if hasattr(self, "global_parameters"):
            new_mesh.global_parameters = copy.deepcopy(self.global_parameters)
        new_mesh.macros = copy.deepcopy(getattr(self, "macros", {}))
        new_mesh.energy_modules = list(getattr(self, "energy_modules", []))
        new_mesh.constraint_modules = list(getattr(self, "constraint_modules", []))
        new_mesh.instructions = list(getattr(self, "instructions", []))
        new_mesh._topology_version = self._topology_version
        return new_mesh

    @property
    def boundary_vertex_ids(self) -> set[int]:
        """Return the set of vertex IDs that lie on the boundary of an open mesh."""
        if self._boundary_vertex_cache_version == self._topology_version:
            return set(self._boundary_vertex_cache)

        self.build_connectivity_maps()
        boundary_vids: set[int] = set()
        for eid, facet_set in self.edge_to_facets.items():
            if len(facet_set) < 2:
                edge = self.edges[eid]
                boundary_vids.add(edge.tail_index)
                boundary_vids.add(edge.head_index)
        self._boundary_vertex_cache = boundary_vids
        self._boundary_vertex_cache_version = self._topology_version
        return set(boundary_vids)

    def get_edge(self, index: int) -> "Edge":
        if index > 0:
            return self.edges[index]
        if index < 0:
            return self.edges[-index].reversed()
        raise InvalidEdgeIndexError(index)

    def compute_total_surface_area(self) -> float:
        return sum(facet.compute_area(self) for facet in self.facets.values())

    def compute_total_volume(self) -> float:
        return sum(body.compute_volume(self) for body in self.bodies.values())

    def compute_surface_radius_of_gyration(
        self, facet_indices: Iterable[int] | None = None
    ) -> float:
        """Return the surface-area-weighted radius of gyration for the mesh."""
        if facet_indices is None:
            facet_indices = self.facets.keys()

        total_area = 0.0
        centroid_sum = np.zeros(3, dtype=float)
        mean_r2_sum = 0.0

        for facet_idx in facet_indices:
            facet = self.facets.get(facet_idx)
            if facet is None:
                continue

            if (
                getattr(self, "facet_vertex_loops", None)
                and facet_idx in self.facet_vertex_loops
            ):
                v_ids_array = self.facet_vertex_loops[facet_idx]
                v_ids = v_ids_array.tolist()
            else:
                v_ids = []
                for signed_ei in facet.edge_indices:
                    edge = self.edges[abs(signed_ei)]
                    tail = edge.tail_index if signed_ei > 0 else edge.head_index
                    if not v_ids or v_ids[-1] != tail:
                        v_ids.append(tail)

            if len(v_ids) < 3:
                continue

            v_pos = np.array([self.vertices[i].position for i in v_ids], dtype=float)
            v0 = v_pos[0]
            for i in range(1, len(v_pos) - 1):
                a = v0
                b = v_pos[i]
                c = v_pos[i + 1]
                cross = np.cross(b - a, c - a)
                area = 0.5 * float(np.linalg.norm(cross))
                if area == 0.0:
                    continue
                centroid = (a + b + c) / 3.0
                mean_r2 = (
                    np.dot(a, a)
                    + np.dot(b, b)
                    + np.dot(c, c)
                    + np.dot(a, b)
                    + np.dot(b, c)
                    + np.dot(c, a)
                ) / 6.0

                total_area += area
                centroid_sum += area * centroid
                mean_r2_sum += area * mean_r2

        if total_area == 0.0:
            return 0.0

        centroid = centroid_sum / total_area
        mean_r2 = mean_r2_sum / total_area
        rg2 = float(mean_r2 - np.dot(centroid, centroid))
        if rg2 < 0.0 and rg2 > -1e-12:
            rg2 = 0.0
        return float(np.sqrt(max(rg2, 0.0)))

    def validate_triangles(self):
        """Validate that all facets are triangles (have exactly 3 oriented edges).
        should only be called after initial triangulation."""
        for facet_idx in self.facets.keys():
            if len(self.facets[facet_idx].edge_indices) != 3:
                raise ValueError(
                    f"Facet {facet_idx} does not have 3 edges. Found {len(self.facets[facet_idx].edge_indices)}."
                )
        return True

    def validate_edge_indices(self):
        for facet_idx in self.facets.keys():
            for signed_index in self.facets[facet_idx].edge_indices:
                edge_index = abs(signed_index)
                if edge_index not in self.edges:
                    raise ValueError(
                        f"Facet {facet_idx} uses invalid edge index {signed_index} (not found in edge list)."
                    )
        return True

    def build_connectivity_maps(self):
        counts = (len(self.vertices), len(self.edges), len(self.facets))
        if (
            self._connectivity_cache_version == self._topology_version
            and self._connectivity_cache_counts == counts
        ):
            return

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

        self._connectivity_cache_version = self._topology_version
        self._connectivity_cache_counts = counts

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
            self._vertex_ids_version += 1

    def positions_view(self) -> "np.ndarray":
        """Return a dense ``(N_vertices, 3)`` array of vertex positions.

        The ordering of rows is given by ``vertex_ids``; this is intended to
        be built once per geometry evaluation and reused across facets/bodies.
        """
        import numpy as np

        self.build_position_cache()
        if (
            self._positions_cache is None
            or self._positions_cache_version != self._version
            or len(self._positions_cache) != len(self.vertex_ids)
        ):
            n_verts = len(self.vertex_ids)
            if self._positions_cache is None or self._positions_cache.shape != (
                n_verts,
                3,
            ):
                self._positions_cache = np.empty((n_verts, 3), dtype=float, order="F")
            for i, vid in enumerate(self.vertex_ids):
                self._positions_cache[i] = self.vertices[vid].position
            self._positions_cache_version = self._version

        return self._positions_cache

    def build_facet_vertex_loops(self):
        """Precompute ordered vertex loops for all facets.

        This avoids reconstructing the same vertex ordering from edges in
        every area/volume/gradient evaluation during minimization.
        """
        import numpy as np

        if not hasattr(self, "facet_vertex_loops"):
            self.facet_vertex_loops = {}
        if not hasattr(self, "_facet_loops_version"):
            self._facet_loops_version = 0

        self.facet_vertex_loops.clear()
        for fid in sorted(self.facets):
            facet = self.facets[fid]
            v_ids: list[int] = []
            for signed_ei in facet.edge_indices:
                edge = self.edges[abs(signed_ei)]
                tail = edge.tail_index if signed_ei > 0 else edge.head_index
                if not v_ids or v_ids[-1] != tail:
                    v_ids.append(tail)
            if v_ids:
                self.facet_vertex_loops[fid] = np.array(v_ids, dtype=int)
        self._facet_loops_version += 1

    def triangle_row_cache(self) -> tuple["np.ndarray | None", list[int]]:
        """Return cached triangle row indices and facet IDs.

        The cache is rebuilt only when facet loops are rebuilt.
        """
        import numpy as np

        if (
            self._triangle_rows_cache is not None
            and self._triangle_rows_cache_version == self._facet_loops_version
        ):
            return self._triangle_rows_cache, self._triangle_row_facets

        if not getattr(self, "facet_vertex_loops", None):
            return None, []

        self.build_position_cache()

        tri_facets: list[int] = []
        for fid in sorted(self.facet_vertex_loops):
            loop = self.facet_vertex_loops[fid]
            if len(loop) == 3:
                tri_facets.append(fid)

        if not tri_facets:
            self._triangle_rows_cache = None
            self._triangle_row_facets = []
            self._triangle_rows_cache_version = self._facet_loops_version
            return None, []

        tri_rows = np.empty((len(tri_facets), 3), dtype=np.int32, order="F")
        for idx, fid in enumerate(tri_facets):
            loop = self.facet_vertex_loops[fid]
            tri_rows[idx, 0] = self.vertex_index_to_row[int(loop[0])]
            tri_rows[idx, 1] = self.vertex_index_to_row[int(loop[1])]
            tri_rows[idx, 2] = self.vertex_index_to_row[int(loop[2])]

        self._triangle_rows_cache = tri_rows
        self._triangle_row_facets = tri_facets
        self._triangle_rows_cache_version = self._facet_loops_version
        return tri_rows, tri_facets

    def get_facets_of_vertex(self, v_id: int) -> List["Facet"]:
        return [self.facets[fid] for fid in self.vertex_to_facets.get(v_id, [])]

    def get_facet_indices_of_vertex(self, v_id: int) -> set[int]:
        """Return the IDs of facets sharing this vertex."""
        return self.vertex_to_facets.get(v_id, set())

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
        return (
            len(self.vertices) + len(self.edges) + len(self.facets) + len(self.bodies)
        )

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
        return (
            index in self.vertices
            or index in self.edges
            or index in self.facets
            or index in self.bodies
        )
