"""Facet entity for membrane meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from geometry.mesh import Mesh


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

    _cached_area: Optional[float] = field(init=False, default=None)
    _cached_grad: Optional[Dict[int, np.ndarray]] = field(init=False, default=None)
    _cached_version: int = field(init=False, default=-1)

    def copy(self):
        return Facet(
            index=self.index,
            edge_indices=self.edge_indices[:],
            refine=self.refine,
            fixed=self.fixed,
            surface_tension=self.surface_tension,
            options=self.options.copy(),
        )

    def compute_normal(self, mesh: Mesh) -> np.ndarray:
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

    def normal(self, mesh: Mesh) -> np.ndarray:
        """Compute normalized normal vector"""
        n = self.compute_normal(mesh)
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError("Degenerate facet with zero normal.")
        return n / norm

    def compute_area(self, mesh: Mesh) -> float:
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

    def compute_area_gradient(self, mesh: Mesh) -> Dict[int, np.ndarray]:
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
        mesh: Mesh,
        positions: np.ndarray | None = None,
        index_map: Dict[int, int] | None = None,
    ) -> tuple[float, Dict[int, np.ndarray]]:
        """
        Compute both the facet area and its gradient with respect to vertex positions.

        This combines the work of ``compute_area`` and ``compute_area_gradient`` so
        callers that need both can avoid redundant geometric computation.
        Uses the Shoelace formula.
        """
        is_cached_pos = positions is None or positions is getattr(
            mesh, "_positions_cache", None
        )
        if (
            is_cached_pos
            and self._cached_version == mesh._version
            and self._cached_area is not None
            and self._cached_grad is not None
        ):
            return self._cached_area, self._cached_grad

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

        if is_cached_pos:
            self._cached_area = area
            self._cached_grad = grad
            self._cached_version = mesh._version

        return area, grad
