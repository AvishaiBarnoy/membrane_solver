# entities.py

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from core.exceptions import BodyOrientationError, InvalidEdgeIndexError
from core.ordered_unique_list import OrderedUniqueList
from core.parameters.global_parameters import GlobalParameters

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
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    tilt_fixed: bool = False
    tilt_in: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    tilt_out: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    tilt_fixed_in: bool = False
    tilt_fixed_out: bool = False
    _mesh: "Mesh | None" = field(default=None, repr=False, compare=False)
    _row: int = field(default=-1, repr=False, compare=False)

    def __getattribute__(self, name: str) -> Any:
        if name in ("tilt", "tilt_in", "tilt_out"):
            mesh = object.__getattribute__(self, "_mesh")
            row = object.__getattribute__(self, "_row")
            if mesh is not None and row >= 0:
                if name == "tilt":
                    return mesh.tilts_view()[row]
                if name == "tilt_in":
                    return mesh.tilts_in_view()[row]
                return mesh.tilts_out_view()[row]
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("fixed", "tilt_fixed", "tilt_fixed_in", "tilt_fixed_out"):
            old = self.__dict__.get(name, None)
            object.__setattr__(self, name, value)
            if old is not None and old != value:
                mesh = self.__dict__.get("_mesh")
                if mesh is not None:
                    if name == "fixed":
                        mesh._touch_fixed_flags()
                    else:
                        mesh._touch_tilt_fixed_flags()
            return
        if name in ("tilt", "tilt_in", "tilt_out"):
            arr = np.asarray(value, dtype=float)
            if arr.shape != (3,):
                raise ValueError(f"{name} must be a 3-vector")
            mesh = self.__dict__.get("_mesh")
            row = self.__dict__.get("_row", -1)
            if mesh is not None and row >= 0:
                if name == "tilt":
                    mesh.tilts_view()[row] = arr
                elif name == "tilt_in":
                    mesh.tilts_in_view()[row] = arr
                else:
                    mesh.tilts_out_view()[row] = arr
            object.__setattr__(self, name, arr)
            return
        object.__setattr__(self, name, value)

    def copy(self):
        return Vertex(
            self.index,
            self.position.copy(),
            fixed=self.fixed,
            options=self.options.copy(),
            tilt=self.tilt.copy(),
            tilt_fixed=self.tilt_fixed,
            tilt_in=self.tilt_in.copy(),
            tilt_out=self.tilt_out.copy(),
            tilt_fixed_in=self.tilt_fixed_in,
            tilt_fixed_out=self.tilt_fixed_out,
        )

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
    _cached_area: Optional[float] = field(init=False, default=None)
    _cached_area_grad: Optional[Dict[int, np.ndarray]] = field(init=False, default=None)
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
        is_cached_pos = positions is None or positions is getattr(
            mesh, "_positions_cache", None
        )
        if (
            is_cached_pos
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
        if self._cached_version == mesh._version and self._cached_area is not None:
            return self._cached_area
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
        self._cached_area = area
        self._cached_version = mesh._version
        return area

    def compute_area_and_gradient(
        self,
        mesh: "Mesh",
        positions: "np.ndarray | None" = None,
        index_map: Dict[int, int] | None = None,
    ) -> tuple[float, Dict[int, np.ndarray]]:
        is_cached_pos = positions is None or positions is getattr(
            mesh, "_positions_cache", None
        )
        if (
            is_cached_pos
            and self._cached_version == mesh._version
            and self._cached_area is not None
            and self._cached_area_grad is not None
        ):
            return self._cached_area, self._cached_area_grad

        total_area = 0.0
        total_grad: Dict[int, np.ndarray] = {}

        for facet_idx in self.facet_indices:
            facet = mesh.facets[facet_idx]
            area, grad = facet.compute_area_and_gradient(
                mesh, positions=positions, index_map=index_map
            )
            total_area += area
            for vid, gvec in grad.items():
                if vid not in total_grad:
                    total_grad[vid] = gvec.copy()
                else:
                    total_grad[vid] += gvec

        if is_cached_pos:
            self._cached_area = total_area
            self._cached_area_grad = total_grad
            self._cached_version = mesh._version

        return total_area, total_grad

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
        is_cached_pos = positions is None or positions is getattr(
            mesh, "_positions_cache", None
        )
        if (
            is_cached_pos
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

                if is_cached_pos:
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

        if is_cached_pos:
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
    energy_modules: List[str] = field(default_factory=OrderedUniqueList)
    constraint_modules: List[str] = field(default_factory=OrderedUniqueList)
    instructions: List[str] = field(default_factory=list)
    macros: Dict[str, List[str]] = field(default_factory=dict)
    definitions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    vertex_to_facets: Dict[int, set] = field(default_factory=dict)
    vertex_to_edges: Dict[int, set] = field(default_factory=dict)
    edge_to_facets: Dict[int, set] = field(default_factory=dict)

    # Cached array views and facet vertex loops for performance.
    vertex_ids: "np.ndarray | None" = None
    vertex_index_to_row: Dict[int, int] = field(default_factory=dict)
    facet_vertex_loops: Dict[int, "np.ndarray"] = field(default_factory=dict)

    _positions_cache: "np.ndarray | None" = None
    _positions_cache_version: int = -1
    _tilts_cache: "np.ndarray | None" = None
    _tilts_cache_version: int = -1
    _tilt_cache_counts: int = -1
    _tilt_cache_vertex_version: int = -1
    _tilts_version: int = 0
    _tilts_in_cache: "np.ndarray | None" = None
    _tilts_in_cache_version: int = -1
    _tilts_in_cache_counts: int = -1
    _tilts_in_cache_vertex_version: int = -1
    _tilts_in_version: int = 0
    _tilts_out_cache: "np.ndarray | None" = None
    _tilts_out_cache_version: int = -1
    _tilts_out_cache_counts: int = -1
    _tilts_out_cache_vertex_version: int = -1
    _tilts_out_version: int = 0
    _vertex_row_binding_version: int = -1
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

    _fixed_flags_version: int = 0
    _tilt_fixed_flags_version: int = 0
    _fixed_mask_cache: "np.ndarray | None" = None
    _fixed_mask_version: int = -1
    _fixed_mask_vertex_version: int = -1

    _parameter_array_cache: Dict[str, "np.ndarray"] = field(default_factory=dict)
    _parameter_cache_version: int = -1

    _curvature_cache: Dict[str, Any] = field(default_factory=dict)
    _curvature_version: int = -1
    _geometry_freeze_depth: int = 0
    _geometry_freeze_version: int = -1
    _geometry_freeze_loops_version: int = -1
    _geometry_freeze_positions_id: int | None = None

    _cached_total_area: Optional[float] = field(init=False, default=None)
    _cached_total_area_grad: Optional[Dict[int, np.ndarray]] = field(
        init=False, default=None
    )
    _cached_total_area_version: int = field(init=False, default=-1)

    _cached_tri_areas: Optional[np.ndarray] = field(init=False, default=None)
    _cached_tri_normals: Optional[np.ndarray] = field(init=False, default=None)
    _cached_tri_areas_version: int = field(init=False, default=-1)

    _cached_vertex_normals: Optional[np.ndarray] = field(init=False, default=None)
    _cached_vertex_normals_version: int = field(init=False, default=-1)
    _cached_vertex_normals_loops_version: int = field(init=False, default=-1)

    # Cached P1 triangle shape gradients (for divergence operators).
    _cached_p1_tri_areas: Optional[np.ndarray] = field(init=False, default=None)
    _cached_p1_tri_g0: Optional[np.ndarray] = field(init=False, default=None)
    _cached_p1_tri_g1: Optional[np.ndarray] = field(init=False, default=None)
    _cached_p1_tri_g2: Optional[np.ndarray] = field(init=False, default=None)
    _cached_p1_tri_grads_version: int = field(init=False, default=-1)
    _cached_p1_tri_grads_rows_version: int = field(init=False, default=-1)

    _cached_barycentric_vertex_areas: Optional[np.ndarray] = field(
        init=False, default=None
    )
    _cached_barycentric_vertex_areas_version: int = field(init=False, default=-1)
    _cached_barycentric_vertex_areas_rows_version: int = field(init=False, default=-1)

    _version: int = 0
    _vertex_ids_version: int = 0

    def increment_version(self):
        self._version += 1

    def _touch_fixed_flags(self) -> None:
        self._fixed_flags_version += 1

    def _touch_tilt_fixed_flags(self) -> None:
        self._tilt_fixed_flags_version += 1

    def _geometry_cache_active(self, positions: Optional[np.ndarray]) -> bool:
        """Return True when positions correspond to the active geometry cache."""
        if positions is None:
            positions = getattr(self, "_positions_cache", None)

        if positions is getattr(self, "_positions_cache", None):
            return self._positions_cache_version == self._version

        if self._geometry_freeze_depth <= 0:
            return False

        if self._geometry_freeze_version != self._version:
            return False

        if self._geometry_freeze_loops_version != self._facet_loops_version:
            return False

        if self._geometry_freeze_positions_id is None:
            return False

        return id(positions) == self._geometry_freeze_positions_id

    @contextmanager
    def geometry_freeze(self, positions: Optional[np.ndarray] = None):
        """Freeze geometry caches while positions remain fixed."""
        if self._geometry_freeze_depth == 0:
            if positions is None:
                positions = self.positions_view()
            self._geometry_freeze_version = self._version
            self._geometry_freeze_loops_version = self._facet_loops_version
            self._geometry_freeze_positions_id = id(positions)
            self._curvature_version = self._version

        self._geometry_freeze_depth += 1
        try:
            yield
        finally:
            self._geometry_freeze_depth -= 1
            if self._geometry_freeze_depth == 0:
                self._geometry_freeze_version = -1
                self._geometry_freeze_loops_version = -1
                self._geometry_freeze_positions_id = None

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
            and self._fixed_mask_version == self._fixed_flags_version
            and self._fixed_mask_vertex_version == self._vertex_ids_version
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
        self._fixed_mask_version = self._fixed_flags_version
        self._fixed_mask_vertex_version = self._vertex_ids_version
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
        new_mesh.energy_modules = OrderedUniqueList(getattr(self, "energy_modules", []))
        new_mesh.constraint_modules = OrderedUniqueList(
            getattr(self, "constraint_modules", [])
        )
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
        if (
            self._cached_total_area_version == self._version
            and self._cached_total_area is not None
        ):
            return self._cached_total_area
        area = sum(facet.compute_area(self) for facet in self.facets.values())
        self._cached_total_area = area
        self._cached_total_area_version = self._version
        return area

    def compute_total_area_and_gradient(
        self,
        positions: "np.ndarray | None" = None,
        index_map: Dict[int, int] | None = None,
    ) -> tuple[float, Dict[int, np.ndarray]]:
        is_cached_pos = positions is None or positions is getattr(
            self, "_positions_cache", None
        )
        if (
            is_cached_pos
            and self._cached_total_area_version == self._version
            and self._cached_total_area is not None
            and self._cached_total_area_grad is not None
        ):
            return self._cached_total_area, self._cached_total_area_grad

        total_area = 0.0
        total_grad: Dict[int, np.ndarray] = {}
        for facet in self.facets.values():
            area, grad = facet.compute_area_and_gradient(
                self, positions=positions, index_map=index_map
            )
            total_area += area
            for vid, gvec in grad.items():
                if vid not in total_grad:
                    total_grad[vid] = gvec.copy()
                else:
                    total_grad[vid] += gvec

        if is_cached_pos:
            self._cached_total_area = total_area
            self._cached_total_area_grad = total_grad
            self._cached_total_area_version = self._version

        return total_area, total_grad

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

    def validate_body_orientation(self) -> bool:
        """Validate that facets in each body have consistent orientation.

        For every edge shared by exactly two facets within a body, the two
        facets must traverse the shared edge with opposite orientation
        (i.e., one facet uses ``+eid`` and the other uses ``-eid``).

        Boundary edges (edges used by only one facet in the body) are allowed;
        this check focuses on orientation consistency, not closure.
        """
        if not self.bodies:
            return True

        for body in self.bodies.values():
            edge_uses: dict[int, list[tuple[int, int]]] = {}
            for fid in body.facet_indices:
                facet = self.facets.get(fid)
                if facet is None:
                    raise BodyOrientationError(
                        f"Body {body.index} references missing facet {fid}.",
                        body_index=body.index,
                        mesh=self,
                    )
                for signed_ei in facet.edge_indices:
                    ei = abs(int(signed_ei))
                    sign = 1 if int(signed_ei) > 0 else -1
                    edge_uses.setdefault(ei, []).append((facet.index, sign))

            for ei, uses in edge_uses.items():
                if len(uses) > 2:
                    facets = [fid for fid, _ in uses]
                    raise BodyOrientationError(
                        f"Body {body.index} is non-manifold: edge {ei} is used by "
                        f"{len(uses)} facets {facets}.",
                        body_index=body.index,
                        edge_index=ei,
                        mesh=self,
                    )
                if len(uses) == 2:
                    (f0, s0), (f1, s1) = uses
                    if s0 != -s1:
                        raise BodyOrientationError(
                            f"Body {body.index} has inconsistent facet orientation across "
                            f"edge {ei}: facets {f0} and {f1} traverse it with the same "
                            "direction.",
                            body_index=body.index,
                            edge_index=ei,
                            facet_indices=(f0, f1),
                            mesh=self,
                        )

        return True

    def _body_edge_uses(self, body: "Body") -> dict[int, list[tuple[int, int]]]:
        """Return mapping ``edge_id -> [(facet_id, sign), ...]`` within a body."""
        edge_uses: dict[int, list[tuple[int, int]]] = {}
        for fid in body.facet_indices:
            facet = self.facets.get(fid)
            if facet is None:
                raise BodyOrientationError(
                    f"Body {body.index} references missing facet {fid}.",
                    body_index=body.index,
                    mesh=self,
                )
            for signed_ei in facet.edge_indices:
                ei = abs(int(signed_ei))
                sign = 1 if int(signed_ei) > 0 else -1
                edge_uses.setdefault(ei, []).append((facet.index, sign))
        return edge_uses

    def _body_is_closed(self, body: "Body") -> bool:
        """Return ``True`` when the body's facets form a closed 2-manifold."""
        edge_uses = self._body_edge_uses(body)
        return bool(edge_uses) and all(len(uses) == 2 for uses in edge_uses.values())

    def validate_body_outwardness(self, volume_tol: float = 1e-12) -> bool:
        """Validate that closed bodies have outward (positive) signed volume.

        Open bodies are skipped because their volume depends on the chosen
        closure surface (e.g. a drop on a hard plane).
        """
        if not self.bodies:
            return True

        tol = float(volume_tol)
        for body in self.bodies.values():
            if not self._body_is_closed(body):
                continue
            vol = float(body.compute_volume(self))
            if abs(vol) <= tol:
                continue
            if vol < -tol:
                raise BodyOrientationError(
                    f"Body {body.index} is inward-oriented (signed volume {vol:.6g} < 0).",
                    body_index=body.index,
                    mesh=self,
                )
        return True

    def orient_body_outward(self, body_index: int, volume_tol: float = 1e-12) -> int:
        """Flip all facets in ``body_index`` if signed volume is negative.

        Returns the number of flipped facets (0 for open bodies or positive volume).
        """
        body = self.bodies.get(body_index)
        if body is None:
            raise KeyError(f"Body {body_index} not found in mesh.")

        if not self._body_is_closed(body):
            return 0

        tol = float(volume_tol)
        vol = float(body.compute_volume(self))
        if vol >= -tol:
            return 0

        for fid in body.facet_indices:
            facet = self.facets.get(fid)
            if facet is None:  # pragma: no cover - defensive
                continue
            facet.edge_indices = [-int(ei) for ei in reversed(facet.edge_indices)]

        self.build_facet_vertex_loops()
        self.increment_version()
        return len(body.facet_indices)

    def orient_body_facets(self, body_index: int) -> int:
        """Re-orient facets in a body to make shared-edge orientations consistent.

        This flips a subset of facets so that any edge shared by two facets is
        traversed with opposite directions by the two facets.

        Returns the number of flipped facets.
        """
        body = self.bodies.get(body_index)
        if body is None:
            raise KeyError(f"Body {body_index} not found in mesh.")

        facet_ids = list(body.facet_indices)
        if not facet_ids:
            return 0

        # Build edge -> [(facet_id, sign)] within the body.
        edge_uses: dict[int, list[tuple[int, int]]] = {}
        for fid in facet_ids:
            facet = self.facets.get(fid)
            if facet is None:
                raise BodyOrientationError(
                    f"Body {body.index} references missing facet {fid}.",
                    body_index=body.index,
                    mesh=self,
                )
            for signed_ei in facet.edge_indices:
                ei = abs(int(signed_ei))
                sign = 1 if int(signed_ei) > 0 else -1
                edge_uses.setdefault(ei, []).append((facet.index, sign))

        # Build facet adjacency across shared edges.
        adjacency: dict[int, list[tuple[int, int, int]]] = {
            fid: [] for fid in facet_ids
        }
        for ei, uses in edge_uses.items():
            if len(uses) > 2:
                facets = [fid for fid, _ in uses]
                raise BodyOrientationError(
                    f"Cannot orient body {body.index}: edge {ei} is used by "
                    f"{len(uses)} facets {facets}.",
                    body_index=body.index,
                    edge_index=ei,
                    mesh=self,
                )
            if len(uses) != 2:
                continue
            (f0, s0), (f1, s1) = uses
            adjacency.setdefault(f0, []).append((f1, s0, s1))
            adjacency.setdefault(f1, []).append((f0, s1, s0))

        # BFS "flip parity" assignment.
        flips: dict[int, int] = {}
        queue: list[int] = []
        for start in facet_ids:
            if start in flips:
                continue
            flips[start] = 0
            queue.append(start)
            while queue:
                fid = queue.pop()
                f_flip = flips[fid]
                for nbr, sign_here, sign_nbr in adjacency.get(fid, []):
                    # If the two facets already traverse the shared edge in opposite
                    # directions, keep the same flip parity; otherwise flip the neighbor.
                    nbr_flip = f_flip if sign_here == -sign_nbr else 1 - f_flip
                    if nbr in flips:
                        if flips[nbr] != nbr_flip:
                            raise BodyOrientationError(
                                f"Cannot orient body {body.index}: inconsistent parity "
                                f"assignment between facets {fid} and {nbr}.",
                                body_index=body.index,
                                edge_index=None,
                                facet_indices=(fid, nbr),
                                mesh=self,
                            )
                        continue
                    flips[nbr] = nbr_flip
                    queue.append(nbr)

        flipped_count = 0
        for fid, flip in flips.items():
            if not flip:
                continue
            facet = self.facets[fid]
            facet.edge_indices = [-int(ei) for ei in reversed(facet.edge_indices)]
            flipped_count += 1

        if flipped_count:
            # Orientation affects cached vertex loops/normals/volume signs.
            self.build_facet_vertex_loops()
            self.increment_version()

        return flipped_count

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

    def tilts_view(self) -> "np.ndarray":
        """Return a dense ``(N_vertices, 3)`` array of vertex tilt vectors.

        The ordering matches ``vertex_ids``. The returned array is cached and
        invalidated via ``touch_tilts``/``set_tilts_from_array``.
        """
        import numpy as np

        self.build_position_cache()
        n_verts = len(self.vertex_ids)
        needs_rebind = (
            self._tilts_cache is None
            or self._tilts_cache.shape != (n_verts, 3)
            or self._tilt_cache_counts != n_verts
            or self._tilt_cache_vertex_version != self._vertex_ids_version
        )
        if needs_rebind:
            old_cache = (
                None
                if self._tilts_cache is None or self._tilts_cache.shape[0] != n_verts
                else self._tilts_cache
            )
            new_cache = np.empty((n_verts, 3), dtype=float, order="F")
            for i, vid in enumerate(self.vertex_ids):
                vertex = self.vertices[int(vid)]
                if (
                    old_cache is not None
                    and vertex._mesh is self
                    and 0 <= vertex._row < old_cache.shape[0]
                ):
                    new_cache[i] = old_cache[vertex._row]
                else:
                    new_cache[i] = object.__getattribute__(vertex, "tilt")
                vertex._mesh = self
                vertex._row = i
            self._tilts_cache = new_cache
            self._tilt_cache_counts = n_verts
            self._tilt_cache_vertex_version = self._vertex_ids_version
            self._vertex_row_binding_version = self._vertex_ids_version
        self._tilts_cache_version = self._tilts_version
        return self._tilts_cache

    def tilts_in_view(self) -> "np.ndarray":
        """Return a dense ``(N_vertices, 3)`` array of inner-leaflet tilt vectors."""
        import numpy as np

        self.build_position_cache()
        n_verts = len(self.vertex_ids)
        needs_rebind = (
            self._tilts_in_cache is None
            or self._tilts_in_cache.shape != (n_verts, 3)
            or self._tilts_in_cache_counts != n_verts
            or self._tilts_in_cache_vertex_version != self._vertex_ids_version
        )
        if needs_rebind:
            old_cache = (
                None
                if self._tilts_in_cache is None
                or self._tilts_in_cache.shape[0] != n_verts
                else self._tilts_in_cache
            )
            new_cache = np.empty((n_verts, 3), dtype=float, order="F")
            for i, vid in enumerate(self.vertex_ids):
                vertex = self.vertices[int(vid)]
                if (
                    old_cache is not None
                    and vertex._mesh is self
                    and 0 <= vertex._row < old_cache.shape[0]
                ):
                    new_cache[i] = old_cache[vertex._row]
                else:
                    new_cache[i] = object.__getattribute__(vertex, "tilt_in")
                vertex._mesh = self
                vertex._row = i
            self._tilts_in_cache = new_cache
            self._tilts_in_cache_counts = n_verts
            self._tilts_in_cache_vertex_version = self._vertex_ids_version
            self._vertex_row_binding_version = self._vertex_ids_version
        self._tilts_in_cache_version = self._tilts_in_version
        return self._tilts_in_cache

    def tilts_out_view(self) -> "np.ndarray":
        """Return a dense ``(N_vertices, 3)`` array of outer-leaflet tilt vectors."""
        import numpy as np

        self.build_position_cache()
        n_verts = len(self.vertex_ids)
        needs_rebind = (
            self._tilts_out_cache is None
            or self._tilts_out_cache.shape != (n_verts, 3)
            or self._tilts_out_cache_counts != n_verts
            or self._tilts_out_cache_vertex_version != self._vertex_ids_version
        )
        if needs_rebind:
            old_cache = (
                None
                if self._tilts_out_cache is None
                or self._tilts_out_cache.shape[0] != n_verts
                else self._tilts_out_cache
            )
            new_cache = np.empty((n_verts, 3), dtype=float, order="F")
            for i, vid in enumerate(self.vertex_ids):
                vertex = self.vertices[int(vid)]
                if (
                    old_cache is not None
                    and vertex._mesh is self
                    and 0 <= vertex._row < old_cache.shape[0]
                ):
                    new_cache[i] = old_cache[vertex._row]
                else:
                    new_cache[i] = object.__getattribute__(vertex, "tilt_out")
                vertex._mesh = self
                vertex._row = i
            self._tilts_out_cache = new_cache
            self._tilts_out_cache_counts = n_verts
            self._tilts_out_cache_vertex_version = self._vertex_ids_version
            self._vertex_row_binding_version = self._vertex_ids_version
        self._tilts_out_cache_version = self._tilts_out_version
        return self._tilts_out_cache

    def touch_tilts(self) -> None:
        """Invalidate cached tilt arrays after direct per-vertex updates."""
        self._tilts_version += 1
        self._tilts_cache_version = self._tilts_version

    def touch_tilts_in(self) -> None:
        """Invalidate cached inner-leaflet tilt arrays after per-vertex updates."""
        self._tilts_in_version += 1
        self._tilts_in_cache_version = self._tilts_in_version

    def touch_tilts_out(self) -> None:
        """Invalidate cached outer-leaflet tilt arrays after per-vertex updates."""
        self._tilts_out_version += 1
        self._tilts_out_cache_version = self._tilts_out_version

    def set_tilts_from_array(self, tilts: "np.ndarray") -> None:
        """Scatter a dense tilt array back onto vertex objects."""
        import numpy as np

        self.build_position_cache()
        tilts_arr = np.asarray(tilts, dtype=float)
        if tilts_arr.shape != (len(self.vertex_ids), 3):
            raise ValueError("tilts must have shape (N_vertices, 3)")

        self._tilts_version += 1
        self._tilts_cache = np.array(tilts_arr, dtype=float, order="F", copy=True)
        self._tilts_cache_version = self._tilts_version
        self._tilt_cache_counts = len(self.vertex_ids)
        self._tilt_cache_vertex_version = self._vertex_ids_version
        for row, vid in enumerate(self.vertex_ids):
            vertex = self.vertices[int(vid)]
            vertex._mesh = self
            vertex._row = row
            object.__setattr__(vertex, "tilt", self._tilts_cache[row].copy())
        self._vertex_row_binding_version = self._vertex_ids_version

    def set_tilts_in_from_array(self, tilts: "np.ndarray") -> None:
        """Scatter a dense inner-leaflet tilt array back onto vertex objects."""
        import numpy as np

        self.build_position_cache()
        tilts_arr = np.asarray(tilts, dtype=float)
        if tilts_arr.shape != (len(self.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

        self._tilts_in_version += 1
        self._tilts_in_cache = np.array(tilts_arr, dtype=float, order="F", copy=True)
        self._tilts_in_cache_version = self._tilts_in_version
        self._tilts_in_cache_counts = len(self.vertex_ids)
        self._tilts_in_cache_vertex_version = self._vertex_ids_version
        cache = self._tilts_in_cache
        need_rebind = self._vertex_row_binding_version != self._vertex_ids_version
        for row, vid in enumerate(self.vertex_ids):
            vertex = self.vertices[int(vid)]
            if need_rebind:
                vertex._mesh = self
                vertex._row = row
            # Store a row view to avoid per-vertex copy allocations.
            object.__setattr__(vertex, "tilt_in", cache[row])
        if need_rebind:
            self._vertex_row_binding_version = self._vertex_ids_version

    def set_tilts_out_from_array(self, tilts: "np.ndarray") -> None:
        """Scatter a dense outer-leaflet tilt array back onto vertex objects."""
        import numpy as np

        self.build_position_cache()
        tilts_arr = np.asarray(tilts, dtype=float)
        if tilts_arr.shape != (len(self.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")

        self._tilts_out_version += 1
        self._tilts_out_cache = np.array(tilts_arr, dtype=float, order="F", copy=True)
        self._tilts_out_cache_version = self._tilts_out_version
        self._tilts_out_cache_counts = len(self.vertex_ids)
        self._tilts_out_cache_vertex_version = self._vertex_ids_version
        cache = self._tilts_out_cache
        need_rebind = self._vertex_row_binding_version != self._vertex_ids_version
        for row, vid in enumerate(self.vertex_ids):
            vertex = self.vertices[int(vid)]
            if need_rebind:
                vertex._mesh = self
                vertex._row = row
            # Store a row view to avoid per-vertex copy allocations.
            object.__setattr__(vertex, "tilt_out", cache[row])
        if need_rebind:
            self._vertex_row_binding_version = self._vertex_ids_version

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

    def triangle_areas(self, positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a cached array of triangle areas for facets in triangle_row_cache."""
        is_cached_pos = positions is None or positions is getattr(
            self, "_positions_cache", None
        )
        if (
            is_cached_pos
            and self._cached_tri_areas_version == self._version
            and self._cached_tri_areas is not None
        ):
            return self._cached_tri_areas

        tri_rows, _ = self.triangle_row_cache()
        if tri_rows is None:
            return np.array([])

        if positions is None:
            positions = self.positions_view()

        v0 = positions[tri_rows[:, 0]]
        v1 = positions[tri_rows[:, 1]]
        v2 = positions[tri_rows[:, 2]]

        normals = _fast_cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(normals, axis=1)

        if is_cached_pos:
            self._cached_tri_areas = areas
            self._cached_tri_normals = normals
            self._cached_tri_areas_version = self._version
        return areas

    def triangle_normals(self, positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a cached array of unnormalized triangle normals."""
        is_cached_pos = positions is None or positions is getattr(
            self, "_positions_cache", None
        )
        if (
            is_cached_pos
            and self._cached_tri_areas_version == self._version
            and self._cached_tri_normals is not None
        ):
            return self._cached_tri_normals

        self.triangle_areas(positions)  # This populates normals
        if is_cached_pos:
            return self._cached_tri_normals

        # If not cached pos, we need to return the one computed just now.
        # But triangle_areas only returns areas.
        # Let's recompute or refactor.
        if positions is None:
            positions = self.positions_view()
        tri_rows, _ = self.triangle_row_cache()
        v0 = positions[tri_rows[:, 0]]
        v1 = positions[tri_rows[:, 1]]
        v2 = positions[tri_rows[:, 2]]
        return _fast_cross(v1 - v0, v2 - v0)

    def barycentric_vertex_areas(
        self,
        positions: Optional[np.ndarray] = None,
        *,
        tri_rows: Optional[np.ndarray] = None,
        areas: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        cache: Optional[bool] = None,
    ) -> np.ndarray:
        """Return cached barycentric vertex areas from triangle areas.

        The barycentric area at a vertex is one third of the areas of its
        incident triangles. Degenerate triangles (area < 1e-12) are ignored.
        When ``areas`` is provided, it must align with ``tri_rows``.
        """
        import numpy as np

        self.build_position_cache()
        n_verts = len(self.vertex_ids)

        if tri_rows is None:
            tri_rows, _ = self.triangle_row_cache()
        if tri_rows is None or tri_rows.size == 0:
            return np.zeros(n_verts, dtype=float)

        if positions is None:
            positions = self.positions_view()

        if cache is None:
            cache = areas is None

        use_cache = cache and self._geometry_cache_active(positions)
        if (
            use_cache
            and self._cached_barycentric_vertex_areas_version == self._version
            and self._cached_barycentric_vertex_areas_rows_version
            == self._facet_loops_version
            and self._cached_barycentric_vertex_areas is not None
            and len(self._cached_barycentric_vertex_areas) == n_verts
        ):
            return self._cached_barycentric_vertex_areas

        if areas is None:
            v0 = positions[tri_rows[:, 0]]
            v1 = positions[tri_rows[:, 1]]
            v2 = positions[tri_rows[:, 2]]

            n = _fast_cross(v1 - v0, v2 - v0)
            n_norm = np.linalg.norm(n, axis=1)
            mask = n_norm >= 1e-12
            areas = 0.5 * n_norm[mask]
            tri_rows = tri_rows[mask]
        else:
            areas = np.asarray(areas, dtype=float)
            if mask is not None:
                tri_rows = tri_rows[mask]
                areas = areas[mask]

        vertex_areas = np.zeros(n_verts, dtype=float)
        if areas.size:
            area_thirds = areas / 3.0
            np.add.at(vertex_areas, tri_rows[:, 0], area_thirds)
            np.add.at(vertex_areas, tri_rows[:, 1], area_thirds)
            np.add.at(vertex_areas, tri_rows[:, 2], area_thirds)

        if use_cache:
            self._cached_barycentric_vertex_areas = vertex_areas
            self._cached_barycentric_vertex_areas_version = self._version
            self._cached_barycentric_vertex_areas_rows_version = (
                self._facet_loops_version
            )

        return vertex_areas

    def vertex_normals(self, positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Return per-vertex unit normals from triangle normals.

        The result is in ``vertex_ids`` row order. Vertices without incident
        triangles receive a zero normal.
        """
        import numpy as np

        is_cached_pos = positions is None or positions is getattr(
            self, "_positions_cache", None
        )
        if (
            is_cached_pos
            and self._cached_vertex_normals is not None
            and self._cached_vertex_normals_version == self._version
            and self._cached_vertex_normals_loops_version == self._facet_loops_version
        ):
            return self._cached_vertex_normals

        tri_rows, _ = self.triangle_row_cache()
        self.build_position_cache()
        n_verts = len(self.vertex_ids)
        if tri_rows is None or len(tri_rows) == 0:
            normals = np.zeros((n_verts, 3), dtype=float)
            if is_cached_pos:
                self._cached_vertex_normals = normals
                self._cached_vertex_normals_version = self._version
                self._cached_vertex_normals_loops_version = self._facet_loops_version
            return normals

        if positions is None:
            positions = self.positions_view()

        v0 = positions[tri_rows[:, 0]]
        v1 = positions[tri_rows[:, 1]]
        v2 = positions[tri_rows[:, 2]]
        tri_normals = _fast_cross(v1 - v0, v2 - v0)

        normals = np.zeros((n_verts, 3), dtype=float)
        np.add.at(normals, tri_rows[:, 0], tri_normals)
        np.add.at(normals, tri_rows[:, 1], tri_normals)
        np.add.at(normals, tri_rows[:, 2], tri_normals)

        lens = np.linalg.norm(normals, axis=1)
        mask = lens >= 1e-12
        normals[mask] /= lens[mask][:, None]

        if is_cached_pos:
            self._cached_vertex_normals = normals
            self._cached_vertex_normals_version = self._version
            self._cached_vertex_normals_loops_version = self._facet_loops_version
        return normals

    def p1_triangle_shape_gradient_cache(
        self, positions: Optional[np.ndarray] = None
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        """Return cached P1 triangle shape gradients for ``triangle_row_cache``.

        The returned arrays are aligned with ``triangle_row_cache()`` ordering.
        When geometry caching is active (via ``positions_view`` or
        ``geometry_freeze``), this avoids recomputing triangle cross products
        and basis gradients across repeated divergence evaluations.
        """
        import numpy as np

        tri_rows, _ = self.triangle_row_cache()
        if tri_rows is None or tri_rows.size == 0:
            zeros1 = np.zeros(0, dtype=float)
            zeros3 = np.zeros((0, 3), dtype=float)
            tri_empty = np.zeros((0, 3), dtype=np.int32)
            return zeros1, zeros3, zeros3, zeros3, tri_empty

        if positions is None:
            positions = self.positions_view()

        use_cache = self._geometry_cache_active(positions)
        if (
            use_cache
            and self._cached_p1_tri_grads_version == self._version
            and self._cached_p1_tri_grads_rows_version == self._facet_loops_version
            and self._cached_p1_tri_areas is not None
            and self._cached_p1_tri_g0 is not None
            and self._cached_p1_tri_g1 is not None
            and self._cached_p1_tri_g2 is not None
        ):
            return (
                self._cached_p1_tri_areas,
                self._cached_p1_tri_g0,
                self._cached_p1_tri_g1,
                self._cached_p1_tri_g2,
                tri_rows,
            )

        v0 = positions[tri_rows[:, 0]]
        v1 = positions[tri_rows[:, 1]]
        v2 = positions[tri_rows[:, 2]]

        n = _fast_cross(v1 - v0, v2 - v0)
        n2 = np.einsum("ij,ij->i", n, n)
        denom = np.maximum(n2, 1e-20)

        e0 = v2 - v1
        e1 = v0 - v2
        e2 = v1 - v0

        g0 = _fast_cross(n, e0) / denom[:, None]
        g1 = _fast_cross(n, e1) / denom[:, None]
        g2 = _fast_cross(n, e2) / denom[:, None]
        area = 0.5 * np.sqrt(np.maximum(n2, 0.0))

        if use_cache:
            self._cached_p1_tri_areas = area
            self._cached_p1_tri_g0 = g0
            self._cached_p1_tri_g1 = g1
            self._cached_p1_tri_g2 = g2
            self._cached_p1_tri_grads_version = self._version
            self._cached_p1_tri_grads_rows_version = self._facet_loops_version

        return area, g0, g1, g2, tri_rows

    def project_tilts_to_tangent(self) -> None:
        """Project all vertex tilt vectors into their local tangent planes."""
        import numpy as np

        self.build_position_cache()
        tilts = self.tilts_view()
        tilts_in = self.tilts_in_view()
        tilts_out = self.tilts_out_view()
        if (
            (tilts is None or tilts.size == 0)
            and (tilts_in is None or tilts_in.size == 0)
            and (tilts_out is None or tilts_out.size == 0)
        ):
            return

        normals = self.vertex_normals()
        if normals.size == 0:
            return

        if tilts is not None and tilts.size:
            dot = np.einsum("ij,ij->i", tilts, normals)
            projected = tilts - dot[:, None] * normals
            self.set_tilts_from_array(projected)

        if tilts_in is not None and tilts_in.size:
            dot_in = np.einsum("ij,ij->i", tilts_in, normals)
            projected_in = tilts_in - dot_in[:, None] * normals
            self.set_tilts_in_from_array(projected_in)

        if tilts_out is not None and tilts_out.size:
            dot_out = np.einsum("ij,ij->i", tilts_out, normals)
            projected_out = tilts_out - dot_out[:, None] * normals
            self.set_tilts_out_from_array(projected_out)

    def initialize_tilts_from_options(self) -> None:
        """Initialize tangent tilt vectors from vertex options.

        Supported option formats
        ------------------------
        - ``[t1, t2]``: components in a deterministic local tangent basis.
        - ``[tx, ty, tz]``: ambient vector projected onto the tangent plane.
        """
        import numpy as np

        def _apply_tilt_field(field_key: str, setter_name: str) -> bool:
            has_field = False
            for vertex in self.vertices.values():
                opts = getattr(vertex, "options", None)
                if isinstance(opts, dict) and opts.get(field_key) is not None:
                    has_field = True
                    break
            if not has_field:
                return False

            self.build_position_cache()
            normals = self.vertex_normals()
            if normals.size == 0:
                return False

            ref_x = np.array([1.0, 0.0, 0.0], dtype=float)
            ref_y = np.array([0.0, 1.0, 0.0], dtype=float)

            for row, vid in enumerate(self.vertex_ids):
                vertex = self.vertices[int(vid)]
                raw = getattr(vertex, "options", {}).get(field_key)
                if raw is None:
                    continue

                if not isinstance(raw, (list, tuple)):
                    raise TypeError(
                        f"Vertex {int(vid)} {field_key} must be a 2- or 3-vector; got {raw!r}"
                    )

                n = normals[row]
                if np.linalg.norm(n) < 1e-12:
                    if len(raw) == 2:
                        vec = np.asarray([raw[0], raw[1], 0.0], dtype=float)
                    elif len(raw) == 3:
                        vec = np.asarray(raw, dtype=float)
                    else:
                        raise TypeError(
                            f"Vertex {int(vid)} {field_key} must have length 2 or 3; got {raw!r}"
                        )
                    setattr(vertex, setter_name, vec)
                    continue

                if len(raw) == 2:
                    t1, t2 = (float(raw[0]), float(raw[1]))
                    e1 = ref_x - float(np.dot(ref_x, n)) * n
                    if np.linalg.norm(e1) < 1e-12:
                        e1 = ref_y - float(np.dot(ref_y, n)) * n
                    e1_norm = np.linalg.norm(e1)
                    if e1_norm < 1e-12:
                        continue
                    e1 = e1 / e1_norm
                    e2 = np.cross(n, e1)
                    vec = t1 * e1 + t2 * e2
                elif len(raw) == 3:
                    t = np.asarray(raw, dtype=float)
                    vec = t - float(np.dot(t, n)) * n
                else:
                    raise TypeError(
                        f"Vertex {int(vid)} {field_key} must have length 2 or 3; got {raw!r}"
                    )
                setattr(vertex, setter_name, vec)

            return True

        any_tilt = _apply_tilt_field("tilt", "tilt")
        any_tilt_in = _apply_tilt_field("tilt_in", "tilt_in")
        any_tilt_out = _apply_tilt_field("tilt_out", "tilt_out")

        if any_tilt:
            self.touch_tilts()
        if any_tilt_in:
            self.touch_tilts_in()
        if any_tilt_out:
            self.touch_tilts_out()

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

        # 3. Ensure bodies have consistent facet orientation (if defined).
        self.validate_body_orientation()
        self.validate_body_outwardness()

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
