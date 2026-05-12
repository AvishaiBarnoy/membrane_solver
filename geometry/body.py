"""Body entity for membrane meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from geometry.facet import _fast_cross

if TYPE_CHECKING:
    from geometry.mesh import Mesh


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

    def _get_triangle_rows(self, mesh: Mesh) -> Optional[np.ndarray]:
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
        mesh: Mesh,
        positions: np.ndarray | None = None,
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
        self, mesh: Mesh, positions: np.ndarray, grad_arr: np.ndarray, factor: float
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
        mesh: Mesh,
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

    def compute_volume_gradient(self, mesh: Mesh) -> Dict[int, np.ndarray]:
        """
        Compute the gradient of the volume with respect to each vertex in the body.
        Returns a dictionary mapping vertex indices to gradient vectors (np.ndarray).
        This version subtracts the body’s centroid so that the tetrahedron formula is applied
        relative to the body’s center rather than the origin.
        """
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

    def compute_surface_area(self, mesh: Mesh) -> float:
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
        mesh: Mesh,
        positions: np.ndarray | None = None,
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
        mesh: Mesh,
        positions: np.ndarray | None = None,
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
