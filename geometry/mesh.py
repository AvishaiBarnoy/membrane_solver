"""Mesh orchestrator for membrane simulations."""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from core.exceptions import InvalidEdgeIndexError
from core.ordered_unique_list import OrderedUniqueList
from core.parameters.global_parameters import GlobalParameters
from geometry.body import Body
from geometry.cache_checks import (
    barycentric_cache_valid,
    is_cached_positions,
    p1_triangle_cache_valid,
    triangle_areas_cache_valid,
    vertex_normals_cache_valid,
)
from geometry.cache_writes import (
    store_barycentric_vertex_areas_cache,
    store_p1_triangle_grad_cache,
    store_triangle_area_normals_cache,
    store_vertex_normals_cache,
)
from geometry.edge import Edge
from geometry.facet import Facet
from geometry.triangle_ops import (
    barycentric_vertex_areas_from_triangles,
    p1_triangle_shape_gradients,
    triangle_normals,
    triangle_normals_and_areas,
    vertex_unit_normals_from_triangles,
)
from geometry.triangle_rows import triangle_facets_from_loops, triangle_rows_from_loops
from geometry.vertex import Vertex

logger = logging.getLogger("membrane_solver")


class MeshError(Exception):
    """Custom exception for invalid mesh topology or geometry."""


@dataclass
class Mesh:
    vertices: Dict[int, Vertex] = field(default_factory=dict)
    edges: Dict[int, Edge] = field(default_factory=dict)
    facets: Dict[int, Facet] = field(default_factory=dict)
    bodies: Dict[int, Body] = field(default_factory=dict)
    global_parameters: GlobalParameters = None
    energy_modules: List[str] = field(default_factory=OrderedUniqueList)
    constraint_modules: List[str] = field(default_factory=OrderedUniqueList)
    instructions: List[str] = field(default_factory=list)
    macros: Dict[str, List[str]] = field(default_factory=dict)
    definitions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    vertex_to_facets: Dict[int, set] = field(default_factory=dict)
    vertex_to_edges: Dict[int, set] = field(default_factory=dict)
    edge_to_facets: Dict[int, set] = field(default_factory=dict)

    # Cached array views and facet vertex loops for performance.
    vertex_ids: np.ndarray | None = None
    vertex_index_to_row: Dict[int, int] = field(default_factory=dict)
    facet_vertex_loops: Dict[int, np.ndarray] = field(default_factory=dict)

    _positions_cache: np.ndarray | None = None
    _positions_cache_version: int = -1
    _tilts_cache: np.ndarray | None = None
    _tilts_cache_version: int = -1
    _tilt_cache_counts: int = -1
    _tilt_cache_vertex_version: int = -1
    _tilts_version: int = 0
    _tilts_in_cache: np.ndarray | None = None
    _tilts_in_cache_version: int = -1
    _tilts_in_cache_counts: int = -1
    _tilts_in_cache_vertex_version: int = -1
    _tilts_in_version: int = 0
    _tilts_out_cache: np.ndarray | None = None
    _tilts_out_cache_version: int = -1
    _tilts_out_cache_counts: int = -1
    _tilts_out_cache_vertex_version: int = -1
    _tilts_out_version: int = 0
    _vertex_row_binding_version: int = -1
    _triangle_rows_cache: np.ndarray | None = None
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
    _fixed_mask_cache: np.ndarray | None = None
    _fixed_mask_version: int = -1
    _fixed_mask_vertex_version: int = -1

    _parameter_array_cache: Dict[str, np.ndarray] = field(default_factory=dict)
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
    def fixed_mask(self) -> np.ndarray:
        """Return a boolean array of fixed status for vertices."""
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

        for i, vid in enumerate(self.vertex_ids):
            if self.vertices[vid].fixed:
                mask[i] = True

        self._fixed_mask_cache = mask
        self._fixed_mask_version = self._fixed_flags_version
        self._fixed_mask_vertex_version = self._vertex_ids_version
        return mask

    def get_facet_parameter_array(
        self, param_name: str, default_val: float | None = None
    ) -> np.ndarray:
        """Return a cached array of parameter values for all cached triangle rows."""
        rows, facets = self.triangle_row_cache()
        if rows is None:
            return np.array([])

        cache_key = f"facet_{param_name}"
        if (
            self._parameter_cache_version == self._version
            and cache_key in self._parameter_array_cache
        ):
            return self._parameter_array_cache[cache_key]

        if self._parameter_cache_version != self._version:
            self._parameter_array_cache.clear()
            self._parameter_cache_version = self._version

        n_facets = len(facets)
        if default_val is None:
            default_val = self.global_parameters.get(param_name) or 0.0

        arr = np.full(n_facets, default_val, dtype=float)

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

        self.triangle_row_cache()

        self._facet_to_row_cache = {
            fid: i for i, fid in enumerate(self._triangle_row_facets)
        }
        self._facet_to_row_version = self._facet_loops_version
        return self._facet_to_row_cache

    def copy(self):
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

    def get_edge(self, index: int) -> Edge:
        if index > 0:
            return self.edges[index]
        if index < 0:
            return self.edges[-index].reversed()
        raise InvalidEdgeIndexError(index)

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
        """Build or refresh the cached vertex ID order and index map."""
        if self.vertex_ids is None or len(self.vertex_ids) != len(self.vertices):
            ids = np.array(sorted(self.vertices.keys()), dtype=int)
            self.vertex_ids = ids
            self.vertex_index_to_row = {vid: i for i, vid in enumerate(ids)}
            self._vertex_ids_version += 1

    def positions_view(self) -> np.ndarray:
        """Return a dense ``(N_vertices, 3)`` array of vertex positions."""
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
                self._positions_cache = np.empty((n_verts, 3), dtype=float, order="C")
            for i, vid in enumerate(self.vertex_ids):
                self._positions_cache[i] = self.vertices[vid].position
            self._positions_cache_version = self._version
        return self._positions_cache

    def tilts_view(self) -> np.ndarray:
        """Return a dense ``(N_vertices, 3)`` array of vertex tilt vectors."""
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

    def tilts_in_view(self) -> np.ndarray:
        """Return a dense ``(N_vertices, 3)`` array of inner-leaflet tilt vectors."""
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

    def tilts_out_view(self) -> np.ndarray:
        """Return a dense ``(N_vertices, 3)`` array of outer-leaflet tilt vectors."""
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

    def set_tilts_from_array(self, tilts: np.ndarray) -> None:
        """Scatter a dense tilt array back onto vertex objects."""
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

    def set_tilts_in_from_array(self, tilts: np.ndarray) -> None:
        """Scatter a dense inner-leaflet tilt array back onto vertex objects."""
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
            object.__setattr__(vertex, "tilt_in", cache[row])
        if need_rebind:
            self._vertex_row_binding_version = self._vertex_ids_version

    def set_tilts_out_from_array(self, tilts: np.ndarray) -> None:
        """Scatter a dense outer-leaflet tilt array back onto vertex objects."""
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
            object.__setattr__(vertex, "tilt_out", cache[row])
        if need_rebind:
            self._vertex_row_binding_version = self._vertex_ids_version

    def build_facet_vertex_loops(self):
        """Precompute ordered vertex loops for all facets."""
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

    def triangle_row_cache(self) -> tuple[np.ndarray | None, list[int]]:
        """Return cached triangle row indices and facet IDs."""
        if (
            self._triangle_rows_cache is not None
            and self._triangle_rows_cache_version == self._facet_loops_version
        ):
            return self._triangle_rows_cache, self._triangle_row_facets
        if not getattr(self, "facet_vertex_loops", None):
            return None, []
        self.build_position_cache()
        tri_facets = triangle_facets_from_loops(self.facet_vertex_loops)
        if not tri_facets:
            self._triangle_rows_cache = None
            self._triangle_row_facets = []
            self._triangle_rows_cache_version = self._facet_loops_version
            return None, []
        tri_rows = np.ascontiguousarray(
            triangle_rows_from_loops(
                tri_facets=tri_facets,
                facet_vertex_loops=self.facet_vertex_loops,
                vertex_index_to_row=self.vertex_index_to_row,
            ),
            dtype=np.int32,
        )
        self._triangle_rows_cache = tri_rows
        self._triangle_row_facets = tri_facets
        self._triangle_rows_cache_version = self._facet_loops_version
        return tri_rows, tri_facets

    def triangle_areas(self, positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a cached array of triangle areas."""
        is_cached_pos = is_cached_positions(
            positions, getattr(self, "_positions_cache", None)
        )
        if triangle_areas_cache_valid(
            is_cached_pos=is_cached_pos,
            cached_version=self._cached_tri_areas_version,
            mesh_version=self._version,
            cached_areas=self._cached_tri_areas,
        ):
            return self._cached_tri_areas
        tri_rows, _ = self.triangle_row_cache()
        if tri_rows is None:
            return np.array([])
        if positions is None:
            positions = self.positions_view()
        normals, areas = triangle_normals_and_areas(positions, tri_rows)
        if is_cached_pos:
            store_triangle_area_normals_cache(self, areas=areas, normals=normals)
        return areas

    def triangle_normals(self, positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a cached array of unnormalized triangle normals."""
        is_cached_pos = is_cached_positions(
            positions, getattr(self, "_positions_cache", None)
        )
        if (
            triangle_areas_cache_valid(
                is_cached_pos=is_cached_pos,
                cached_version=self._cached_tri_areas_version,
                mesh_version=self._version,
                cached_areas=self._cached_tri_areas,
            )
            and self._cached_tri_normals is not None
        ):
            return self._cached_tri_normals
        self.triangle_areas(positions)
        if is_cached_pos:
            return self._cached_tri_normals
        if positions is None:
            positions = self.positions_view()
        tri_rows, _ = self.triangle_row_cache()
        return triangle_normals(positions, tri_rows)

    def barycentric_vertex_areas(
        self,
        positions: Optional[np.ndarray] = None,
        *,
        tri_rows: Optional[np.ndarray] = None,
        areas: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        cache: Optional[bool] = None,
    ) -> np.ndarray:
        """Return cached barycentric vertex areas from triangle areas."""
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
        if barycentric_cache_valid(
            use_cache=use_cache,
            cached_version=self._cached_barycentric_vertex_areas_version,
            mesh_version=self._version,
            cached_rows_version=self._cached_barycentric_vertex_areas_rows_version,
            loops_version=self._facet_loops_version,
            cached_values=self._cached_barycentric_vertex_areas,
            expected_size=n_verts,
        ):
            return self._cached_barycentric_vertex_areas
        if areas is None:
            n, _ = triangle_normals_and_areas(positions, tri_rows)
            n_norm = np.linalg.norm(n, axis=1)
            mask = n_norm >= 1e-12
            areas = 0.5 * n_norm[mask]
            tri_rows = tri_rows[mask]
        else:
            areas = np.asarray(areas, dtype=float)
            if mask is not None:
                tri_rows = tri_rows[mask]
                areas = areas[mask]
        vertex_areas = barycentric_vertex_areas_from_triangles(
            n_verts=n_verts, tri_rows=tri_rows, areas=areas
        )
        if use_cache:
            store_barycentric_vertex_areas_cache(self, vertex_areas=vertex_areas)
        return vertex_areas

    def vertex_normals(self, positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Return per-vertex unit normals."""
        is_cached_pos = is_cached_positions(
            positions, getattr(self, "_positions_cache", None)
        )
        if vertex_normals_cache_valid(
            is_cached_pos=is_cached_pos,
            cached_values=self._cached_vertex_normals,
            cached_version=self._cached_vertex_normals_version,
            mesh_version=self._version,
            cached_loops_version=self._cached_vertex_normals_loops_version,
            loops_version=self._facet_loops_version,
        ):
            return self._cached_vertex_normals
        tri_rows, _ = self.triangle_row_cache()
        self.build_position_cache()
        n_verts = len(self.vertex_ids)
        if tri_rows is None or len(tri_rows) == 0:
            normals = np.zeros((n_verts, 3), dtype=float)
            if is_cached_pos:
                store_vertex_normals_cache(self, normals=normals)
            return normals
        if positions is None:
            positions = self.positions_view()
        tri_normals = triangle_normals(positions, tri_rows)
        normals = vertex_unit_normals_from_triangles(
            n_verts=n_verts, tri_rows=tri_rows, tri_normals=tri_normals
        )
        if is_cached_pos:
            store_vertex_normals_cache(self, normals=normals)
        return normals

    def p1_triangle_shape_gradient_cache(
        self, positions: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return cached P1 triangle shape gradients."""
        tri_rows, _ = self.triangle_row_cache()
        if tri_rows is None or tri_rows.size == 0:
            zeros1 = np.zeros(0, dtype=float)
            zeros3 = np.zeros((0, 3), dtype=float)
            tri_empty = np.zeros((0, 3), dtype=np.int32)
            return zeros1, zeros3, zeros3, zeros3, tri_empty
        if positions is None:
            positions = self.positions_view()
        use_cache = self._geometry_cache_active(positions)
        if p1_triangle_cache_valid(
            use_cache=use_cache,
            cached_version=self._cached_p1_tri_grads_version,
            mesh_version=self._version,
            cached_rows_version=self._cached_p1_tri_grads_rows_version,
            loops_version=self._facet_loops_version,
            cached_area=self._cached_p1_tri_areas,
            cached_g0=self._cached_p1_tri_g0,
            cached_g1=self._cached_p1_tri_g1,
            cached_g2=self._cached_p1_tri_g2,
        ):
            return (
                self._cached_p1_tri_areas,
                self._cached_p1_tri_g0,
                self._cached_p1_tri_g1,
                self._cached_p1_tri_g2,
                tri_rows,
            )
        area, g0, g1, g2 = p1_triangle_shape_gradients(positions, tri_rows)
        if use_cache:
            store_p1_triangle_grad_cache(self, area=area, g0=g0, g1=g1, g2=g2)
        return area, g0, g1, g2, tri_rows

    def project_tilts_to_tangent(self) -> None:
        """Project all vertex tilt vectors into their local tangent planes."""
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
        """Initialize tangent tilt vectors from vertex options."""

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
                        f"Vertex {int(vid)} {field_key} must be a vector; got {raw!r}"
                    )
                n = normals[row]
                if np.linalg.norm(n) < 1e-12:
                    if len(raw) == 2:
                        vec = np.asarray([raw[0], raw[1], 0.0], dtype=float)
                    elif len(raw) == 3:
                        vec = np.asarray(raw, dtype=float)
                    else:
                        raise TypeError(f"Invalid length for {field_key}")
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
                    raise TypeError(f"Invalid length for {field_key}")
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

    def get_facets_of_vertex(self, v_id: int) -> List[Facet]:
        return [self.facets[fid] for fid in self.vertex_to_facets.get(v_id, [])]

    def get_facet_indices_of_vertex(self, v_id: int) -> set[int]:
        """Return the IDs of facets sharing this vertex."""
        return self.vertex_to_facets.get(v_id, set())

    def get_edges_of_vertex(self, v_id: int) -> List[Edge]:
        return [self.edges[eid] for eid in self.vertex_to_edges.get(v_id, [])]

    def get_facets_of_edge(self, e_id: int) -> List[Facet]:
        return [self.facets[fid] for fid in self.edge_to_facets.get(e_id, [])]

    def full_mesh_validate(self):
        """Perform full mesh validation."""
        self.validate_triangles()
        self.validate_edge_indices()
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

    # --- Backward compatibility wrappers ---
    def validate_body_orientation(self) -> bool:
        from .mesh_orientation import validate_body_orientation

        return validate_body_orientation(self)

    def _body_edge_uses(self, body: Body) -> dict[int, list[tuple[int, int]]]:
        from .mesh_orientation import _body_edge_uses

        return _body_edge_uses(self, body=body)

    def _body_is_closed(self, body: Body) -> bool:
        from .mesh_orientation import _body_is_closed

        return _body_is_closed(self, body=body)

    def validate_body_outwardness(self, volume_tol: float = 1e-12) -> bool:
        from .mesh_orientation import validate_body_outwardness

        return validate_body_outwardness(self, volume_tol=volume_tol)

    def orient_body_outward(self, body_index: int, volume_tol: float = 1e-12) -> int:
        from .mesh_orientation import orient_body_outward

        return orient_body_outward(self, body_index=body_index, volume_tol=volume_tol)

    def orient_body_facets(self, body_index: int) -> int:
        from .mesh_orientation import orient_body_facets

        return orient_body_facets(self, body_index=body_index)

    def validate_triangles(self):
        from .mesh_orientation import validate_triangles

        return validate_triangles(self)

    def validate_edge_indices(self):
        from .mesh_orientation import validate_edge_indices

        return validate_edge_indices(self)

    def compute_total_surface_area(self) -> float:
        from .mesh_computations import compute_total_surface_area

        return compute_total_surface_area(self)

    def compute_total_area_and_gradient(
        self,
        positions: np.ndarray | None = None,
        index_map: Dict[int, int] | None = None,
    ) -> tuple[float, Dict[int, np.ndarray]]:
        from .mesh_computations import compute_total_area_and_gradient

        return compute_total_area_and_gradient(
            self, positions=positions, index_map=index_map
        )

    def compute_total_volume(self) -> float:
        from .mesh_computations import compute_total_volume

        return compute_total_volume(self)

    def compute_surface_radius_of_gyration(
        self, facet_indices: Iterable[int] | None = None
    ) -> float:
        from .mesh_computations import compute_surface_radius_of_gyration

        return compute_surface_radius_of_gyration(self, facet_indices=facet_indices)
