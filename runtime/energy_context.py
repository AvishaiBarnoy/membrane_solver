"""Evaluation context scaffolding for externalized geometry/energy caches."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geometry.entities import Mesh, _fast_cross


@dataclass
class GeometryCache:
    """Version-bound cache storage for geometry-derived arrays."""

    _bound_mesh_version: int = -1
    _bound_vertex_ids_version: int = -1
    _bound_topology_version: int = -1
    _soa_mesh_version: int = -1
    _soa_vertex_ids_version: int = -1
    _triangle_rows_loops_version: int = -1
    _triangle_rows_vertex_ids_version: int = -1
    _triangle_geom_mesh_version: int = -1
    _triangle_geom_loops_version: int = -1
    _triangle_geom_vertex_ids_version: int = -1
    _bary_mesh_version: int = -1
    _bary_loops_version: int = -1
    _bary_vertex_ids_version: int = -1
    _p1_mesh_version: int = -1
    _p1_loops_version: int = -1
    _p1_vertex_ids_version: int = -1
    _store: dict[str, Any] = field(default_factory=dict)

    def is_valid_for(self, mesh: Mesh) -> bool:
        """Return True when this cache matches the mesh version tuple."""
        return (
            self._bound_mesh_version == int(mesh._version)
            and self._bound_vertex_ids_version == int(mesh._vertex_ids_version)
            and self._bound_topology_version == int(mesh._topology_version)
        )

    def bind(self, mesh: Mesh) -> None:
        """Bind cache metadata to the current mesh version tuple."""
        self._bound_mesh_version = int(mesh._version)
        self._bound_vertex_ids_version = int(mesh._vertex_ids_version)
        self._bound_topology_version = int(mesh._topology_version)

    def clear(self) -> None:
        """Clear all cached geometry values."""
        self._store.clear()

    def ensure_for_mesh(self, mesh: Mesh) -> None:
        """Rebind current mesh version tuple without clearing all caches."""
        self.bind(mesh)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a cached value."""
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store a cached value."""
        self._store[key] = value

    def soa_views(self, mesh: Mesh) -> tuple[np.ndarray, dict[int, int]]:
        """Return cached positions/index-map views for energy assembly.

        This uses mesh-provided SoA builders as the fallback source and stores
        the resulting views inside the context cache.
        """
        self.bind(mesh)
        positions = self.get("positions")
        index_map = self.get("index_map")
        if (
            positions is None
            or index_map is None
            or self._soa_mesh_version != int(mesh._version)
            or self._soa_vertex_ids_version != int(mesh._vertex_ids_version)
        ):
            positions = mesh.positions_view()
            index_map = mesh.vertex_index_to_row
            self.set("positions", positions)
            self.set("index_map", index_map)
            self._soa_mesh_version = int(mesh._version)
            self._soa_vertex_ids_version = int(mesh._vertex_ids_version)
        return positions, index_map

    def triangle_rows(self, mesh: Mesh) -> tuple[np.ndarray | None, list[int]]:
        """Return context-cached triangle rows and facet IDs.

        This computes rows from facet vertex loops and vertex-id index maps
        without mutating mesh-owned triangle-row cache fields.
        """
        self.bind(mesh)
        tri_rows = self.get("triangle_rows")
        tri_facets = self.get("triangle_row_facets")
        if (
            tri_facets is not None
            and tri_rows is not None
            and self._triangle_rows_loops_version == int(mesh._facet_loops_version)
            and self._triangle_rows_vertex_ids_version == int(mesh._vertex_ids_version)
        ):
            return tri_rows, tri_facets
        if (
            tri_facets is not None
            and tri_rows is None
            and self._triangle_rows_loops_version == int(mesh._facet_loops_version)
            and self._triangle_rows_vertex_ids_version == int(mesh._vertex_ids_version)
        ):
            return None, tri_facets

        if not getattr(mesh, "facet_vertex_loops", None):
            self.set("triangle_rows", None)
            self.set("triangle_row_facets", [])
            self._triangle_rows_loops_version = int(mesh._facet_loops_version)
            self._triangle_rows_vertex_ids_version = int(mesh._vertex_ids_version)
            return None, []

        mesh.build_position_cache()
        tri_facets = []
        for fid in sorted(mesh.facet_vertex_loops):
            loop = mesh.facet_vertex_loops[fid]
            if len(loop) == 3:
                tri_facets.append(fid)

        if not tri_facets:
            self.set("triangle_rows", None)
            self.set("triangle_row_facets", [])
            self._triangle_rows_loops_version = int(mesh._facet_loops_version)
            self._triangle_rows_vertex_ids_version = int(mesh._vertex_ids_version)
            return None, []

        tri_rows = np.empty((len(tri_facets), 3), dtype=np.int32, order="F")
        for idx, fid in enumerate(tri_facets):
            loop = mesh.facet_vertex_loops[fid]
            tri_rows[idx, 0] = mesh.vertex_index_to_row[int(loop[0])]
            tri_rows[idx, 1] = mesh.vertex_index_to_row[int(loop[1])]
            tri_rows[idx, 2] = mesh.vertex_index_to_row[int(loop[2])]

        self.set("triangle_rows", tri_rows)
        self.set("triangle_row_facets", tri_facets)
        self._triangle_rows_loops_version = int(mesh._facet_loops_version)
        self._triangle_rows_vertex_ids_version = int(mesh._vertex_ids_version)
        return tri_rows, tri_facets

    def triangle_areas_normals(
        self, mesh: Mesh, positions: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return context-cached triangle areas and unnormalized normals."""
        self.bind(mesh)
        tri_rows, _ = self.triangle_rows(mesh)
        if tri_rows is None or tri_rows.size == 0:
            return np.array([], dtype=float), np.zeros((0, 3), dtype=float)

        if positions is None:
            positions, _ = self.soa_views(mesh)

        if (
            self._triangle_geom_mesh_version == int(mesh._version)
            and self._triangle_geom_loops_version == int(mesh._facet_loops_version)
            and self._triangle_geom_vertex_ids_version == int(mesh._vertex_ids_version)
            and self.get("triangle_areas") is not None
            and self.get("triangle_normals") is not None
        ):
            return self.get("triangle_areas"), self.get("triangle_normals")

        v0 = positions[tri_rows[:, 0]]
        v1 = positions[tri_rows[:, 1]]
        v2 = positions[tri_rows[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(normals, axis=1)

        self.set("triangle_areas", areas)
        self.set("triangle_normals", normals)
        self._triangle_geom_mesh_version = int(mesh._version)
        self._triangle_geom_loops_version = int(mesh._facet_loops_version)
        self._triangle_geom_vertex_ids_version = int(mesh._vertex_ids_version)
        return areas, normals

    def triangle_areas(
        self, mesh: Mesh, positions: np.ndarray | None = None
    ) -> np.ndarray:
        """Return context-cached triangle areas."""
        areas, _ = self.triangle_areas_normals(mesh, positions)
        return areas

    def barycentric_vertex_areas(
        self, mesh: Mesh, positions: np.ndarray | None = None
    ) -> np.ndarray:
        """Return context-cached barycentric per-vertex areas."""
        self.bind(mesh)
        n_verts = len(mesh.vertex_ids)
        if n_verts == 0:
            return np.array([], dtype=float)

        tri_rows, _ = self.triangle_rows(mesh)
        if tri_rows is None or tri_rows.size == 0:
            return np.zeros(n_verts, dtype=float)

        if (
            self._bary_mesh_version == int(mesh._version)
            and self._bary_loops_version == int(mesh._facet_loops_version)
            and self._bary_vertex_ids_version == int(mesh._vertex_ids_version)
            and self.get("barycentric_vertex_areas") is not None
        ):
            return self.get("barycentric_vertex_areas")

        areas = self.triangle_areas(mesh, positions)
        vertex_areas = np.zeros(n_verts, dtype=float)
        if areas.size:
            area_thirds = areas / 3.0
            np.add.at(vertex_areas, tri_rows[:, 0], area_thirds)
            np.add.at(vertex_areas, tri_rows[:, 1], area_thirds)
            np.add.at(vertex_areas, tri_rows[:, 2], area_thirds)

        self.set("barycentric_vertex_areas", vertex_areas)
        self._bary_mesh_version = int(mesh._version)
        self._bary_loops_version = int(mesh._facet_loops_version)
        self._bary_vertex_ids_version = int(mesh._vertex_ids_version)
        return vertex_areas

    def p1_triangle_shape_gradients(
        self, mesh: Mesh, positions: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return context-cached P1 triangle shape gradients."""
        self.bind(mesh)
        tri_rows, _ = self.triangle_rows(mesh)
        if tri_rows is None or tri_rows.size == 0:
            zeros1 = np.zeros(0, dtype=float)
            zeros3 = np.zeros((0, 3), dtype=float)
            tri_empty = np.zeros((0, 3), dtype=np.int32)
            return zeros1, zeros3, zeros3, zeros3, tri_empty

        if positions is None:
            positions, _ = self.soa_views(mesh)

        if (
            self._p1_mesh_version == int(mesh._version)
            and self._p1_loops_version == int(mesh._facet_loops_version)
            and self._p1_vertex_ids_version == int(mesh._vertex_ids_version)
            and self.get("p1_area") is not None
            and self.get("p1_g0") is not None
            and self.get("p1_g1") is not None
            and self.get("p1_g2") is not None
        ):
            return (
                self.get("p1_area"),
                self.get("p1_g0"),
                self.get("p1_g1"),
                self.get("p1_g2"),
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

        self.set("p1_area", area)
        self.set("p1_g0", g0)
        self.set("p1_g1", g1)
        self.set("p1_g2", g2)
        self._p1_mesh_version = int(mesh._version)
        self._p1_loops_version = int(mesh._facet_loops_version)
        self._p1_vertex_ids_version = int(mesh._vertex_ids_version)
        return area, g0, g1, g2, tri_rows


@dataclass
class EnergyContext:
    """Context object that owns geometry caches and reusable scratch buffers."""

    geometry: GeometryCache = field(default_factory=GeometryCache)
    scratch: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def ensure_for_mesh(self, mesh: Mesh) -> None:
        """Ensure cache validity for the current mesh state."""
        self.geometry.ensure_for_mesh(mesh)

    def scratch_array(
        self,
        key: str,
        *,
        shape: tuple[int, ...],
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        """Return a reusable zeroed scratch array in this context."""
        arr = self.scratch.get(key)
        if arr is None or arr.shape != shape or arr.dtype != np.dtype(dtype):
            arr = np.zeros(shape, dtype=dtype)
            self.scratch[key] = arr
        else:
            arr.fill(0.0)
        return arr
