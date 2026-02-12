"""Evaluation context scaffolding for externalized geometry/energy caches."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geometry.entities import Mesh


@dataclass
class GeometryCache:
    """Version-bound cache storage for geometry-derived arrays."""

    _mesh_version: int = -1
    _vertex_ids_version: int = -1
    _topology_version: int = -1
    _store: dict[str, Any] = field(default_factory=dict)

    def is_valid_for(self, mesh: Mesh) -> bool:
        """Return True when this cache matches the mesh version tuple."""
        return (
            self._mesh_version == int(mesh._version)
            and self._vertex_ids_version == int(mesh._vertex_ids_version)
            and self._topology_version == int(mesh._topology_version)
        )

    def bind(self, mesh: Mesh) -> None:
        """Bind cache metadata to the current mesh version tuple."""
        self._mesh_version = int(mesh._version)
        self._vertex_ids_version = int(mesh._vertex_ids_version)
        self._topology_version = int(mesh._topology_version)

    def clear(self) -> None:
        """Clear all cached geometry values."""
        self._store.clear()

    def ensure_for_mesh(self, mesh: Mesh) -> None:
        """Invalidate and rebind when the mesh version tuple changes."""
        if not self.is_valid_for(mesh):
            self.clear()
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
        self.ensure_for_mesh(mesh)
        positions = self.get("positions")
        index_map = self.get("index_map")
        if positions is None or index_map is None:
            positions = mesh.positions_view()
            index_map = mesh.vertex_index_to_row
            self.set("positions", positions)
            self.set("index_map", index_map)
        return positions, index_map

    def triangle_rows(self, mesh: Mesh) -> tuple[np.ndarray | None, list[int]]:
        """Return context-cached triangle rows and facet IDs.

        This computes rows from facet vertex loops and vertex-id index maps
        without mutating mesh-owned triangle-row cache fields.
        """
        self.ensure_for_mesh(mesh)
        tri_rows = self.get("triangle_rows")
        tri_facets = self.get("triangle_row_facets")
        if tri_facets is not None and tri_rows is not None:
            return tri_rows, tri_facets
        if tri_facets is not None and tri_rows is None:
            return None, tri_facets

        if not getattr(mesh, "facet_vertex_loops", None):
            self.set("triangle_rows", None)
            self.set("triangle_row_facets", [])
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
            return None, []

        tri_rows = np.empty((len(tri_facets), 3), dtype=np.int32, order="F")
        for idx, fid in enumerate(tri_facets):
            loop = mesh.facet_vertex_loops[fid]
            tri_rows[idx, 0] = mesh.vertex_index_to_row[int(loop[0])]
            tri_rows[idx, 1] = mesh.vertex_index_to_row[int(loop[1])]
            tri_rows[idx, 2] = mesh.vertex_index_to_row[int(loop[2])]

        self.set("triangle_rows", tri_rows)
        self.set("triangle_row_facets", tri_facets)
        return tri_rows, tri_facets


@dataclass
class EnergyContext:
    """Context object that owns geometry caches and reusable scratch buffers."""

    geometry: GeometryCache = field(default_factory=GeometryCache)
    scratch: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def ensure_for_mesh(self, mesh: Mesh) -> None:
        """Ensure cache validity for the current mesh state."""
        if not self.geometry.is_valid_for(mesh):
            self.geometry.ensure_for_mesh(mesh)
            self.scratch.clear()

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
