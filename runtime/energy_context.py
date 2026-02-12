"""Evaluation context scaffolding for externalized geometry/energy caches."""

from dataclasses import dataclass, field
from typing import Any

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
