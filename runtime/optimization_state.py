"""Explicit snapshots for optimization fields that require rollback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TiltStateSnapshot:
    """Immutable ownership record for one vertex- or leaflet-tilt state."""

    tilts: np.ndarray | None = None
    tilts_in: np.ndarray | None = None
    tilts_out: np.ndarray | None = None

    @property
    def uses_leaflet_tilts(self) -> bool:
        """Return whether this snapshot contains the two leaflet fields."""
        return self.tilts_in is not None


def capture_tilt_state(mesh: Any, *, uses_leaflet_tilts: bool) -> TiltStateSnapshot:
    """Copy the active tilt representation without mutating mesh versions."""
    if uses_leaflet_tilts:
        return TiltStateSnapshot(
            tilts_in=mesh.tilts_in_view().copy(order="F"),
            tilts_out=mesh.tilts_out_view().copy(order="F"),
        )
    return TiltStateSnapshot(tilts=mesh.tilts_view().copy(order="F"))


def restore_tilt_state(
    mesh: Any,
    snapshot: TiltStateSnapshot,
    *,
    set_leaflet_tilts: Callable[[np.ndarray, np.ndarray], None],
) -> None:
    """Restore a snapshot through the established mesh mutation paths."""
    if snapshot.uses_leaflet_tilts:
        assert snapshot.tilts_in is not None
        assert snapshot.tilts_out is not None
        set_leaflet_tilts(snapshot.tilts_in, snapshot.tilts_out)
        return
    if snapshot.tilts is None:
        raise ValueError("Tilt snapshot contains neither vertex nor leaflet fields")
    mesh.set_tilts_from_array(snapshot.tilts)


@dataclass(frozen=True)
class PositionStateSnapshot:
    """Dense position state paired with the vertex-row ordering that produced it."""

    vertex_ids: np.ndarray
    positions: np.ndarray


def capture_position_state(mesh: Any) -> PositionStateSnapshot:
    """Copy dense positions and row order without mutating mesh versions."""
    positions = mesh.positions_view()
    return PositionStateSnapshot(
        vertex_ids=np.array(mesh.vertex_ids, copy=True),
        positions=positions.copy(order="F"),
    )


def restore_position_state(mesh: Any, snapshot: PositionStateSnapshot) -> None:
    """Restore positions by row without changing version ownership."""
    current_ids = np.asarray(mesh.vertex_ids)
    if not np.array_equal(current_ids, snapshot.vertex_ids):
        raise ValueError("Position snapshot vertex ordering no longer matches mesh")
    for row, vertex_id in enumerate(snapshot.vertex_ids):
        mesh.vertices[int(vertex_id)].position[:] = snapshot.positions[row]


def commit_position_rows(
    mesh: Any,
    *,
    vertex_ids: np.ndarray,
    rows: np.ndarray,
    positions: np.ndarray,
) -> None:
    """Commit selected dense rows without changing version ownership."""
    for row in rows:
        mesh.vertices[int(vertex_ids[row])].position[:] = positions[row]
