"""Helper utilities extracted from ``runtime.minimizer``.

These helpers keep behavior-preserving logic centralized while letting
``Minimizer`` stay focused on optimization flow.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def capture_diagnostic_state(
    mesh, global_params, *, uses_leaflet_tilts: bool
) -> dict[str, Any]:
    """Capture mutable state that debug diagnostics must not persist."""
    snapshot: dict[str, Any] = {
        "global_params": dict(global_params.to_dict()),
        "tilts": None,
        "tilts_in": None,
        "tilts_out": None,
    }
    if uses_leaflet_tilts:
        snapshot["tilts_in"] = mesh.tilts_in_view().copy(order="F")
        snapshot["tilts_out"] = mesh.tilts_out_view().copy(order="F")
    else:
        snapshot["tilts"] = mesh.tilts_view().copy(order="F")
    return snapshot


def restore_diagnostic_state(mesh, global_params, snapshot: dict[str, Any]) -> None:
    """Restore mutable state after debug diagnostics."""
    if snapshot.get("tilts_in") is not None:
        if not np.array_equal(mesh.tilts_in_view(), snapshot["tilts_in"]):
            mesh.set_tilts_in_from_array(snapshot["tilts_in"])
        if not np.array_equal(mesh.tilts_out_view(), snapshot["tilts_out"]):
            mesh.set_tilts_out_from_array(snapshot["tilts_out"])
    elif snapshot.get("tilts") is not None:
        if not np.array_equal(mesh.tilts_view(), snapshot["tilts"]):
            mesh.set_tilts_from_array(snapshot["tilts"])

    if snapshot.get("global_params") != global_params.to_dict():
        global_params._params = dict(snapshot["global_params"])


def get_cached_tilt_fixed_mask(
    *,
    mesh,
    flag_attr: str,
    cached_mask: np.ndarray | None,
    cached_flags_version: int,
    cached_vertex_version: int,
) -> tuple[np.ndarray, int, int]:
    """Return cached tilt fixed mask (or recompute when mesh versions changed)."""
    flags_version = mesh._tilt_fixed_flags_version
    vertex_version = mesh._vertex_ids_version
    if (
        cached_mask is not None
        and cached_flags_version == flags_version
        and cached_vertex_version == vertex_version
        and len(cached_mask) == len(mesh.vertices)
    ):
        return cached_mask, flags_version, vertex_version

    mask = np.array(
        [
            bool(getattr(mesh.vertices[int(vid)], flag_attr, False))
            for vid in mesh.vertex_ids
        ],
        dtype=bool,
    )
    return mask, flags_version, vertex_version
