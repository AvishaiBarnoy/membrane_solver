"""Shared geometric and diagnostic utility logic for tilt energy modules."""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh, _fast_cross
from modules.constraints.local_interface_shells import build_local_interface_shell_data
from modules.energy.tilt_params import (
    _resolve_exclude_shared_rim_outer_rows,
)


def _triangle_geometry(
    positions: np.ndarray,
    tri_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return vertex positions and unnormalized normals for the given triangles."""
    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]
    n = _fast_cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm >= 1e-12
    return v0, v1, v2, n, n_norm, mask


def _resolve_shared_rim_outer_row_energy_weight(
    param_resolver, leaflet: str
) -> float | None:
    raw = param_resolver.get(None, f"tilt_{leaflet}_shared_rim_outer_row_energy_weight")
    if raw is None:
        return None
    weight = float(raw)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(
            f"tilt_{leaflet}_shared_rim_outer_row_energy_weight must be a finite nonnegative float."
        )
    return weight


def _shared_rim_outer_shell_rows(mesh: Mesh, leaflet: str) -> np.ndarray:
    """Return rows in the first outer shell used by shared-rim relief controls."""
    cache_attr = f"_tilt_{leaflet}_shared_rim_outer_shell_rows_cache"
    cache = getattr(mesh, cache_attr, None)
    if cache is None:
        cache = {}
        setattr(mesh, cache_attr, cache)

    cache_key = (int(mesh._version), int(mesh._vertex_ids_version))
    rows = cache.get(cache_key)
    if rows is not None:
        return rows

    tagged_rows: list[int] = []
    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if str(opts.get("rim_slope_match_group") or "") == "outer":
            tagged_rows.append(int(row))
    if tagged_rows:
        rows = np.asarray(tagged_rows, dtype=int)
    else:
        mesh.build_position_cache()
        try:
            shell_data = build_local_interface_shell_data(
                mesh, positions=mesh.positions_view()
            )
            rows = np.asarray(shell_data.outer_rows, dtype=int)
        except AssertionError:
            rows = np.zeros(0, dtype=int)

    cache.clear()
    cache[cache_key] = rows
    return rows


def _shared_rim_active_row_weights(
    mesh: Mesh, param_resolver, leaflet: str
) -> np.ndarray | None:
    """Return per-row tilt weights for the shared-rim leaflet surrogate."""
    exclude_outer_rows = _resolve_exclude_shared_rim_outer_rows(param_resolver, leaflet)

    # Inner leaflet has some extra features for diagnostic weighting
    exclude_rim_rows = False
    outer_row_energy_weight = None
    if leaflet == "in":
        from modules.energy.tilt_utils import _resolve_exclude_shared_rim_rows

        exclude_rim_rows = _resolve_exclude_shared_rim_rows(param_resolver)
        outer_row_energy_weight = _resolve_shared_rim_outer_row_energy_weight(
            param_resolver, "in"
        )

    mode = str(param_resolver.get(None, "rim_slope_match_mode") or "").strip().lower()

    has_explicit_override = (
        exclude_rim_rows or exclude_outer_rows or outer_row_energy_weight is not None
    )
    if mode != "shared_rim_staggered_v1" and not has_explicit_override:
        return None
    if not has_explicit_override:
        return None

    cache_attr = f"_tilt_{leaflet}_shared_rim_active_row_weights_cache"
    cache = getattr(mesh, cache_attr, None)
    if cache is None:
        cache = {}
        setattr(mesh, cache_attr, cache)

    cache_key = (
        int(mesh._version),
        int(mesh._vertex_ids_version),
        mode,
        bool(exclude_rim_rows),
        bool(exclude_outer_rows),
        None if outer_row_energy_weight is None else float(outer_row_energy_weight),
    )
    weights = cache.get(cache_key)
    if weights is not None and weights.shape == (len(mesh.vertex_ids),):
        return weights

    weights = np.ones(len(mesh.vertex_ids), dtype=float)
    outer_shell_rows = _shared_rim_outer_shell_rows(mesh, leaflet)
    outer_shell_row_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    if outer_shell_rows.size:
        outer_shell_row_mask[outer_shell_rows] = True

    outer_row_scale = None
    if outer_row_energy_weight is not None:
        outer_row_scale = float(np.sqrt(outer_row_energy_weight))

    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        group = str(opts.get("rim_slope_match_group") or "")

        if exclude_rim_rows and group == "rim":
            weights[row] = 0.0
            continue

        if group == "outer" or outer_shell_row_mask[row]:
            if exclude_outer_rows:
                weights[row] = 0.0
            elif outer_row_scale is not None:
                weights[row] = outer_row_scale
        elif leaflet == "out" and group == "outer":
            # Legacy tilt_out logic was simpler
            weights[row] = 0.0 if exclude_outer_rows else 1.0

    cache.clear()
    cache[cache_key] = weights
    return weights


def _explicit_trace_layer_active_row_weights(
    mesh: Mesh, param_resolver, leaflet: str
) -> np.ndarray | None:
    """Return interface-shell row weights for the explicit trace layer rows."""
    mode = str(param_resolver.get(None, "rim_slope_match_mode") or "").strip().lower()
    trace_radius = param_resolver.get(None, "parity_trace_layer_radius")
    lane = str(param_resolver.get(None, "theory_parity_lane") or "").strip()
    if mode != "physical_edge_staggered_v1" or trace_radius is None or not lane:
        return None

    cache_attr = f"_tilt_{leaflet}_trace_layer_active_row_weights_cache"
    cache = getattr(mesh, cache_attr, None)
    if cache is None:
        cache = {}
        setattr(mesh, cache_attr, cache)

    cache_key = (
        int(mesh._version),
        int(mesh._vertex_ids_version),
        float(trace_radius),
        str(lane),
    )
    weights = cache.get(cache_key)
    if weights is not None and weights.shape == (len(mesh.vertex_ids),):
        return weights

    mesh.build_position_cache()
    try:
        shell_data = build_local_interface_shell_data(
            mesh, positions=mesh.positions_view()
        )
    except AssertionError:
        return None

    denom = float(shell_data.outer_radius) - float(shell_data.disk_radius)
    numer = float(shell_data.rim_radius) - float(shell_data.disk_radius)
    if denom <= 1.0e-12:
        return None
    shell_fraction = min(1.0, max(0.0, numer / denom))
    shell_scale = float(np.sqrt(shell_fraction))

    weights = np.ones(len(mesh.vertex_ids), dtype=float)
    weights[np.asarray(shell_data.rim_rows, dtype=int)] = shell_scale
    cache.clear()
    cache[cache_key] = weights
    return weights


def _resolve_exclude_shared_rim_rows(param_resolver) -> bool:
    raw = param_resolver.get(None, "tilt_in_exclude_shared_rim_rows")
    if raw is None:
        raw = param_resolver.get(None, "tilt_exclude_shared_rim_rows_in")
    if raw is None:
        return False
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _resolve_shared_rim_outer_shell_mass_mode(
    param_resolver, leaflet: str
) -> str | None:
    raw = param_resolver.get(None, f"tilt_{leaflet}_shared_rim_outer_shell_mass_mode")
    if raw is None:
        return None
    txt = str(raw).strip().lower()
    if txt not in {"lumped", "consistent"}:
        raise ValueError(
            f"tilt_{leaflet}_shared_rim_outer_shell_mass_mode must be 'lumped' or 'consistent'."
        )
    return txt


def _active_row_weights(mesh: Mesh, param_resolver, leaflet: str) -> np.ndarray | None:
    """Return the combined per-row weights for diagnostic shell controls."""
    shared = _shared_rim_active_row_weights(mesh, param_resolver, leaflet)
    trace = _explicit_trace_layer_active_row_weights(mesh, param_resolver, leaflet)
    if shared is None:
        return trace
    if trace is None:
        return shared
    return shared * trace


def _shared_rim_outer_support_triangle_mask(
    mesh: Mesh, tri_rows: np.ndarray, leaflet: str
) -> np.ndarray | None:
    """Return a mask for triangles spanning only the first outer support shell."""
    if tri_rows.size == 0:
        return None

    cache_attr = f"_tilt_{leaflet}_shared_rim_outer_support_tri_cache"
    cache = getattr(mesh, cache_attr, None)
    if cache is None:
        cache = {}
        setattr(mesh, cache_attr, cache)

    cache_key = (int(mesh._version), int(mesh._vertex_ids_version), tri_rows.shape[0])
    mask = cache.get(cache_key)
    if mask is not None and mask.shape == (len(tri_rows),):
        return mask

    outer_row_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    rim_row_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    disk_row_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    outer_shell_rows = _shared_rim_outer_shell_rows(mesh, leaflet)
    if outer_shell_rows.size:
        outer_row_mask[outer_shell_rows] = True
    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if str(opts.get("rim_slope_match_group") or "") == "rim":
            rim_row_mask[row] = True
        if opts.get("preset") == "disk":
            disk_row_mask[row] = True

    has_outer = np.any(outer_row_mask[tri_rows], axis=1)
    has_rim = np.any(rim_row_mask[tri_rows], axis=1)
    has_disk = np.any(disk_row_mask[tri_rows], axis=1)
    mask = has_outer & (~has_rim) & (~has_disk)
    cache.clear()
    cache[cache_key] = mask
    return mask
