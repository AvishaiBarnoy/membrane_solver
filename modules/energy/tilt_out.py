"""Outer-leaflet tilt magnitude energy module.

This module models a per-vertex tilt penalty for the outer leaflet:

    E = 1/2 * k_t * sum_v (|t_out,v|^2 * A_v)

where ``t_out,v`` is a 3D tangent tilt vector stored on each vertex and ``A_v``
is a barycentric area weight.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross
from modules.constraints.local_interface_shells import build_local_interface_shell_data
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)

USES_TILT_LEAFLETS = True


def _resolve_exclude_shared_rim_outer_rows(param_resolver) -> bool:
    raw = param_resolver.get(None, "tilt_out_exclude_shared_rim_outer_rows")
    if raw is None:
        raw = param_resolver.get(None, "tilt_out_exclude_shared_rim_rows")
    if raw is None:
        raw = param_resolver.get(None, "tilt_exclude_shared_rim_rows_out")
    if raw is None:
        return False
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _shared_rim_active_row_weights(mesh: Mesh, param_resolver) -> np.ndarray | None:
    """Return per-row tilt weights for the shared-rim outer-leaflet surrogate."""
    if not _resolve_exclude_shared_rim_outer_rows(param_resolver):
        return None
    mode = str(param_resolver.get(None, "rim_slope_match_mode") or "").strip().lower()
    if mode != "shared_rim_staggered_v1":
        return None

    cache = getattr(mesh, "_tilt_out_shared_rim_active_row_weights_cache", None)
    if cache is None:
        cache = {}
        setattr(mesh, "_tilt_out_shared_rim_active_row_weights_cache", cache)

    cache_key = (
        int(mesh._version),
        int(mesh._vertex_ids_version),
        mode,
    )
    weights = cache.get(cache_key)
    if weights is not None and weights.shape == (len(mesh.vertex_ids),):
        return weights

    weights = np.ones(len(mesh.vertex_ids), dtype=float)
    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if str(opts.get("rim_slope_match_group") or "") == "outer":
            weights[row] = 0.0
    cache.clear()
    cache[cache_key] = weights
    return weights


def _explicit_trace_layer_active_row_weights(
    mesh: Mesh, param_resolver
) -> np.ndarray | None:
    """Return interface-shell row weights for the explicit trace layer rows.

    The inserted `R+epsilon` shell is an interface support layer, not a full
    independent annulus. Its tilt mass should therefore scale with the radial
    thickness it represents between the disk boundary and the first real
    free-side shell.
    """
    mode = str(param_resolver.get(None, "rim_slope_match_mode") or "").strip().lower()
    trace_radius = param_resolver.get(None, "parity_trace_layer_radius")
    lane = str(param_resolver.get(None, "theory_parity_lane") or "").strip()
    if mode != "physical_edge_staggered_v1" or trace_radius is None or not lane:
        return None

    cache = getattr(mesh, "_tilt_out_trace_layer_active_row_weights_cache", None)
    if cache is None:
        cache = {}
        setattr(mesh, "_tilt_out_trace_layer_active_row_weights_cache", cache)

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


def _active_row_weights(mesh: Mesh, param_resolver) -> np.ndarray | None:
    """Return the combined per-row weights for shared-rim and ghost-shell controls."""
    shared = _shared_rim_active_row_weights(mesh, param_resolver)
    trace = _explicit_trace_layer_active_row_weights(mesh, param_resolver)
    if shared is None:
        return trace
    if trace is None:
        return shared
    return shared * trace


def _resolve_tilt_modulus(param_resolver) -> float:
    k = param_resolver.get(None, "tilt_modulus_out")
    if k is None:
        k = param_resolver.get(None, "tilt_modolus_out")
    return float(k or 0.0)


def _resolve_tilt_mass_mode(param_resolver) -> str:
    mode = param_resolver.get(None, "tilt_mass_mode_out")
    if mode is None:
        mode = param_resolver.get(None, "tilt_mass_mode")
    txt = str(mode or "lumped").strip().lower()
    if txt not in {"lumped", "consistent"}:
        raise ValueError("tilt_mass_mode_out must be 'lumped' or 'consistent'.")
    return txt


def _triangle_geometry(
    positions: np.ndarray,
    tri_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-triangle geometric arrays for area and area gradient."""
    tri_pos = positions[tri_rows]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]

    n = _fast_cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm >= 1e-12
    return v0, v1, v2, n, n_norm, mask


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver
) -> Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return tilt energy and gradients for the outer leaflet."""
    k_tilt = _resolve_tilt_modulus(param_resolver)
    shape_grad = {i: np.zeros(3, dtype=float) for i in mesh.vertices}
    tilt_grad = {i: np.zeros(3, dtype=float) for i in mesh.vertices}
    if k_tilt == 0.0:
        return 0.0, shape_grad, tilt_grad

    mesh.build_position_cache()
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0, shape_grad, tilt_grad

    tilts_out = mesh.tilts_out_view()
    active_row_weights = _active_row_weights(mesh, param_resolver)
    if active_row_weights is not None:
        tilts_eff = tilts_out * active_row_weights[:, None]
    else:
        tilts_eff = tilts_out
    mode = _resolve_tilt_mass_mode(param_resolver)
    v0, v1, v2, n, n_norm, mask = _triangle_geometry(positions, tri_rows)
    if not np.any(mask):
        return 0.0, shape_grad, tilt_grad

    areas = 0.5 * n_norm[mask]
    tri_rows_m = tri_rows[mask]
    t0 = tilts_eff[tri_rows_m[:, 0]]
    t1 = tilts_eff[tri_rows_m[:, 1]]
    t2 = tilts_eff[tri_rows_m[:, 2]]

    if mode == "lumped":
        tri_tilt_sq_sum = (
            np.einsum("ij,ij->i", t0, t0)
            + np.einsum("ij,ij->i", t1, t1)
            + np.einsum("ij,ij->i", t2, t2)
        )
        coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)
        energy = float(np.dot(coeff, areas))

        vertex_areas = np.zeros(len(mesh.vertex_ids), dtype=float)
        area_thirds = areas / 3.0
        np.add.at(vertex_areas, tri_rows_m[:, 0], area_thirds)
        np.add.at(vertex_areas, tri_rows_m[:, 1], area_thirds)
        np.add.at(vertex_areas, tri_rows_m[:, 2], area_thirds)
        tilt_grad_arr = k_tilt * tilts_eff * vertex_areas[:, None]
    else:
        s = (
            np.einsum("ij,ij->i", t0, t0)
            + np.einsum("ij,ij->i", t1, t1)
            + np.einsum("ij,ij->i", t2, t2)
            + np.einsum("ij,ij->i", t0, t1)
            + np.einsum("ij,ij->i", t1, t2)
            + np.einsum("ij,ij->i", t2, t0)
        )
        coeff = (k_tilt / 12.0) * s
        energy = float(np.dot(coeff, areas))

        tilt_grad_arr = np.zeros_like(positions)
        tri_factor = (k_tilt * areas / 12.0)[:, None]
        np.add.at(tilt_grad_arr, tri_rows_m[:, 0], tri_factor * ((2.0 * t0) + t1 + t2))
        np.add.at(tilt_grad_arr, tri_rows_m[:, 1], tri_factor * ((2.0 * t1) + t2 + t0))
        np.add.at(tilt_grad_arr, tri_rows_m[:, 2], tri_factor * ((2.0 * t2) + t0 + t1))

    n_hat = n[mask] / n_norm[mask][:, None]
    g0 = 0.5 * _fast_cross(n_hat, (v2[mask] - v1[mask]))
    g1 = 0.5 * _fast_cross(n_hat, (v0[mask] - v2[mask]))
    g2 = 0.5 * _fast_cross(n_hat, (v1[mask] - v0[mask]))

    grad_arr = np.zeros_like(positions)
    c = coeff[:, None]
    np.add.at(grad_arr, tri_rows_m[:, 0], c * g0)
    np.add.at(grad_arr, tri_rows_m[:, 1], c * g1)
    np.add.at(grad_arr, tri_rows_m[:, 2], c * g2)
    for row, vid in enumerate(mesh.vertex_ids):
        vidx = int(vid)
        shape_grad[vidx] = grad_arr[row]
        tilt_grad[vidx] = tilt_grad_arr[row]

    return energy, shape_grad, tilt_grad


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray | None,
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
    tilt_in_grad_arr: np.ndarray | None = None,
    tilt_out_grad_arr: np.ndarray | None = None,
) -> float:
    """Dense-array outer-leaflet tilt energy accumulation."""
    _ = index_map, tilts_in, tilt_in_grad_arr
    k_tilt = _resolve_tilt_modulus(param_resolver)
    if k_tilt == 0.0:
        return 0.0
    mode = _resolve_tilt_mass_mode(param_resolver)

    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0

    mesh.build_position_cache()
    absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet="out")
    tri_keep = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_mask
    )
    if tri_keep.size and not np.any(tri_keep):
        return 0.0

    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()
    else:
        tilts_out = np.asarray(tilts_out, dtype=float)
        if tilts_out.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")

    active_row_weights = _active_row_weights(mesh, param_resolver)
    if active_row_weights is not None:
        tilts_eff = tilts_out * active_row_weights[:, None]
    else:
        tilts_eff = tilts_out

    tri_rows_eff = tri_rows[tri_keep] if tri_keep.size else tri_rows
    v0, v1, v2, n, n_norm, mask = _triangle_geometry(positions, tri_rows_eff)
    if not np.any(mask):
        return 0.0

    areas = 0.5 * n_norm[mask]
    tri_rows_m = tri_rows_eff[mask]
    t0 = tilts_eff[tri_rows_m[:, 0]]
    t1 = tilts_eff[tri_rows_m[:, 1]]
    t2 = tilts_eff[tri_rows_m[:, 2]]

    if mode == "lumped":
        tri_tilt_sq_sum = (
            np.einsum("ij,ij->i", t0, t0)
            + np.einsum("ij,ij->i", t1, t1)
            + np.einsum("ij,ij->i", t2, t2)
        )
        coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)
        energy = float(np.dot(coeff, areas))
    else:
        s = (
            np.einsum("ij,ij->i", t0, t0)
            + np.einsum("ij,ij->i", t1, t1)
            + np.einsum("ij,ij->i", t2, t2)
            + np.einsum("ij,ij->i", t0, t1)
            + np.einsum("ij,ij->i", t1, t2)
            + np.einsum("ij,ij->i", t2, t0)
        )
        coeff = (k_tilt / 12.0) * s
        energy = float(np.dot(coeff, areas))

    if grad_arr is not None:
        n_hat = n[mask] / n_norm[mask][:, None]
        g0 = 0.5 * _fast_cross(n_hat, (v2[mask] - v1[mask]))
        g1 = 0.5 * _fast_cross(n_hat, (v0[mask] - v2[mask]))
        g2 = 0.5 * _fast_cross(n_hat, (v1[mask] - v0[mask]))

        c = coeff[:, None]
        np.add.at(grad_arr, tri_rows_m[:, 0], c * g0)
        np.add.at(grad_arr, tri_rows_m[:, 1], c * g1)
        np.add.at(grad_arr, tri_rows_m[:, 2], c * g2)

    if tilt_out_grad_arr is not None:
        tilt_out_grad_arr = np.asarray(tilt_out_grad_arr, dtype=float)
        if tilt_out_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_out_grad_arr must have shape (N_vertices, 3)")

        if mode == "lumped":
            cache_ok = tri_keep.size == 0 or np.all(tri_keep)
            vertex_areas = mesh.barycentric_vertex_areas(
                positions,
                tri_rows=tri_rows_eff,
                areas=areas,
                mask=mask,
                cache=cache_ok,
            )
            tilt_out_grad_arr += k_tilt * tilts_eff * vertex_areas[:, None]
        else:
            tri_factor = (k_tilt * areas / 12.0)[:, None]
            np.add.at(
                tilt_out_grad_arr,
                tri_rows_m[:, 0],
                tri_factor * ((2.0 * t0) + t1 + t2),
            )
            np.add.at(
                tilt_out_grad_arr,
                tri_rows_m[:, 1],
                tri_factor * ((2.0 * t1) + t2 + t0),
            )
            np.add.at(
                tilt_out_grad_arr,
                tri_rows_m[:, 2],
                tri_factor * ((2.0 * t2) + t0 + t1),
            )

    return energy


def compute_energy_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
) -> float:
    """Dense-array outer-leaflet tilt energy (energy only)."""
    _ = index_map, tilts_in
    k_tilt = _resolve_tilt_modulus(param_resolver)
    if k_tilt == 0.0:
        return 0.0
    mode = _resolve_tilt_mass_mode(param_resolver)

    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0

    mesh.build_position_cache()
    absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet="out")
    tri_keep = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_mask
    )
    if tri_keep.size and not np.any(tri_keep):
        return 0.0

    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()
    else:
        tilts_out = np.asarray(tilts_out, dtype=float)
        if tilts_out.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")

    active_row_weights = _active_row_weights(mesh, param_resolver)
    if active_row_weights is not None:
        tilts_eff = tilts_out * active_row_weights[:, None]
    else:
        tilts_eff = tilts_out

    tri_rows_eff = tri_rows[tri_keep] if tri_keep.size else tri_rows
    _v0, _v1, _v2, n, n_norm, mask = _triangle_geometry(positions, tri_rows_eff)
    if not np.any(mask):
        return 0.0

    areas = 0.5 * n_norm[mask]
    tri_rows_m = tri_rows_eff[mask]
    t0 = tilts_eff[tri_rows_m[:, 0]]
    t1 = tilts_eff[tri_rows_m[:, 1]]
    t2 = tilts_eff[tri_rows_m[:, 2]]
    if mode == "lumped":
        tri_tilt_sq_sum = (
            np.einsum("ij,ij->i", t0, t0)
            + np.einsum("ij,ij->i", t1, t1)
            + np.einsum("ij,ij->i", t2, t2)
        )
        coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)
    else:
        s = (
            np.einsum("ij,ij->i", t0, t0)
            + np.einsum("ij,ij->i", t1, t1)
            + np.einsum("ij,ij->i", t2, t2)
            + np.einsum("ij,ij->i", t0, t1)
            + np.einsum("ij,ij->i", t1, t2)
            + np.einsum("ij,ij->i", t2, t0)
        )
        coeff = (k_tilt / 12.0) * s

    return float(np.dot(coeff, areas))


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]
