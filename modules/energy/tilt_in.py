"""Inner-leaflet tilt magnitude energy module.

This module models a per-vertex tilt penalty for the inner leaflet:

    E = 1/2 * k_t * sum_v (|t_in,v|^2 * A_v)

where ``t_in,v`` is a 3D tangent tilt vector stored on each vertex and ``A_v``
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


def _resolve_exclude_shared_rim_rows(param_resolver) -> bool:
    raw = param_resolver.get(None, "tilt_in_exclude_shared_rim_rows")
    if raw is None:
        raw = param_resolver.get(None, "tilt_exclude_shared_rim_rows_in")
    if raw is None:
        return False
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _resolve_exclude_shared_rim_outer_rows(param_resolver) -> bool:
    raw = param_resolver.get(None, "tilt_in_exclude_shared_rim_outer_rows")
    if raw is None:
        raw = param_resolver.get(None, "tilt_exclude_shared_rim_outer_rows_in")
    if raw is None:
        return False
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _resolve_shared_rim_outer_row_energy_weight(param_resolver) -> float | None:
    raw = param_resolver.get(None, "tilt_in_shared_rim_outer_row_energy_weight")
    if raw is None:
        return None
    weight = float(raw)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(
            "tilt_in_shared_rim_outer_row_energy_weight must be a finite nonnegative float."
        )
    return weight


def _shared_rim_outer_shell_rows(mesh: Mesh) -> np.ndarray:
    """Return rows in the first outer shell used by shared-rim relief controls."""
    cache = getattr(mesh, "_tilt_in_shared_rim_outer_shell_rows_cache", None)
    if cache is None:
        cache = {}
        setattr(mesh, "_tilt_in_shared_rim_outer_shell_rows_cache", cache)

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


def _shared_rim_active_row_weights(mesh: Mesh, param_resolver) -> np.ndarray | None:
    """Return per-row tilt weights for the shared-rim inner-leaflet surrogate.

    Shared-rim row exclusions and experimental support-band weights apply only
    in ``shared_rim_staggered_v1``. The optional
    ``tilt_in_shared_rim_outer_row_energy_weight`` remains diagnostic-only; the
    canonical lane does not assign a default fractional weight.
    """
    exclude_rim_rows = _resolve_exclude_shared_rim_rows(param_resolver)
    exclude_outer_rows = _resolve_exclude_shared_rim_outer_rows(param_resolver)
    outer_row_energy_weight = _resolve_shared_rim_outer_row_energy_weight(
        param_resolver
    )
    mode = str(param_resolver.get(None, "rim_slope_match_mode") or "").strip().lower()
    has_explicit_override = (
        exclude_rim_rows or exclude_outer_rows or outer_row_energy_weight is not None
    )
    if mode != "shared_rim_staggered_v1" and not has_explicit_override:
        return None
    if not has_explicit_override:
        return None

    cache = getattr(mesh, "_tilt_in_shared_rim_active_row_weights_cache", None)
    if cache is None:
        cache = {}
        setattr(mesh, "_tilt_in_shared_rim_active_row_weights_cache", cache)

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
    outer_shell_rows = _shared_rim_outer_shell_rows(mesh)
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
    cache.clear()
    cache[cache_key] = weights
    return weights


def _explicit_trace_layer_active_row_weights(
    mesh: Mesh, param_resolver
) -> np.ndarray | None:
    """Return per-row weights when an explicit ghost trace layer is present.

    The inserted `R+epsilon` shell is an interface support layer used to read
    interface values directly. Its tilt mass should scale with the radial
    thickness it represents between the disk boundary and the first real
    free-side shell, not as a full extra annulus.
    """
    mode = str(param_resolver.get(None, "rim_slope_match_mode") or "").strip().lower()
    trace_radius = param_resolver.get(None, "parity_trace_layer_radius")
    lane = str(param_resolver.get(None, "theory_parity_lane") or "").strip()
    if mode != "physical_edge_staggered_v1" or trace_radius is None or not lane:
        return None

    cache = getattr(mesh, "_tilt_in_trace_layer_active_row_weights_cache", None)
    if cache is None:
        cache = {}
        setattr(mesh, "_tilt_in_trace_layer_active_row_weights_cache", cache)

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
    """Return the combined per-row weights for diagnostic shell controls."""
    shared = _shared_rim_active_row_weights(mesh, param_resolver)
    trace = _explicit_trace_layer_active_row_weights(mesh, param_resolver)
    if shared is None:
        return trace
    if trace is None:
        return shared
    return shared * trace


def _resolve_tilt_modulus(param_resolver) -> float:
    k = param_resolver.get(None, "tilt_modulus_in")
    if k is None:
        k = param_resolver.get(None, "tilt_modolus_in")
    return float(k or 0.0)


def _resolve_tilt_mass_mode(param_resolver) -> str:
    mode = param_resolver.get(None, "tilt_mass_mode_in")
    if mode is None:
        mode = param_resolver.get(None, "tilt_mass_mode")
    txt = str(mode or "lumped").strip().lower()
    if txt not in {"lumped", "consistent"}:
        raise ValueError("tilt_mass_mode_in must be 'lumped' or 'consistent'.")
    return txt


def _resolve_shared_rim_outer_shell_mass_mode(param_resolver) -> str | None:
    raw = param_resolver.get(None, "tilt_in_shared_rim_outer_shell_mass_mode")
    if raw is None:
        return None
    txt = str(raw).strip().lower()
    if txt not in {"lumped", "consistent"}:
        raise ValueError(
            "tilt_in_shared_rim_outer_shell_mass_mode must be 'lumped' or 'consistent'."
        )
    return txt


def _shared_rim_outer_support_triangle_mask(
    mesh: Mesh, tri_rows: np.ndarray
) -> np.ndarray | None:
    """Return a mask for triangles spanning only the first outer support shell."""
    if tri_rows.size == 0:
        return None

    cache = getattr(mesh, "_tilt_in_shared_rim_outer_support_tri_cache", None)
    if cache is None:
        cache = {}
        setattr(mesh, "_tilt_in_shared_rim_outer_support_tri_cache", cache)

    cache_key = (int(mesh._version), int(mesh._vertex_ids_version), tri_rows.shape[0])
    mask = cache.get(cache_key)
    if mask is not None and mask.shape == (len(tri_rows),):
        return mask

    outer_row_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    rim_row_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    disk_row_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    outer_shell_rows = _shared_rim_outer_shell_rows(mesh)
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
    """Return tilt energy and gradients for the inner leaflet."""
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
    absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet="in")
    tri_keep = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_mask
    )
    if tri_keep.size and not np.any(tri_keep):
        return 0.0, shape_grad, tilt_grad

    tilts_in = mesh.tilts_in_view()
    active_row_weights = _active_row_weights(mesh, param_resolver)
    if active_row_weights is not None:
        tilts_eff = tilts_in * active_row_weights[:, None]
    else:
        tilts_eff = tilts_in
    mode = _resolve_tilt_mass_mode(param_resolver)
    shell_mode = _resolve_shared_rim_outer_shell_mass_mode(param_resolver)
    tri_rows_eff = tri_rows[tri_keep] if tri_keep.size else tri_rows
    v0, v1, v2, n, n_norm, mask = _triangle_geometry(positions, tri_rows_eff)
    if not np.any(mask):
        return 0.0, shape_grad, tilt_grad

    areas = 0.5 * n_norm[mask]
    tri_rows_m = tri_rows_eff[mask]
    t0 = tilts_eff[tri_rows_m[:, 0]]
    t1 = tilts_eff[tri_rows_m[:, 1]]
    t2 = tilts_eff[tri_rows_m[:, 2]]
    tri_tilt_sq_sum = (
        np.einsum("ij,ij->i", t0, t0)
        + np.einsum("ij,ij->i", t1, t1)
        + np.einsum("ij,ij->i", t2, t2)
    )
    consistent_s = (
        tri_tilt_sq_sum
        + np.einsum("ij,ij->i", t0, t1)
        + np.einsum("ij,ij->i", t1, t2)
        + np.einsum("ij,ij->i", t2, t0)
    )
    outer_support_mask = _shared_rim_outer_support_triangle_mask(mesh, tri_rows_eff)
    if outer_support_mask is not None:
        outer_support_mask = outer_support_mask[mask]
    use_consistent = np.full(len(tri_rows_m), mode == "consistent", dtype=bool)
    if shell_mode is not None and outer_support_mask is not None:
        use_consistent[outer_support_mask] = shell_mode == "consistent"
    use_lumped = ~use_consistent

    coeff = np.empty(len(tri_rows_m), dtype=float)
    coeff[use_lumped] = 0.5 * k_tilt * (tri_tilt_sq_sum[use_lumped] / 3.0)
    coeff[use_consistent] = (k_tilt / 12.0) * consistent_s[use_consistent]
    energy = float(np.dot(coeff, areas))

    tilt_grad_arr = np.zeros_like(positions)
    if np.any(use_lumped):
        tri_factor = (k_tilt * areas[use_lumped] / 3.0)[:, None]
        rows_l = tri_rows_m[use_lumped]
        np.add.at(tilt_grad_arr, rows_l[:, 0], tri_factor * t0[use_lumped])
        np.add.at(tilt_grad_arr, rows_l[:, 1], tri_factor * t1[use_lumped])
        np.add.at(tilt_grad_arr, rows_l[:, 2], tri_factor * t2[use_lumped])
    if np.any(use_consistent):
        tri_factor = (k_tilt * areas[use_consistent] / 12.0)[:, None]
        rows_c = tri_rows_m[use_consistent]
        np.add.at(
            tilt_grad_arr,
            rows_c[:, 0],
            tri_factor
            * ((2.0 * t0[use_consistent]) + t1[use_consistent] + t2[use_consistent]),
        )
        np.add.at(
            tilt_grad_arr,
            rows_c[:, 1],
            tri_factor
            * ((2.0 * t1[use_consistent]) + t2[use_consistent] + t0[use_consistent]),
        )
        np.add.at(
            tilt_grad_arr,
            rows_c[:, 2],
            tri_factor
            * ((2.0 * t2[use_consistent]) + t0[use_consistent] + t1[use_consistent]),
        )

    if active_row_weights is not None:
        tilt_grad_arr *= active_row_weights[:, None]

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
    """Dense-array inner-leaflet tilt energy accumulation."""
    _ = global_params, index_map, tilts_out, tilt_out_grad_arr
    k_tilt = _resolve_tilt_modulus(param_resolver)
    if k_tilt == 0.0:
        return 0.0
    mode = _resolve_tilt_mass_mode(param_resolver)

    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0
    absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet="in")
    tri_keep = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_mask
    )
    if tri_keep.size and not np.any(tri_keep):
        return 0.0

    mesh.build_position_cache()
    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

    active_row_weights = _active_row_weights(mesh, param_resolver)
    if active_row_weights is not None:
        tilts_eff = tilts_in * active_row_weights[:, None]
    else:
        tilts_eff = tilts_in
    shell_mode = _resolve_shared_rim_outer_shell_mass_mode(param_resolver)

    tri_rows_eff = tri_rows[tri_keep] if tri_keep.size else tri_rows
    v0, v1, v2, n, n_norm, mask = _triangle_geometry(positions, tri_rows_eff)
    if not np.any(mask):
        return 0.0

    areas = 0.5 * n_norm[mask]
    tri_rows_m = tri_rows_eff[mask]
    t0 = tilts_eff[tri_rows_m[:, 0]]
    t1 = tilts_eff[tri_rows_m[:, 1]]
    t2 = tilts_eff[tri_rows_m[:, 2]]
    tri_tilt_sq_sum = (
        np.einsum("ij,ij->i", t0, t0)
        + np.einsum("ij,ij->i", t1, t1)
        + np.einsum("ij,ij->i", t2, t2)
    )
    consistent_s = (
        tri_tilt_sq_sum
        + np.einsum("ij,ij->i", t0, t1)
        + np.einsum("ij,ij->i", t1, t2)
        + np.einsum("ij,ij->i", t2, t0)
    )
    outer_support_mask = _shared_rim_outer_support_triangle_mask(mesh, tri_rows_eff)
    if outer_support_mask is not None:
        outer_support_mask = outer_support_mask[mask]
    use_consistent = np.full(len(tri_rows_m), mode == "consistent", dtype=bool)
    if shell_mode is not None and outer_support_mask is not None:
        use_consistent[outer_support_mask] = shell_mode == "consistent"
    use_lumped = ~use_consistent

    coeff = np.empty(len(tri_rows_m), dtype=float)
    coeff[use_lumped] = 0.5 * k_tilt * (tri_tilt_sq_sum[use_lumped] / 3.0)
    coeff[use_consistent] = (k_tilt / 12.0) * consistent_s[use_consistent]
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

    if tilt_in_grad_arr is not None:
        tilt_in_grad_arr = np.asarray(tilt_in_grad_arr, dtype=float)
        if tilt_in_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_in_grad_arr must have shape (N_vertices, 3)")

        tilt_grad_local = np.zeros_like(tilt_in_grad_arr)
        if np.any(use_lumped):
            tri_factor = (k_tilt * areas[use_lumped] / 3.0)[:, None]
            rows_l = tri_rows_m[use_lumped]
            np.add.at(
                tilt_grad_local,
                rows_l[:, 0],
                tri_factor * t0[use_lumped],
            )
            np.add.at(
                tilt_grad_local,
                rows_l[:, 1],
                tri_factor * t1[use_lumped],
            )
            np.add.at(
                tilt_grad_local,
                rows_l[:, 2],
                tri_factor * t2[use_lumped],
            )
        if np.any(use_consistent):
            tri_factor = (k_tilt * areas[use_consistent] / 12.0)[:, None]
            rows_c = tri_rows_m[use_consistent]
            np.add.at(
                tilt_grad_local,
                rows_c[:, 0],
                tri_factor
                * (
                    (2.0 * t0[use_consistent]) + t1[use_consistent] + t2[use_consistent]
                ),
            )
            np.add.at(
                tilt_grad_local,
                rows_c[:, 1],
                tri_factor
                * (
                    (2.0 * t1[use_consistent]) + t2[use_consistent] + t0[use_consistent]
                ),
            )
            np.add.at(
                tilt_grad_local,
                rows_c[:, 2],
                tri_factor
                * (
                    (2.0 * t2[use_consistent]) + t0[use_consistent] + t1[use_consistent]
                ),
            )
        if active_row_weights is not None:
            tilt_grad_local *= active_row_weights[:, None]
        tilt_in_grad_arr += tilt_grad_local

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
    """Dense-array inner-leaflet tilt energy (energy only)."""
    _ = global_params, index_map, tilts_out
    k_tilt = _resolve_tilt_modulus(param_resolver)
    if k_tilt == 0.0:
        return 0.0
    mode = _resolve_tilt_mass_mode(param_resolver)

    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0

    mesh.build_position_cache()
    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

    active_row_weights = _active_row_weights(mesh, param_resolver)
    if active_row_weights is not None:
        tilts_eff = tilts_in * active_row_weights[:, None]
    else:
        tilts_eff = tilts_in
    shell_mode = _resolve_shared_rim_outer_shell_mass_mode(param_resolver)

    v0, v1, v2, n, n_norm, mask = _triangle_geometry(positions, tri_rows)
    if not np.any(mask):
        return 0.0

    areas = 0.5 * n_norm[mask]
    tri_rows_m = tri_rows[mask]
    t0 = tilts_eff[tri_rows_m[:, 0]]
    t1 = tilts_eff[tri_rows_m[:, 1]]
    t2 = tilts_eff[tri_rows_m[:, 2]]
    tri_tilt_sq_sum = (
        np.einsum("ij,ij->i", t0, t0)
        + np.einsum("ij,ij->i", t1, t1)
        + np.einsum("ij,ij->i", t2, t2)
    )
    consistent_s = (
        tri_tilt_sq_sum
        + np.einsum("ij,ij->i", t0, t1)
        + np.einsum("ij,ij->i", t1, t2)
        + np.einsum("ij,ij->i", t2, t0)
    )
    outer_support_mask = _shared_rim_outer_support_triangle_mask(mesh, tri_rows)
    if outer_support_mask is not None:
        outer_support_mask = outer_support_mask[mask]
    use_consistent = np.full(len(tri_rows_m), mode == "consistent", dtype=bool)
    if shell_mode is not None and outer_support_mask is not None:
        use_consistent[outer_support_mask] = shell_mode == "consistent"
    use_lumped = ~use_consistent

    coeff = np.empty(len(tri_rows_m), dtype=float)
    coeff[use_lumped] = 0.5 * k_tilt * (tri_tilt_sq_sum[use_lumped] / 3.0)
    coeff[use_consistent] = (k_tilt / 12.0) * consistent_s[use_consistent]

    return float(np.dot(coeff, areas))


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]
