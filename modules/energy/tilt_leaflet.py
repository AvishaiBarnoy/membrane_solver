"""Unified tilt magnitude energy module for inner and outer leaflets."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)
from modules.energy.tilt_params import (
    _resolve_tilt_mass_mode,
    _resolve_tilt_modulus,
)
from modules.energy.tilt_utils import (
    _active_row_weights,
    _resolve_shared_rim_outer_shell_mass_mode,
    _shared_rim_outer_support_triangle_mask,
    _triangle_geometry,
)


def compute_energy_and_gradient_array_leaflet(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray | None,
    ctx=None,
    tilts: np.ndarray | None,
    tilt_grad_arr: np.ndarray | None,
    leaflet: str,
) -> float:
    """Compute tilt magnitude energy and accumulate gradients for a leaflet."""
    _ = global_params, index_map, ctx
    k_tilt = _resolve_tilt_modulus(param_resolver, leaflet)
    if k_tilt == 0.0:
        return 0.0
    mode = _resolve_tilt_mass_mode(param_resolver, leaflet)
    shell_mode = _resolve_shared_rim_outer_shell_mass_mode(param_resolver, leaflet)

    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0

    mesh.build_position_cache()
    absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet=leaflet)
    tri_keep = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_mask
    )
    if tri_keep.size and not np.any(tri_keep):
        return 0.0

    if tilts is None:
        if leaflet == "in":
            tilts = mesh.tilts_in_view()
        else:
            tilts = mesh.tilts_out_view()
    else:
        tilts = np.asarray(tilts, dtype=float)
        if tilts.shape != (len(mesh.vertex_ids), 3):
            raise ValueError(
                f"tilts for leaflet '{leaflet}' must have shape (N_vertices, 3)"
            )

    active_row_weights = _active_row_weights(mesh, param_resolver, leaflet)
    if active_row_weights is not None:
        tilts_eff = tilts * active_row_weights[:, None]
    else:
        tilts_eff = tilts

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

    outer_support_mask = _shared_rim_outer_support_triangle_mask(
        mesh, tri_rows_eff, leaflet
    )
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

    if tilt_grad_arr is not None:
        grad_local = np.zeros_like(tilts)
        if np.any(use_lumped):
            tri_factor = (k_tilt * areas[use_lumped] / 3.0)[:, None]
            rows_l = tri_rows_m[use_lumped]
            np.add.at(grad_local, rows_l[:, 0], tri_factor * t0[use_lumped])
            np.add.at(grad_local, rows_l[:, 1], tri_factor * t1[use_lumped])
            np.add.at(grad_local, rows_l[:, 2], tri_factor * t2[use_lumped])
        if np.any(use_consistent):
            tri_factor = (k_tilt * areas[use_consistent] / 12.0)[:, None]
            rows_c = tri_rows_m[use_consistent]
            np.add.at(
                grad_local,
                rows_c[:, 0],
                tri_factor
                * (
                    (2.0 * t0[use_consistent]) + t1[use_consistent] + t2[use_consistent]
                ),
            )
            np.add.at(
                grad_local,
                rows_c[:, 1],
                tri_factor
                * (
                    (2.0 * t1[use_consistent]) + t2[use_consistent] + t0[use_consistent]
                ),
            )
            np.add.at(
                grad_local,
                rows_c[:, 2],
                tri_factor
                * (
                    (2.0 * t2[use_consistent]) + t0[use_consistent] + t1[use_consistent]
                ),
            )

        if active_row_weights is not None:
            grad_local *= active_row_weights[:, None]
        tilt_grad_arr += grad_local

    if grad_arr is not None:
        from geometry.facet import _fast_cross

        n_hat = n[mask] / n_norm[mask][:, None]
        g0 = 0.5 * _fast_cross(n_hat, (v2[mask] - v1[mask]))
        g1 = 0.5 * _fast_cross(n_hat, (v0[mask] - v2[mask]))
        g2 = 0.5 * _fast_cross(n_hat, (v1[mask] - v0[mask]))

        c = coeff[:, None]
        np.add.at(grad_arr, tri_rows_m[:, 0], c * g0)
        np.add.at(grad_arr, tri_rows_m[:, 1], c * g1)
        np.add.at(grad_arr, tri_rows_m[:, 2], c * g2)

    return energy


def compute_energy_array_leaflet(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts: np.ndarray | None = None,
    leaflet: str,
) -> float:
    """Compute tilt magnitude energy for a leaflet (energy only)."""
    return compute_energy_and_gradient_array_leaflet(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=None,
        tilts=tilts,
        tilt_grad_arr=None,
        leaflet=leaflet,
    )


def compute_energy_and_gradient_leaflet(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    leaflet: str,
) -> Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return tilt energy and gradients for a leaflet."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions)

    energy = compute_energy_and_gradient_array_leaflet(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts=None,
        tilt_grad_arr=tilt_grad_arr,
        leaflet=leaflet,
    )

    shape_grad = {}
    tg_dict = {}
    for row, vid in enumerate(mesh.vertex_ids):
        vidx = int(vid)
        shape_grad[vidx] = grad_arr[row]
        tg_dict[vidx] = tilt_grad_arr[row]

    return energy, shape_grad, tg_dict
