"""Inner-leaflet split splay/twist tilt-gradient energy."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross
from geometry.tilt_operators import p1_triangle_shape_gradients

USES_TILT_LEAFLETS = True


def _resolve_splay_modulus(param_resolver) -> float:
    """Resolve inner-leaflet splay modulus with bending-modulus fallback."""
    val = param_resolver.get(None, "tilt_splay_modulus_in")
    if val is None:
        val = param_resolver.get(None, "bending_modulus_in")
    if val is None:
        val = param_resolver.get(None, "bending_modulus")
    k = float(val or 0.0)
    if k < 0.0:
        raise ValueError("tilt_splay_modulus_in must be non-negative.")
    return k


def _resolve_twist_modulus(param_resolver) -> float:
    """Resolve inner-leaflet twist modulus with default zero twist."""
    val = param_resolver.get(None, "tilt_twist_modulus_in")
    if val is None:
        val = param_resolver.get(None, "tilt_twist_modulus")
    k = float(val or 0.0)
    if k < 0.0:
        raise ValueError("tilt_twist_modulus_in must be non-negative.")
    return k


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver, *, compute_gradient: bool = True
) -> (
    Tuple[float, Dict[int, np.ndarray]]
    | Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Dict-based wrapper returning ``(E, shape_grad[, tilt_grad])``."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions) if compute_gradient else None
    energy = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts_in=None,
        tilt_in_grad_arr=tilt_grad_arr,
    )
    if not compute_gradient:
        return float(energy), {}

    shape_grad = {
        int(vid): grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }
    tilt_grad = {
        int(vid): tilt_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if tilt_grad_arr is not None and np.any(tilt_grad_arr[row])
    }
    return float(energy), shape_grad, tilt_grad


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray,
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
    tilt_in_grad_arr: np.ndarray | None = None,
    tilt_out_grad_arr: np.ndarray | None = None,
) -> float:
    """Dense-array inner-leaflet splay/twist energy."""
    _ = global_params, index_map, grad_arr, tilts_out, tilt_out_grad_arr
    k_splay = _resolve_splay_modulus(param_resolver)
    k_twist = _resolve_twist_modulus(param_resolver)
    if k_splay == 0.0 and k_twist == 0.0:
        return 0.0

    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0

    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

    area, g0, g1, g2 = p1_triangle_shape_gradients(
        positions=positions, tri_rows=tri_rows
    )

    t0 = tilts_in[tri_rows[:, 0]]
    t1 = tilts_in[tri_rows[:, 1]]
    t2 = tilts_in[tri_rows[:, 2]]

    div_tri = (
        np.einsum("ij,ij->i", t0, g0)
        + np.einsum("ij,ij->i", t1, g1)
        + np.einsum("ij,ij->i", t2, g2)
    )

    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]
    n = _fast_cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1)
    n_hat = np.zeros_like(n)
    good = n_norm > 1e-20
    n_hat[good] = n[good] / n_norm[good, None]

    curl_vec = _fast_cross(g0, t0) + _fast_cross(g1, t1) + _fast_cross(g2, t2)
    curl_n = np.einsum("ij,ij->i", curl_vec, n_hat)

    energy_density = k_splay * div_tri * div_tri + k_twist * curl_n * curl_n
    energy = float(0.5 * np.sum(area * energy_density))

    if tilt_in_grad_arr is not None:
        tilt_in_grad_arr = np.asarray(tilt_in_grad_arr, dtype=float)
        if tilt_in_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_in_grad_arr must have shape (N_vertices, 3)")

        coeff_div = area * k_splay * div_tri
        coeff_curl = area * k_twist * curl_n

        d0 = coeff_div[:, None] * g0 + coeff_curl[:, None] * _fast_cross(n_hat, g0)
        d1 = coeff_div[:, None] * g1 + coeff_curl[:, None] * _fast_cross(n_hat, g1)
        d2 = coeff_div[:, None] * g2 + coeff_curl[:, None] * _fast_cross(n_hat, g2)

        np.add.at(tilt_in_grad_arr, tri_rows[:, 0], d0)
        np.add.at(tilt_in_grad_arr, tri_rows[:, 1], d1)
        np.add.at(tilt_in_grad_arr, tri_rows[:, 2], d2)

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
    """Dense-array inner-leaflet splay/twist energy (energy-only)."""
    _ = global_params, index_map, tilts_out
    return compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=np.zeros_like(positions),
        tilts_in=tilts_in,
        tilt_in_grad_arr=None,
    )


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]
