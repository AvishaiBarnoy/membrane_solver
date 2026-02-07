"""Inner-leaflet tilt smoothness energy module.

This module applies a Dirichlet energy to the inner-leaflet tilt field:

    E = 1/2 * k_s * ∫ |∇t_in|^2 dA

The stiffness defaults to ``bending_modulus`` unless an explicit
``bending_modulus_in`` is provided.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh
from modules.energy import tilt_smoothness as _base

USES_TILT_LEAFLETS = True


def _resolve_bending_modulus(param_resolver) -> float:
    k = param_resolver.get(None, "bending_modulus_in")
    if k is None:
        k = param_resolver.get(None, "bending_modulus")
    return float(k or 0.0)


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver, *, compute_gradient: bool = True
) -> (
    Tuple[float, Dict[int, np.ndarray]]
    | Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Dict-based wrapper returning (E, shape_grad[, tilt_grad])."""
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
    """Dense-array inner-leaflet smoothness energy accumulation."""
    _ = global_params, index_map, tilts_out, tilt_out_grad_arr, grad_arr
    k_smooth = _resolve_bending_modulus(param_resolver)
    if k_smooth == 0.0:
        return 0.0

    weights, tri_rows = _base._get_weights_and_tris(
        mesh, positions=positions, index_map=index_map
    )
    if tri_rows is None:
        return 0.0

    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

    c0 = weights[:, 0]
    c1 = weights[:, 1]
    c2 = weights[:, 2]

    t0 = tilts_in[tri_rows[:, 0]]
    t1 = tilts_in[tri_rows[:, 1]]
    t2 = tilts_in[tri_rows[:, 2]]

    d12 = t1 - t2
    d20 = t2 - t0
    d01 = t0 - t1

    n12 = np.einsum("ij,ij->i", d12, d12)
    n20 = np.einsum("ij,ij->i", d20, d20)
    n01 = np.einsum("ij,ij->i", d01, d01)

    energy = float(0.25 * k_smooth * np.sum(c0 * n12 + c1 * n20 + c2 * n01))

    if tilt_in_grad_arr is not None:
        tilt_in_grad_arr = np.asarray(tilt_in_grad_arr, dtype=float)
        if tilt_in_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_in_grad_arr must have shape (N_vertices, 3)")

        factor = 0.5 * k_smooth
        np.add.at(
            tilt_in_grad_arr,
            tri_rows[:, 0],
            factor * (c1[:, None] * (t0 - t2) + c2[:, None] * (t0 - t1)),
        )
        np.add.at(
            tilt_in_grad_arr,
            tri_rows[:, 1],
            factor * (c2[:, None] * (t1 - t0) + c0[:, None] * (t1 - t2)),
        )
        np.add.at(
            tilt_in_grad_arr,
            tri_rows[:, 2],
            factor * (c0[:, None] * (t2 - t1) + c1[:, None] * (t2 - t0)),
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
    """Dense-array inner-leaflet smoothness energy (energy only)."""
    _ = global_params, index_map, tilts_out
    k_smooth = _resolve_bending_modulus(param_resolver)
    if k_smooth == 0.0:
        return 0.0

    weights, tri_rows = _base._get_weights_and_tris(
        mesh, positions=positions, index_map=index_map
    )
    if tri_rows is None:
        return 0.0

    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

    c0 = weights[:, 0]
    c1 = weights[:, 1]
    c2 = weights[:, 2]

    t0 = tilts_in[tri_rows[:, 0]]
    t1 = tilts_in[tri_rows[:, 1]]
    t2 = tilts_in[tri_rows[:, 2]]

    d12 = t1 - t2
    d20 = t2 - t0
    d01 = t0 - t1

    n12 = np.einsum("ij,ij->i", d12, d12)
    n20 = np.einsum("ij,ij->i", d20, d20)
    n01 = np.einsum("ij,ij->i", d01, d01)

    return float(0.25 * k_smooth * np.sum(c0 * n12 + c1 * n20 + c2 * n01))


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]
