"""Shape-aware curved local-interface law on the true shell family near r=R.

This module penalizes mismatch between the outer-leaflet radial tilt on the
first shell outside the disk and a local outer slope built from the first two
outside shells. Unlike the earlier penalty term, this law contributes the
corresponding z-shape gradients of the local slope target.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh
from modules.constraints.local_interface_shells import build_local_interface_shell_data

USES_TILT_LEAFLETS = True


def _resolve_strength(param_resolver) -> float:
    value = param_resolver.get(None, "curved_local_interface_law_strength")
    return float(value or 0.0)


def _payload(mesh: Mesh, positions: np.ndarray) -> dict[str, np.ndarray] | None:
    shell_data = build_local_interface_shell_data(mesh, positions=positions)
    rim_rows = np.asarray(shell_data.rim_rows_matched, dtype=int)
    outer_rows = np.asarray(shell_data.outer_rows, dtype=int)
    if rim_rows.size == 0 or outer_rows.size == 0:
        return None

    radii = np.linalg.norm(positions[:, :2], axis=1)
    dr = radii[outer_rows] - radii[rim_rows]
    valid = np.abs(dr) > 1.0e-12
    if not np.any(valid):
        return None

    phi = np.zeros(rim_rows.size, dtype=float)
    inv_dr = np.zeros(rim_rows.size, dtype=float)
    inv_dr[valid] = 1.0 / dr[valid]
    phi[valid] = (
        positions[outer_rows[valid], 2] - positions[rim_rows[valid], 2]
    ) * inv_dr[valid]
    return {
        "rim_rows": rim_rows,
        "outer_rows": outer_rows,
        "r_hat": np.asarray(shell_data.rim_r_hat, dtype=float),
        "phi": phi,
        "inv_dr": inv_dr,
        "valid": valid,
    }


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver
) -> Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Dict-based wrapper returning (E, shape_grad, tilt_out_grad)."""
    positions = mesh.positions_view()
    grad_arr = np.zeros_like(positions)
    tilt_out_grad_arr = np.zeros_like(positions)
    energy = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad_arr,
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=tilt_out_grad_arr,
    )
    shape_grad = {
        int(vid): grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }
    tilt_out_grad = {
        int(vid): tilt_out_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(tilt_out_grad_arr[row])
    }
    return float(energy), shape_grad, tilt_out_grad


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
    """Dense-array curved local-interface law accumulation."""
    _ = global_params, index_map, tilts_in, tilt_in_grad_arr
    strength = _resolve_strength(param_resolver)
    if strength == 0.0:
        return 0.0

    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()
    else:
        tilts_out = np.asarray(tilts_out, dtype=float)
        if tilts_out.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")

    payload = _payload(mesh, positions)
    if payload is None:
        return 0.0

    rim_rows = payload["rim_rows"]
    outer_rows = payload["outer_rows"]
    r_hat = payload["r_hat"]
    phi = payload["phi"]
    inv_dr = payload["inv_dr"]
    valid = payload["valid"]

    diff = np.einsum("ij,ij->i", tilts_out[rim_rows], r_hat) - phi
    diff = np.where(valid, diff, 0.0)
    energy = 0.5 * strength * float(np.dot(diff, diff))

    if tilt_out_grad_arr is not None:
        tilt_out_grad_arr = np.asarray(tilt_out_grad_arr, dtype=float)
        if tilt_out_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_out_grad_arr must have shape (N_vertices, 3)")
        tilt_contrib = (strength * diff)[:, None] * r_hat
        tilt_contrib[~valid] = 0.0
        np.add.at(tilt_out_grad_arr, rim_rows, tilt_contrib)

    if grad_arr is not None:
        grad_arr = np.asarray(grad_arr, dtype=float)
        if grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("grad_arr must have shape (N_vertices, 3)")
        shape_scale = strength * diff * inv_dr
        shape_scale[~valid] = 0.0
        rim_shape = np.zeros((rim_rows.size, 3), dtype=float)
        outer_shape = np.zeros((outer_rows.size, 3), dtype=float)
        rim_shape[:, 2] = shape_scale
        outer_shape[:, 2] = -shape_scale
        np.add.at(grad_arr, rim_rows, rim_shape)
        np.add.at(grad_arr, outer_rows, outer_shape)

    return float(energy)


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
    """Dense-array curved local-interface law (energy only)."""
    _ = global_params, index_map, tilts_in
    strength = _resolve_strength(param_resolver)
    if strength == 0.0:
        return 0.0

    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()
    else:
        tilts_out = np.asarray(tilts_out, dtype=float)
        if tilts_out.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")

    payload = _payload(mesh, positions)
    if payload is None:
        return 0.0

    diff = (
        np.einsum("ij,ij->i", tilts_out[payload["rim_rows"]], payload["r_hat"])
        - payload["phi"]
    )
    diff = np.where(payload["valid"], diff, 0.0)
    return 0.5 * strength * float(np.dot(diff, diff))


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]
