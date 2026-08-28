"""Outer-leaflet soft target for the analytic radial disk tilt profile."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.entities import Mesh
from modules.energy.tilt_disk_target_common import compute_target_energy

USES_TILT_LEAFLETS = True


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
    """Dense-array outer-leaflet disk target energy accumulation."""
    _ = global_params, index_map, tilts_in, tilt_in_grad_arr
    return compute_target_energy(
        mesh,
        param_resolver,
        leaflet="out",
        positions=positions,
        tilts=tilts_out,
        grad_arr=grad_arr,
        tilt_grad_arr=tilt_out_grad_arr,
    )


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver
) -> tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return energy and gradients for the outer-leaflet disk target."""
    positions = mesh.positions_view()
    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions)
    energy = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad_arr,
        tilts_out=None,
        tilt_out_grad_arr=tilt_grad_arr,
    )
    shape_grad = {
        int(vertex_id): grad_arr[row].copy()
        for row, vertex_id in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }
    tilt_grad = {
        int(vertex_id): tilt_grad_arr[row].copy()
        for row, vertex_id in enumerate(mesh.vertex_ids)
        if np.any(tilt_grad_arr[row])
    }
    return float(energy), shape_grad, tilt_grad


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
    """Dense-array outer-leaflet disk target energy (energy only)."""
    _ = global_params, index_map, tilts_in
    return compute_target_energy(
        mesh,
        param_resolver,
        leaflet="out",
        positions=positions,
        tilts=tilts_out,
        grad_arr=None,
        tilt_grad_arr=None,
        compute_gradients=False,
    )


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]
