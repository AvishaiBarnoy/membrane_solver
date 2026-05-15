"""Outer-leaflet tilt magnitude energy module.

This module models a per-vertex tilt penalty for the outer leaflet:

    E = 1/2 * k_t * sum_v (|t_out,v|^2 * A_v)

where ``t_out,v`` is a 3D tangent tilt vector stored on each vertex and ``A_v``
is a barycentric area weight.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh
from modules.constraints.local_interface_shells import build_local_interface_shell_data
from modules.energy import tilt_leaflet
from modules.energy import tilt_utils as _tilt_utils

USES_TILT_LEAFLETS = True


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver
) -> Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return tilt energy and gradients for the outer leaflet."""
    return tilt_leaflet.compute_energy_and_gradient_leaflet(
        mesh, global_params, param_resolver, leaflet="out"
    )


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray | None,
    ctx=None,
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
    tilt_in_grad_arr: np.ndarray | None = None,
    tilt_out_grad_arr: np.ndarray | None = None,
) -> float:
    """Dense-array outer-leaflet tilt energy accumulation."""
    _ = tilts_in, tilt_in_grad_arr
    return tilt_leaflet.compute_energy_and_gradient_array_leaflet(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        ctx=ctx,
        tilts=tilts_out,
        tilt_grad_arr=tilt_out_grad_arr,
        leaflet="out",
    )


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
    return tilt_leaflet.compute_energy_array_leaflet(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        tilts=tilts_out,
        leaflet="out",
    )


def _active_row_weights(mesh: Mesh, param_resolver) -> np.ndarray | None:
    return _tilt_utils._active_row_weights(mesh, param_resolver, "out")


def _shared_rim_outer_shell_rows(mesh: Mesh) -> np.ndarray:
    return _tilt_utils._shared_rim_outer_shell_rows(mesh, "out")


def _resolve_shared_rim_outer_row_energy_weight(param_resolver) -> float | None:
    return _tilt_utils._resolve_shared_rim_outer_row_energy_weight(
        param_resolver, "out"
    )


def _resolve_shared_rim_outer_shell_mass_mode(param_resolver) -> str | None:
    return _tilt_utils._resolve_shared_rim_outer_shell_mass_mode(param_resolver, "out")


def _explicit_trace_layer_active_row_weights(
    mesh: Mesh, param_resolver
) -> np.ndarray | None:
    return _tilt_utils._explicit_trace_layer_active_row_weights(
        mesh, param_resolver, "out"
    )


def _shared_rim_active_row_weights(mesh: Mesh, param_resolver) -> np.ndarray | None:
    return _tilt_utils._shared_rim_active_row_weights(mesh, param_resolver, "out")


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
    "_active_row_weights",
    "_shared_rim_outer_shell_rows",
    "_resolve_shared_rim_outer_row_energy_weight",
    "_resolve_shared_rim_outer_shell_mass_mode",
    "_explicit_trace_layer_active_row_weights",
    "_shared_rim_active_row_weights",
    "build_local_interface_shell_data",
]
