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
from modules.energy import tilt_smoothness_leaflet as _leaflet
from modules.energy import tilt_smoothness_utils as _utils

USES_TILT_LEAFLETS = True


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver, *, compute_gradient: bool = True
) -> (
    Tuple[float, Dict[int, np.ndarray]]
    | Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Dict-based wrapper returning (E, shape_grad[, tilt_grad])."""
    return _leaflet.compute_energy_and_gradient_leaflet(
        mesh,
        global_params,
        param_resolver,
        leaflet="in",
        compute_gradient=compute_gradient,
    )


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
    ctx=None,
) -> float:
    """Dense-array inner-leaflet smoothness energy accumulation."""
    _ = tilts_out, tilt_out_grad_arr
    return _leaflet.compute_energy_and_gradient_array_leaflet(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        ctx=ctx,
        tilts=tilts_in,
        tilt_grad_arr=tilt_in_grad_arr,
        leaflet="in",
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
    ctx=None,
) -> float:
    """Dense-array inner-leaflet smoothness energy (energy only)."""
    return _leaflet.compute_energy_array_leaflet(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        ctx=ctx,
        tilts=tilts_in,
        leaflet="in",
    )


def _masked_weights_and_tris(mesh: Mesh, global_params, **kwargs):
    return _utils._masked_weights_and_tris(mesh, global_params, leaflet="in", **kwargs)


def _resolve_smoothness_rigidity(param_resolver) -> float:
    return _utils._resolve_smoothness_rigidity(param_resolver, "in")


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
    "_masked_weights_and_tris",
    "_resolve_smoothness_rigidity",
]
