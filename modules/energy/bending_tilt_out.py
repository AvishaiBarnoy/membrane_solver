"""Helfrich bending energy with outer-leaflet tilt-splay coupling."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.entities import Mesh
from modules.energy import bending_tilt_leaflet as _leaflet

USES_TILT_LEAFLETS = True


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
    """Compute coupled bending+tilt energy for outer-leaflet tilts."""
    _ = tilts_in, tilt_in_grad_arr
    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()

    return _leaflet.compute_energy_and_gradient_array_leaflet(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts=tilts_out,
        tilt_grad_arr=tilt_out_grad_arr,
        kappa_key="bending_modulus_out",
        cache_tag="out",
        div_sign=1.0,
    )


def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> (
    tuple[float, Dict[int, np.ndarray]]
    | tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Return energy and gradients for outer-leaflet bending-tilt coupling."""
    tilts_out = mesh.tilts_out_view()
    return _leaflet.compute_energy_and_gradient_leaflet(
        mesh,
        global_params,
        param_resolver,
        tilts=tilts_out,
        kappa_key="bending_modulus_out",
        cache_tag="out",
        div_sign=1.0,
        compute_gradient=compute_gradient,
    )


__all__ = ["compute_energy_and_gradient", "compute_energy_and_gradient_array"]
