r"""Gaussian curvature (Helfrich Gaussian modulus) energy module.

Implements the Helfrich Gaussian term

    E_G = \bar{kappa} \int K \, dA

for closed vesicles.

For a closed surface with constant ``gaussian_modulus`` and fixed topology,
Gauss–Bonnet implies

    \int K \, dA = 2π * chi,

so the energy is a constant and its shape gradient is zero. This module
therefore adds a constant energy offset without affecting minimization
forces.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from geometry.curvature import compute_angle_defects
from geometry.entities import Mesh

logger = logging.getLogger("membrane_solver")


def _gaussian_modulus(global_params) -> float:
    return float(global_params.get("gaussian_modulus", 0.0) or 0.0)


def _euler_characteristic(mesh: Mesh) -> int:
    """Euler characteristic of the current mesh complex (V - E + F)."""
    return int(len(mesh.vertices) - len(mesh.edges) + len(mesh.facets))


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray,
) -> float:
    """Compute Gaussian curvature energy (closed surfaces; constant modulus).

    Parameters
    ----------
    grad_arr:
        Included for interface consistency; this module does not write to it
        for closed surfaces with constant modulus.
    """
    kappa_bar = _gaussian_modulus(global_params)
    if kappa_bar == 0.0:
        return 0.0

    boundary_vids = getattr(mesh, "boundary_vertex_ids", None) or []
    if boundary_vids:
        raise NotImplementedError(
            "gaussian_curvature energy currently supports only closed surfaces "
            "(no boundary vertices)."
        )

    chi = _euler_characteristic(mesh)
    energy = float(2.0 * np.pi * kappa_bar * chi)

    if bool(global_params.get("gaussian_curvature_check_defects", False)):
        defects = compute_angle_defects(mesh, positions, index_map)
        defect_sum = float(np.sum(defects))
        target = float(2.0 * np.pi * chi)
        err = abs(defect_sum - target)
        if err > 1e-6:
            logger.warning(
                "Gaussian curvature defect sum mismatch: sum(defect)=%.6e, "
                "2πχ=%.6e (|Δ|=%.3e). Check for non-manifold topology.",
                defect_sum,
                target,
                err,
            )

    return energy


def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> tuple[float, Dict[int, np.ndarray]]:
    """Dictionary-based wrapper around the array backend."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    energy = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
    )
    if not compute_gradient:
        return float(energy), {}
    return float(energy), {}


__all__ = ["compute_energy_and_gradient", "compute_energy_and_gradient_array"]
