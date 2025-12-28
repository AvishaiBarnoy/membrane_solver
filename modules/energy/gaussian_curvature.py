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
from runtime.diagnostics.gauss_bonnet import (
    extract_boundary_loops,
    find_boundary_edges,
    gauss_bonnet_invariant,
)

logger = logging.getLogger("membrane_solver")


def _gaussian_modulus(global_params) -> float:
    return float(global_params.get("gaussian_modulus", 0.0) or 0.0)


def _euler_characteristic(mesh: Mesh) -> int:
    """Euler characteristic of the current mesh complex (V - E + F)."""
    return int(len(mesh.vertices) - len(mesh.edges) + len(mesh.facets))


def _strict_topology_check(
    mesh: Mesh,
    positions: np.ndarray,
    index_map: Dict[int, int],
    *,
    tol: float,
    boundary_edges: list | None = None,
    boundary_loops: list | None = None,
) -> None:
    """Raise if the mesh is non-manifold or Gauss-Bonnet mismatch is large."""
    mesh.build_connectivity_maps()
    non_manifold = [
        eid for eid, facets in mesh.edge_to_facets.items() if len(facets) > 2
    ]
    if non_manifold:
        raise ValueError(
            "gaussian_curvature strict check: non-manifold edges detected "
            f"(count={len(non_manifold)})."
        )

    if boundary_edges is None:
        boundary_edges = []
    if boundary_loops is None:
        boundary_loops = []

    if boundary_edges:
        deg = {}
        for edge in boundary_edges:
            deg[edge.tail_index] = deg.get(edge.tail_index, 0) + 1
            deg[edge.head_index] = deg.get(edge.head_index, 0) + 1
        bad = {vid: cnt for vid, cnt in deg.items() if cnt != 2}
        if bad:
            raise ValueError(
                "gaussian_curvature strict check: boundary vertex degree != 2 "
                f"(count={len(bad)})."
            )
        if not boundary_loops:
            raise ValueError(
                "gaussian_curvature strict check: boundary edges present but no loops found."
            )
        short = [loop for loop in boundary_loops if len(loop) < 3]
        if short:
            raise ValueError(
                "gaussian_curvature strict check: boundary loop too short "
                f"(count={len(short)})."
            )
        return

    defects = compute_angle_defects(mesh, positions, index_map)
    defect_sum = float(np.sum(defects))
    chi = _euler_characteristic(mesh)
    target = float(2.0 * np.pi * chi)
    err = abs(defect_sum - target)
    if err > tol:
        raise ValueError(
            "gaussian_curvature strict check: defect sum mismatch "
            f"(sum(defect)={defect_sum:.6e}, 2πχ={target:.6e}, |Δ|={err:.3e})."
        )


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

    def facet_filter(facet):
        return not bool(facet.options.get("gauss_bonnet_exclude", False))
    boundary_edges = find_boundary_edges(mesh, facet_filter=facet_filter)
    boundary_loops = extract_boundary_loops(mesh, boundary_edges)

    if boundary_edges:
        if not boundary_loops:
            logger.warning(
                "gaussian_curvature: boundary edges detected but no loops extracted."
            )
        for loop in boundary_loops:
            if len(loop) < 3:
                logger.warning(
                    "gaussian_curvature: boundary loop has <3 vertices; results may be unreliable."
                )
        g_total, _, _, _ = gauss_bonnet_invariant(mesh, facet_filter=facet_filter)
        energy = float(kappa_bar * g_total)
    else:
        chi = _euler_characteristic(mesh)
        energy = float(2.0 * np.pi * kappa_bar * chi)

    if (not boundary_edges) and bool(
        global_params.get("gaussian_curvature_check_defects", False)
    ):
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
    if bool(global_params.get("gaussian_curvature_strict_topology", False)):
        tol = float(global_params.get("gaussian_curvature_defect_tol", 1e-6))
        _strict_topology_check(
            mesh,
            positions,
            index_map,
            tol=tol,
            boundary_edges=boundary_edges,
            boundary_loops=boundary_loops,
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
