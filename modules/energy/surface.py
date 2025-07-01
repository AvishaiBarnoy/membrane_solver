# modules/surface.py
# Here goes energy functions relevant for area of facets

from geometry.entities import Mesh, Facet
from typing import Dict
from collections import defaultdict
from logging_config import setup_logging
import numpy as np

logger = setup_logging('membrane_solver')

def calculate_surface_energy(mesh: Mesh, global_params) -> float:
    """Compute the total surface energy for all facets."""
    gammas = []
    areas = []
    for facet in mesh.facets.values():
        gammas.append(facet.options.get("surface_tension", global_params.get("surface_tension")))
        areas.append(facet.compute_area(mesh))
    if not gammas:
        return 0.0
    return float(np.dot(gammas, areas))

def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> tuple[float, Dict[int, np.ndarray]]:
    """Compute surface energy and optionally its gradient.

    Parameters
    ----------
    mesh : Mesh
        The mesh containing facets.
    global_params : GlobalParameters
        Global parameter store with defaults.
    param_resolver : ParameterResolver
        Resolver to obtain per-object parameters.
    compute_gradient : bool, optional
        If ``True`` also compute the gradient, by default ``True``.

    Returns
    -------
    tuple[float, Dict[int, np.ndarray]]
        Total surface energy and gradient per vertex.
    """

    E = 0.0

    if compute_gradient:
        vidxs = list(mesh.vertices.keys())
        idx_map = {v: i for i, v in enumerate(vidxs)}
        grad_arr = np.zeros((len(vidxs), 3))
    else:
        idx_map = {}
        grad_arr = None

    for facet in mesh.facets.values():
        surface_tension = param_resolver.get(facet, "surface_tension")
        if surface_tension is None:
            surface_tension = global_params.get("surface_tension")

        area = facet.compute_area(mesh)
        E += surface_tension * area

        if compute_gradient:
            area_gradient = facet.compute_area_gradient(mesh)
            for vertex_index, gradient_vector in area_gradient.items():
                grad_arr[idx_map[vertex_index]] += surface_tension * gradient_vector

    if compute_gradient:
        grad = {v: grad_arr[i] for v, i in idx_map.items()}
        logger.debug(f"Computed surface energy: {E}")
        logger.debug(f"Computed surface energy gradient: {grad}")
        return E, grad
    else:
        logger.debug(f"Computed surface energy: {E}")
        return E, {}
