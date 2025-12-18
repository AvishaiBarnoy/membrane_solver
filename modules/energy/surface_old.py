# modules/surface.py
# Here goes energy functions relevant for area of facets

from collections import defaultdict
from typing import Dict

import numpy as np

from runtime.logging_config import setup_logging

logger = setup_logging("membrane_solver.log")


def calculate_surface_energy(mesh, global_params):
    """Compute the total surface energy for all facets."""
    surface_energy = 0.0
    for facet in mesh.facets.values():
        gamma = facet.options.get(
            "surface_tension", global_params.get("surface_tension")
        )
        area = facet.compute_area(mesh)
        surface_energy += gamma * area
    return surface_energy


def compute_energy_and_gradient(
    mesh, global_params, param_resolver, *, compute_gradient: bool = True
):
    E = 0.0
    grad: Dict[int, np.ndarray] | None = (
        defaultdict(lambda: np.zeros(3)) if compute_gradient else None
    )

    for facet in mesh.facets.values():
        # Retrieve the surface tension parameter (γ) for the facet
        surface_tension = param_resolver.get(facet, "surface_tension")
        if surface_tension is None:
            surface_tension = global_params.get("surface_tension")

        area = facet.compute_area(mesh)
        E += surface_tension * area

        if compute_gradient:
            area_gradient = facet.compute_area_gradient(mesh)
            for vertex_index, gradient_vector in area_gradient.items():
                grad[vertex_index] += surface_tension * gradient_vector

    # Log the computed energy and gradient
    logger.debug(f"Computed surface energy: {E}")
    logger.debug(f"Computed surface energy gradient: {grad}")

    if compute_gradient:
        return E, dict(grad)
    else:
        return E, {}
