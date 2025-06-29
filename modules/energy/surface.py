# modules/surface.py
# Here goes energy functions relevant for area of facets

from geometry.entities import Mesh, Facet
from typing import Dict
from logging_config import setup_logging
import numpy as np

logger = setup_logging('membrane_solver')

def calculate_surface_energy(mesh, global_params):
    """
    Compute surface energy as gamma * area.
    - `gamma` can be defined locally in facet.options or defaults to global.
    """
    gamma = facet.options.get("surface_tension", global_params.surface_tension)

    # Calculate area
    area = facet.calculate_area()

    # Compute surface energy
    surface_energy = gamma * area
    return surface_energy

def compute_energy_and_gradient(mesh, global_params, param_resolver, *, compute_gradient: bool = True):
    E = 0.0
    grad: Dict[int, np.ndarray] | None = {i: np.zeros(3) for i in mesh.vertices} if compute_gradient else None

    for facet in mesh.facets.values():
        # Retrieve the surface tension parameter (γ) for the facet
        surface_tension = param_resolver.get(facet, 'surface_tension')
        if surface_tension is None:
            surface_tension = global_params.surface_tension

        area = facet.compute_area(mesh)
        E += surface_tension * area

        if compute_gradient:
            area_gradient = facet.compute_area_gradient(mesh)
            for vertex_index, gradient_vector in area_gradient.items():
                grad[vertex_index] += surface_tension * area_gradient[vertex_index]

    # Log the computed energy and gradient
    logger.debug(f"Computed surface energy: {E}")
    logger.debug(f"Computed surface energy gradient: {grad}")

    if compute_gradient:
        return E, grad
    else:
        return E, {}
