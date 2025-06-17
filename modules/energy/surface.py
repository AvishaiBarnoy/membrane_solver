# modules/surface.py
# Here goes energy functions relevant for area of facets

from geometry.entities import Mesh, Facet
from logging_config import setup_logging
import numpy as np

logger = setup_logging('membrane_solver')

def calculate_surface_energy(facet: Facet, mesh: Mesh, global_params):
    """Return the surface energy of a single facet.

    The energy is ``γ * A`` where ``γ`` is the surface tension.  ``γ`` can be
    specified per facet via ``facet.options['surface_tension']`` or defaults to
    ``global_params.surface_tension``.
    """
    gamma = facet.options.get("surface_tension", global_params.surface_tension)

    # Calculate area of the facet
    area = facet.compute_area(mesh)

    return gamma * area


def compute_energy_and_gradient(mesh, global_params, param_resolver):
    E = 0.0
    grad: Dict[int,np.ndarray] = {i: np.zeros(3) for i in mesh.vertices}

    for facet in mesh.facets.values():
        # Retrieve the surface tension parameter (γ) for the facet
        surface_tension = param_resolver.get(facet, 'surface_tension')
        # If not defined, use the global parameter
        if surface_tension is None:
            surface_tension = global_params.surface_tension

        # Compute facet area
        area = facet.compute_area(mesh)

        # compute area A, then E += γ*A
        E += surface_tension* area

        # compute ∇A wrt each vertex in facet → add to grad
        area_gradient = facet.compute_area_gradient(mesh)
        for vertex_index, gradient_vector in area_gradient.items():
            # Add the gradient contribution to the total gradient
            grad[vertex_index] += surface_tension * area_gradient[vertex_index]
    # Log the computed energy and gradient
    logger.debug(f"Computed surface energy: {E}")
    logger.debug(f"Computed surface energy gradient: {grad}")
    # Return the total energy and gradient
    # Note: grad is a dictionary where keys are vertex indices and values are gradients
    # (3D vectors)
    return E, grad
