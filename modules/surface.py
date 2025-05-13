# modules/surface.py
# Here goes energy functions relevant for area of facets

from geometry.entities import Mesh
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


def compute_energy_and_gradient(mesh, global_params, param_resolver):
    E = 0.0
    grad: Dict[int,np.ndarray] = {i: np.zeros(3) for i in mesh.vertices}

    for facet in mesh.facets.values():
        g = param_resolver.get(facet, 'surface_tension')
        # compute area A, then E += γ*A
        E += g * facet.compute_area(mesh)
        # compute ∇A wrt each vertex in facet → add to grad
        ...
    return E, grad
