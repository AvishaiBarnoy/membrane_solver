# area.py
# Here goes energy functions relevant for area of facets

from geometry_entities import Facet
from logging_config import setup_logging

logger = setup_logging('membrane_solver')

def calculate_surface_energy(facet, global_params):
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

