# area.py
# Here goes energy functions relevant for area of facets

from geometry_entities import Facet
from logging_config import setup_logging

logger = setup_logging('membrane_solver')

def total_surface_energy(facets, body, global_params):
    """
    Calculate total surface energy as sum of surface tension * area over all facets.

    Parameters:
    - facets: list of Facet objects
    - global_params: dict, must include "surface tension"

    Returns:
    - float: total surface energy
    """
    global_tension = global_params.get("surface tension", 1.0)
    total_energy = 0.0

    for facet in facets:
        tension = facet.options.get("surface tension", global_tension)
        area = facet.calculate_area()
        total_energy += surface_tension * area

    logger.debug(f"Calculated total surface energy: {total_energy}")

    return surface_energy

def calculate_surface_energy(facet, global_params):
    # TODO: add documentation
    gamma = facet.options.get("surface_tension", global_params.surface_tension)

    # Calculate area
    area = facet.calculate_area()

    # Compute surface energy
    surface_energy = gamma * area

    return surface_energy

