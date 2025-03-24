# area.py
# Here goes energy functions relevant for area of facets

from geometry_entities import Facet
from logging_config import setup_logging

logger = setup_logging('membrane_solver')

def energy(facets, body, global_params):
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

    logger.debug(f"Calculated surface energy: {total_energy}")

    return surface_energy
