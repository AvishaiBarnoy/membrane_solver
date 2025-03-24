# volume.py
# Here goes energy terms relevant for volume of defined bodies


# Pseudo code for volume difference calculation
#   energy should be spread over all relevant facets, scaling by the surface
#   area of the facets.
# if target_volume:
#   if current_vol != target_volume:
#       total_energy += k_vol * (current_volume - target_volume)

import numpy as np
from geometry_entities import Body
from logging_config import setup_logging

logger = setup_logging('membrane_solver')

def energy(vertices, facets, global_params, body):
    """
    takes vertices, facets, and body objects and returns the energy the volume
    energye which is the deviation from the target volume
    """
    target_volume = body["target_volume"][0]
    volume_stiffness = global_params.get("volume_stiffness", 1000)

    temp_body = Body(vertices, facets)
    current_volume = temp_body.calculate_volum()

    volume_energy = 0.5 * volume_stiffness * (current_volume
                                              - target_volume)**2

    logger.info(f"[volume.py] Current volume: {current_volume}")
    logger.info(f"[volume.py] Target volume: {target_volume}")
    logger.info(f"[volume.py] Volume energy contribution: {volume_energy}")

    return volume_energy
