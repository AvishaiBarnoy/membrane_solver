# volume.py
# Here goes energy terms relevant for volume of defined bodies

# TODO: check if this pseudo-code is needed
# Pseudo code for volume difference calculation
#   energy should be spread over all relevant facets, scaling by the surface
#   area of the facets.
# if target_volume:
#   if current_vol != target_volume:
#       total_energy += k_vol * (current_volume - target_volume)

from geometry_entities import Body
from logging_config import setup_logging

logger = setup_logging('membrane_solver')

def calculate_volume_energy(facets, body, global_params):
    """
    takes a body object and returns the volume energy
    energy which is the deviation from the target volume
    """
    target_volume = body.options.get("target_volume")
    stiffness = global_params.get("volume_stiffness",
                                  global_params.volume_stiffness)

    current_volume = body.calculate_volume()

    volume_energy = 0.5 * volume_stiffness * (current_volume - target_volume)**2

    logger.info(f"[volume.py] Current volume: {current_volume}")
    logger.info(f"[volume.py] Target volume: {target_volume}")
    logger.info(f"[volume.py] Volume energy contribution: {volume_energy}")

    return volume_energy
