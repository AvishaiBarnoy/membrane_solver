# volume.py

from geometry_entities import Body
from logging_config import setup_logging

logger = setup_logging('membrane_solver')

def calculate_volume_energy(body, global_params):
    """
    Compute volume energy as:
        E = 1/2 * k * (V - V₀)²
    where:
    - V is current volume
    - V₀ is target volume
    - k is stiffness (local or from global parameters)
    """
    V = body.calculate_volume()
    V0 = body.options.get("target_volume", 0)
    logger.debut(f"Default target_volume is 0!")
    k = body.options.get("volume_stiffness", global_params.volume_stiffness)

    volume_energy = 0.5 * k * (V - V0)**2

    logger.info(f"[volume.py] Current volume: {current_volume}")
    logger.info(f"[volume.py] Target volume: {target_volume}")
    logger.info(f"[volume.py] Volume energy contribution: {volume_energy}")

    return volume_energy
