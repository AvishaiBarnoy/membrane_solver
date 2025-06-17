# modules/volume.py

from geometry.entities import Body
import numpy as np
from logging_config import setup_logging

logger = setup_logging('membrane_solver')

def calculate_volume_energy(mesh, global_params):
    """
    Compute volume energy as:
        E = 1/2 * k * (V - V₀)²
    where:
    - V is current volume
    - V₀ is target volume
    - k is stiffness (local or from global parameters)
    """
    volume_energy = 0.0

    for body in mesh.bodies.values():
        V = body.compute_volume(mesh)
        V0 = (body.target_volume
              if body.target_volume is not None
              else body.options.get("target_volume", 0))
        logger.debug(f"Default target_volume is 0!")
        k = body.options.get("volume_stiffness", global_params.volume_stiffness)

        volume_energy += 0.5 * k * (V - V0)**2

    #logger.info(f"[volume.py] Current volume: {current_volume}")
    #logger.info(f"[volume.py] Target volume: {target_volume}")
    #logger.info(f"[volume.py] Volume energy contribution: {volume_energy}")

    return volume_energy

def compute_energy_and_gradient(mesh, global_params, param_resolver):
    E = 0.0
    grad: Dict[int,np.ndarray] = {i: np.zeros(3) for i in mesh.vertices}
    for body in mesh.bodies.values():
        #print(f"############# volume energy and gradient ############")
        k = param_resolver.get(body, 'volume_stiffness')
        if k is None:
            k = global_params.volume_stiffness

        #V0 = body.options.get("target_volume", 0)
        #print(f"[DEBUG] Body entity: {body}")
        V0 = (body.target_volume
              if body.target_volume is not None
              else body.options.get("target_volume", 0))
        #print(f"[DEBUG] V0 {V0}")
        V = body.compute_volume(mesh)

        # Energy: 0.5 * k * (V - V0)**2
        E += 0.5 * k * (V - V0)**2
        # Gradient: k * (V - V0) * ∇V
        volume_gradient = body.compute_volume_gradient(mesh)
        for vertex_index, gradient_vector in volume_gradient.items():
            # Add the gradient contribution to the total gradient
            grad[vertex_index] += k * (V - V0) * volume_gradient[vertex_index]
        # Log the computed energy and gradient
        #logger.info(f"Computed volume energy: {E}")
        #logger.info(f"Computed volume energy gradient: {grad}")

    # Return the total energy and gradient
    return E, grad
