# modules/constraints/fixed_volume.py

import numpy as np
from typing import Dict
from collections import defaultdict
from logging_config import setup_logging

logger = setup_logging('membrane_solver')

def compute_energy_and_gradient(mesh, global_params, param_resolver, *, compute_gradient: bool = True):
    """
    Compute Lagrange multiplier energy term for volume conservation:
        E = λ * (V - V₀)
    where:
    - V is current volume
    - V₀ is target volume  
    - λ is the Lagrange multiplier
    
    This implements hard constraints by including the Lagrange multiplier
    directly in the energy formulation, unlike the penalty method in volume.py
    """
    E = 0.0
    grad: Dict[int, np.ndarray] | None = (
        defaultdict(lambda: np.zeros(3)) if compute_gradient else None
    )
    
    for body in mesh.bodies.values():
        # Get Lagrange multiplier for this body
        if param_resolver is not None:
            λ = param_resolver.get(body, 'volume_multiplier')
        else:
            λ = None
        if λ is None:
            λ = global_params.get('volume_multiplier', 0.0)
            
        V0 = (body.target_volume
              if body.target_volume is not None
              else body.options.get('target_volume', 0))
              
        if V0 == 0:
            continue  # Skip if no target volume specified
            
        V = body.compute_volume(mesh)
        constraint_violation = V - V0
        
        # Lagrange multiplier energy term
        E += λ * constraint_violation
        
        if compute_gradient:
            # Gradient is λ * ∇V
            volume_gradient = body.compute_volume_gradient(mesh)
            for vertex_index, gradient_vector in volume_gradient.items():
                grad[vertex_index] += λ * gradient_vector
                
        logger.debug(f"Body {body.index}: V={V:.6f}, V0={V0:.6f}, lambda={λ:.6f}, constraint_violation={constraint_violation:.6f}")
    
    if compute_gradient:
        return E, dict(grad)
    else:
        return E, {}


def calculate_energy(mesh, global_params):
    """
    Legacy interface for energy calculation without gradients.
    """
    E, _ = compute_energy_and_gradient(mesh, global_params, None, compute_gradient=False)
    return E

