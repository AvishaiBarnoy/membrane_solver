# modules/volume.py

from collections import defaultdict
from typing import Dict

import numpy as np

from runtime.logging_config import setup_logging

logger = setup_logging("membrane_solver.log")

def calculate_volume_energy(mesh, global_params):
    """
    Compute volume energy as a soft quadratic penalty:
        E = 1/2 * k * (V - V₀)²
    where:
    - V is current volume
    - V₀ is target volume
    - k is stiffness (local or from global parameters)
    """
    mode = global_params.get("volume_constraint_mode", "lagrange")
    if mode != "penalty":
        return 0.0

    volume_energy = 0.0

    for body in mesh.bodies.values():
        V = body.compute_volume(mesh)
        V0 = (body.target_volume
              if body.target_volume is not None
              else body.options.get("target_volume", 0))
        k = body.options.get("volume_stiffness", global_params.volume_stiffness)

        volume_energy += 0.5 * k * (V - V0)**2

    return volume_energy


def compute_energy_and_gradient(mesh, global_params, param_resolver, *, compute_gradient: bool = True):
    """Volume energy and gradient."""

    mode = global_params.get("volume_constraint_mode", "lagrange")
    if mode != "penalty":
        return 0.0, {}

    E = 0.0
    grad: Dict[int, np.ndarray] | None = (
        defaultdict(lambda: np.zeros(3)) if compute_gradient else None
    )

    # Prepare positions and index_map if we can use batched operations
    positions = None
    index_map: Dict[int, int] | None = None
    if getattr(mesh, "facet_vertex_loops", None):
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row

    for body in mesh.bodies.values():
        k = param_resolver.get(body, 'volume_stiffness')
        if k is None:
            k = global_params.volume_stiffness

        V0 = (body.target_volume
              if body.target_volume is not None
              else body.options.get('target_volume', 0))

        if compute_gradient:
            V, volume_gradient = body.compute_volume_and_gradient(
                mesh, positions=positions, index_map=index_map
            )
        else:
            V = body.compute_volume(
                mesh, positions=positions, index_map=index_map
            )

        delta = V - V0
        E += 0.5 * k * delta ** 2

        if compute_gradient:
            factor = k * delta
            for vertex_index, gradient_vector in volume_gradient.items():
                grad[vertex_index] += factor * gradient_vector

    if compute_gradient:
        return E, dict(grad)
    else:
        return E, {}
