# modules/volume.py

from geometry.entities import Body
from typing import Dict
from collections import defaultdict
import numpy as np
from logging_config import setup_logging

logger = setup_logging('membrane_solver')

def calculate_volume_energy(mesh, global_params):
    """
    Compute volume energy as a soft quadratic penalty:
        E = 1/2 * k * (V - V₀)²
    where:
    - V is current volume
    - V₀ is target volume
    - k is stiffness (local or from global parameters)
    """
    # In "lagrange" mode volume is enforced as a hard constraint via the
    # optimizer, so this energy term is disabled by default. The legacy
    # penalty path is still available for tests or explicit configurations.
    mode = global_params.get("volume_constraint_mode", "lagrange")
    if mode != "penalty":
        return 0.0

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


def _body_volume_batched(mesh, body: Body, positions: np.ndarray, index_map: Dict[int, int]) -> float:
    """Compute body volume by batching over its triangular facets.

    Assumes each facet in ``body.facet_indices`` has a cached vertex loop of
    length 3 in ``mesh.facet_vertex_loops``.
    """
    tri_v0 = []
    tri_v1 = []
    tri_v2 = []

    for facet_idx in body.facet_indices:
        loop = mesh.facet_vertex_loops.get(facet_idx)
        if loop is None or len(loop) != 3:
            # Fallback will be used by caller if any facet is not a triangle.
            return body.compute_volume(mesh)

        i0 = index_map[int(loop[0])]
        i1 = index_map[int(loop[1])]
        i2 = index_map[int(loop[2])]
        tri_v0.append(i0)
        tri_v1.append(i1)
        tri_v2.append(i2)

    if not tri_v0:
        return 0.0

    v0 = positions[np.array(tri_v0, dtype=int)]
    v1 = positions[np.array(tri_v1, dtype=int)]
    v2 = positions[np.array(tri_v2, dtype=int)]

    # For each triangle, volume contribution is dot(cross(v1, v2), v0) / 6
    cross = np.cross(v1, v2)
    vol_contrib = np.einsum("ij,ij->i", cross, v0)
    return float(vol_contrib.sum() / 6.0)

def compute_energy_and_gradient(mesh, global_params, param_resolver, *, compute_gradient: bool = True):
    """Volume energy and gradient.

    In the default ``\"lagrange\"`` mode, volume is treated as a hard
    constraint integrated into the optimizer, so this module returns zero
    energy and gradient. When ``global_params.volume_constraint_mode`` is
    set to ``\"penalty\"``, it reverts to the classic quadratic penalty
    behaviour used in the original implementation and tests.
    """

    mode = global_params.get("volume_constraint_mode", "lagrange")
    if mode != "penalty":
        # Hard‑constraint mode: no explicit volume energy term.
        return 0.0, {}

    E = 0.0
    grad: Dict[int, np.ndarray] | None = (
        defaultdict(lambda: np.zeros(3)) if compute_gradient else None
    )

    # For energy-only path we can batch volume over triangles using cached loops.
    positions = None
    index_map: Dict[int, int] | None = None
    if not compute_gradient and getattr(mesh, "facet_vertex_loops", None):
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
            V, volume_gradient = body.compute_volume_and_gradient(mesh)
        else:
            # Try batched triangle volume if we have positions and loops.
            if positions is not None and index_map is not None:
                V = _body_volume_batched(mesh, body, positions, index_map)
            else:
                V = body.compute_volume(mesh)

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
