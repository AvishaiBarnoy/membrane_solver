# modules/volume.py

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

from geometry.entities import Body, _fast_cross
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
        logger.debug("Default target_volume is 0!")
        k = body.options.get("volume_stiffness", global_params.volume_stiffness)

        volume_energy += 0.5 * k * (V - V0)**2

    #logger.info(f"[volume.py] Current volume: {current_volume}")
    #logger.info(f"[volume.py] Target volume: {target_volume}")
    #logger.info(f"[volume.py] Volume energy contribution: {volume_energy}")

    return volume_energy


def _body_volume_batched(mesh, body: Body, positions: np.ndarray, index_map: Dict[int, int]) -> float:
    """Compute body volume by batching over its triangular facets.

    Assumes each facet in ``body.facet_indices`` has a cached vertex loop of
    length = 3 in ``mesh.facet_vertex_loops``.
    """
    n_facets = len(body.facet_indices)
    tri_indices = np.empty((n_facets, 3), dtype=int)
    valid_count = 0

    for facet_idx in body.facet_indices:
        loop = mesh.facet_vertex_loops.get(facet_idx)
        if loop is None or len(loop) != 3:
            # Fallback will be used by caller if any facet is not a triangle.
            return body.compute_volume(mesh)

        tri_indices[valid_count, 0] = index_map[int(loop[0])]
        tri_indices[valid_count, 1] = index_map[int(loop[1])]
        tri_indices[valid_count, 2] = index_map[int(loop[2])]
        valid_count += 1

    if valid_count == 0:
        return 0.0

    # If valid_count < n_facets (shouldn't happen given the check above), slice.
    tri_indices = tri_indices[:valid_count]

    v0 = positions[tri_indices[:, 0]]
    v1 = positions[tri_indices[:, 1]]
    v2 = positions[tri_indices[:, 2]]

    # For each triangle, volume contribution is dot(cross(v1, v2), v0) / 6
    cross = _fast_cross(v1, v2)
    vol_contrib = np.einsum("ij,ij->i", cross, v0)
    return float(vol_contrib.sum() / 6.0)


def _batched_volume_and_gradient(mesh, body: Body, positions: np.ndarray, index_map: Dict[int, int]) -> Tuple[float, Dict[int, np.ndarray]]:
    """Compute volume and gradient for a body using vectorized operations on triangles.

    Returns:
        Tuple of (volume, gradient_dict)
    """
    n_facets = len(body.facet_indices)
    tri_indices = np.empty((n_facets, 3), dtype=int)
    valid_count = 0

    # Collect indices
    for facet_idx in body.facet_indices:
        loop = mesh.facet_vertex_loops.get(facet_idx)
        if loop is None or len(loop) != 3:
            # Fallback for non-triangular facets
            return body.compute_volume_and_gradient(mesh)

        tri_indices[valid_count, 0] = index_map[int(loop[0])]
        tri_indices[valid_count, 1] = index_map[int(loop[1])]
        tri_indices[valid_count, 2] = index_map[int(loop[2])]
        valid_count += 1

    if valid_count == 0:
        return 0.0, {}

    v0 = positions[tri_indices[:, 0]]
    v1 = positions[tri_indices[:, 1]]
    v2 = positions[tri_indices[:, 2]]

    # Volume: sum(dot(cross(v1, v2), v0)) / 6
    cross_v1_v2 = _fast_cross(v1, v2)
    vol_contrib = np.einsum("ij,ij->i", cross_v1_v2, v0)
    volume = float(vol_contrib.sum() / 6.0)

    # Gradients
    # grad(v0) = (v1 x v2) / 6
    # grad(v1) = (v2 x v0) / 6
    # grad(v2) = (v0 x v1) / 6

    g0 = cross_v1_v2 / 6.0
    g1 = _fast_cross(v2, v0) / 6.0
    g2 = _fast_cross(v0, v1) / 6.0

    # Accumulate into per-vertex gradient array
    n_vertices = len(mesh.vertex_ids)
    grad_arr = np.zeros((n_vertices, 3), dtype=float)

    i0 = tri_indices[:, 0]
    i1 = tri_indices[:, 1]
    i2 = tri_indices[:, 2]

    np.add.at(grad_arr, i0, g0)
    np.add.at(grad_arr, i1, g1)
    np.add.at(grad_arr, i2, g2)

    # Convert to dict
    grad = {vid: grad_arr[row] for vid, row in index_map.items() if np.any(grad_arr[row])}

    return volume, grad


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
            if positions is not None and index_map is not None:
                V, volume_gradient = _batched_volume_and_gradient(mesh, body, positions, index_map)
            else:
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
