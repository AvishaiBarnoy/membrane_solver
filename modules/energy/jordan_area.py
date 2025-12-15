"""Jordan-area penalty energy for planar boundary loops.

This module adds a soft penalty based on the planar area enclosed by the
mesh boundary, using the polygon (shoelace) formula in the xy-plane:

    A_J = 1/2 * sum_i (x_i y_{i+1} - x_{i+1} y_i)

The energy for a single boundary component is

    E = 1/2 * k * (|A_J| - A_target)^2

where ``k`` is a stiffness parameter and ``A_target`` is the desired
Jordan area. Both are taken from global parameters:

    - ``jordan_target_area`` (float, required for this module to act)
    - ``jordan_stiffness`` (float, default 0.0)

The gradient is constructed with respect to boundary vertex positions
only, and is zeroed for interior vertices. This module assumes that:

    - the mesh is effectively planar (e.g. enforced via a plane
      constraint), and
    - there is a single simple boundary loop. If multiple boundary
      components are present, only the outermost loop is used and a
      warning is logged.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from geometry.entities import Mesh
from logging_config import setup_logging

logger = setup_logging("membrane_solver.log")


def _boundary_edges(mesh: Mesh) -> List[int]:
    """Return indices of edges that lie on the boundary (single-adjacent facets)."""
    mesh.build_connectivity_maps()
    return [eid for eid, facets in mesh.edge_to_facets.items() if len(facets) == 1]


def _build_boundary_loop(mesh: Mesh, edge_ids: List[int]) -> List[int]:
    """Construct an ordered vertex loop from a set of boundary edges.

    Parameters
    ----------
    mesh : Mesh
        The mesh whose connectivity is used.
    edge_ids : list[int]
        Indices of edges that form a single closed boundary component.

    Returns
    -------
    list[int]
        Ordered list of vertex indices forming a closed loop.
    """
    if not edge_ids:
        return []

    remaining = set(edge_ids)
    # Start from an arbitrary edge; orient tail->head.
    first_eid = next(iter(remaining))
    first_edge = mesh.edges[first_eid]
    loop: List[int] = [first_edge.tail_index, first_edge.head_index]
    remaining.remove(first_eid)

    current = first_edge.head_index

    while remaining:
        # Find an edge incident to the current vertex.
        next_eid = None
        next_head = None
        for eid in list(remaining):
            e = mesh.edges[eid]
            if e.tail_index == current:
                next_eid = eid
                next_head = e.head_index
                break
            if e.head_index == current:
                next_eid = eid
                next_head = e.tail_index
                break
        if next_eid is None:
            logger.warning(
                "Jordan area: failed to build closed boundary loop; boundary "
                "may be disconnected or non-manifold."
            )
            break
        remaining.remove(next_eid)
        current = next_head
        if current == loop[0]:
            # Closed the loop.
            break
        loop.append(current)

    return loop


def _jordan_area_and_gradient(
    mesh: Mesh, loop: List[int]
) -> Tuple[float, Dict[int, np.ndarray]]:
    """Compute planar Jordan area and gradient for a vertex loop.

    The loop is interpreted in the xy-plane; z-components of vertices are
    ignored. The gradient is returned as a dictionary mapping vertex
    indices to (dx, dy, dz) contributions.
    """
    n = len(loop)
    grad: Dict[int, np.ndarray] = {vid: np.zeros(3, dtype=float) for vid in loop}

    if n < 3:
        return 0.0, grad

    xs = np.array([mesh.vertices[vid].position[0] for vid in loop], dtype=float)
    ys = np.array([mesh.vertices[vid].position[1] for vid in loop], dtype=float)

    xs_next = np.roll(xs, -1)
    ys_next = np.roll(ys, -1)

    area = 0.5 * float(np.dot(xs, ys_next) - np.dot(xs_next, ys))

    # Gradient of A = 0.5 * sum_i (x_i y_{i+1} - x_{i+1} y_i)
    # dA/dx_i = 0.5 * (y_{i+1} - y_{i-1})
    # dA/dy_i = 0.5 * (x_{i-1} - x_{i+1})
    ys_prev = np.roll(ys, 1)
    xs_prev = np.roll(xs, 1)

    dA_dx = 0.5 * (ys_next - ys_prev)
    dA_dy = 0.5 * (xs_prev - xs_next)

    for idx, vid in enumerate(loop):
        g = grad[vid]
        g[0] = dA_dx[idx]
        g[1] = dA_dy[idx]
        # z-gradient remains zero for planar area.

    return area, grad


def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> tuple[float, Dict[int, np.ndarray]]:
    """Jordan-area penalty energy and gradient for planar meshes."""
    target = global_params.get("jordan_target_area")
    if target is None:
        return 0.0, {}

    stiffness = float(global_params.get("jordan_stiffness", 0.0) or 0.0)
    if stiffness == 0.0:
        return 0.0, {}

    b_edges = _boundary_edges(mesh)
    if not b_edges:
        logger.debug("Jordan area: no boundary edges detected; skipping.")
        return 0.0, {}

    loop = _build_boundary_loop(mesh, b_edges)
    if len(loop) < 3:
        logger.debug("Jordan area: boundary loop too short; skipping.")
        return 0.0, {}

    area, gradA = _jordan_area_and_gradient(mesh, loop)

    # Use absolute area so orientation (CW/CCW) does not matter.
    sign = 1.0 if area >= 0.0 else -1.0
    area_eff = sign * area
    delta = area_eff - float(target)

    energy = 0.5 * stiffness * delta * delta

    if not compute_gradient:
        return float(energy), {}

    grad: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=float))
    factor = stiffness * delta * sign
    for vid, gvec in gradA.items():
        grad[vid] += factor * gvec

    return float(energy), dict(grad)


__all__ = ["compute_energy_and_gradient"]
