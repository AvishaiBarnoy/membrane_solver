# modules/surface.py
# Here goes energy functions relevant for area of facets

from geometry.entities import Mesh, Facet
from typing import Dict
from collections import defaultdict
from logging_config import setup_logging
import numpy as np

logger = setup_logging('membrane_solver')

def calculate_surface_energy(mesh: Mesh, global_params) -> float:
    """Compute the total surface energy for all facets.

    This is the energy-only path, so we batch over facets using the cached
    vertex loops and a single positions array where possible.
    """
    if not mesh.facets:
        return 0.0

    # Fast path: all facets have cached vertex loops and are triangles.
    # We can compute all triangle areas in one vectorized pass.
    if getattr(mesh, "facet_vertex_loops", None) and all(
        len(mesh.facet_vertex_loops.get(f.index, [])) == 3
        for f in mesh.facets.values()
    ):
        positions = mesh.positions_view()
        gammas = []
        v0 = []
        v1 = []
        v2 = []
        index_map = mesh.vertex_index_to_row

        for facet in mesh.facets.values():
            loop = mesh.facet_vertex_loops[facet.index]
            i0, i1, i2 = (index_map[int(loop[0])],
                          index_map[int(loop[1])],
                          index_map[int(loop[2])])
            v0.append(i0)
            v1.append(i1)
            v2.append(i2)
            gammas.append(
                facet.options.get(
                    "surface_tension",
                    global_params.get("surface_tension"),
                )
            )

        v0 = positions[np.array(v0, dtype=int)]
        v1 = positions[np.array(v1, dtype=int)]
        v2 = positions[np.array(v2, dtype=int)]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        return float(np.dot(np.asarray(gammas, dtype=float), areas))

    # Fallback: per-facet loop if we cannot batch.
    gammas = []
    areas = []
    for facet in mesh.facets.values():
        gammas.append(
            facet.options.get(
                "surface_tension",
                global_params.get("surface_tension"),
            )
        )
        areas.append(facet.compute_area(mesh))
    if not gammas:
        return 0.0
    return float(np.dot(np.asarray(gammas, dtype=float),
                        np.asarray(areas, dtype=float)))

def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> tuple[float, Dict[int, np.ndarray]]:
    """Compute surface energy and optionally its gradient.

    Parameters
    ----------
    mesh : Mesh
        The mesh containing facets.
    global_params : GlobalParameters
        Global parameter store with defaults.
    param_resolver : ParameterResolver
        Resolver to obtain per-object parameters.
    compute_gradient : bool, optional
        If ``True`` also compute the gradient, by default ``True``.

    Returns
    -------
    tuple[float, Dict[int, np.ndarray]]
        Total surface energy and gradient per vertex.
    """

    E = 0.0

    if compute_gradient:
        vidxs = list(mesh.vertices.keys())
        idx_map = {v: i for i, v in enumerate(vidxs)}
        grad_arr = np.zeros((len(vidxs), 3))
    else:
        idx_map = {}
        grad_arr = None

    positions = mesh.positions_view()

    # Per-facet computation; we still reuse the shared positions array to
    # avoid rebuilding small arrays, but we rely on the established
    # compute_area_and_gradient implementation for correctness.
    for facet in mesh.facets.values():
        surface_tension = param_resolver.get(facet, "surface_tension")
        if surface_tension is None:
            surface_tension = global_params.get("surface_tension")

        if compute_gradient:
            area, area_gradient = facet.compute_area_and_gradient(
                mesh, positions=positions, index_map=mesh.vertex_index_to_row
            )
        else:
            area = facet.compute_area(mesh)
        E += surface_tension * area

        if compute_gradient:
            for vertex_index, gradient_vector in area_gradient.items():
                grad_arr[idx_map[vertex_index]] += surface_tension * gradient_vector

    if compute_gradient:
        grad = {v: grad_arr[i] for v, i in idx_map.items()}
        logger.debug(f"Computed surface energy: {E}")
        logger.debug(f"Computed surface energy gradient: {grad}")
        return E, grad
    else:
        logger.debug(f"Computed surface energy: {E}")
        return E, {}
