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

    # Fast path: all facets are cached triangles -> batch energy + gradient.
    if compute_gradient and getattr(mesh, "facet_vertex_loops", None) and all(
        len(mesh.facet_vertex_loops.get(f.index, [])) == 3 for f in mesh.facets.values()
    ):
        index_map_rows = mesh.vertex_index_to_row

        # Collect per-facet vertex rows and surface tensions.
        i0_list = []
        i1_list = []
        i2_list = []
        gammas = []

        for facet in mesh.facets.values():
            loop = mesh.facet_vertex_loops[facet.index]
            i0_list.append(index_map_rows[int(loop[0])])
            i1_list.append(index_map_rows[int(loop[1])])
            i2_list.append(index_map_rows[int(loop[2])])

            gamma = param_resolver.get(facet, "surface_tension")
            if gamma is None:
                gamma = global_params.get("surface_tension")
            gammas.append(gamma)

        i0 = np.array(i0_list, dtype=int)
        i1 = np.array(i1_list, dtype=int)
        i2 = np.array(i2_list, dtype=int)
        gammas = np.asarray(gammas, dtype=float)

        v0 = positions[i0]
        v1 = positions[i1]
        v2 = positions[i2]

        # Triangle area: 0.5 * ||(v1 - v0) x (v2 - v0)||
        ba = v1 - v0
        ca = v2 - v0
        n = np.cross(ba, ca)
        A = np.linalg.norm(n, axis=1)

        # Avoid division by zero for degenerate triangles.
        mask = A >= 1e-12
        if not np.any(mask):
            return 0.0, {v: np.zeros(3) for v in vidxs}

        n_hat = np.zeros_like(n)
        n_hat[mask] = n[mask] / A[mask][:, None]

        # Energy
        E = float(np.dot(gammas[mask], 0.5 * A[mask]))

        # Gradient per vertex using analytic formulas:
        # g0 = -0.5 * (n_hat x (v2 - v1))
        # g1 =  0.5 * (n_hat x (v2 - v0))
        # g2 = -0.5 * (n_hat x (v1 - v0))
        g0 = -0.5 * np.cross(n_hat[mask], v2[mask] - v1[mask])
        g1 =  0.5 * np.cross(n_hat[mask], v2[mask] - v0[mask])
        g2 = -0.5 * np.cross(n_hat[mask], v1[mask] - v0[mask])

        # Scale by surface tension per facet.
        gamma_col = gammas[mask][:, None]
        g0 *= gamma_col
        g1 *= gamma_col
        g2 *= gamma_col

        # Accumulate into per-vertex gradient array.
        active_i0 = i0[mask]
        active_i1 = i1[mask]
        active_i2 = i2[mask]

        np.add.at(grad_arr, active_i0, g0)
        np.add.at(grad_arr, active_i1, g1)
        np.add.at(grad_arr, active_i2, g2)

    else:
        # General path (non-batched or no caches): per-facet computation.
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
