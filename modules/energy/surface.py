# modules/surface.py
# Here goes energy functions relevant for area of facets

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross
from runtime.logging_config import setup_logging

logger = setup_logging("membrane_solver.log")


def _all_facets_are_triangles(mesh: Mesh) -> bool:
    """Return ``True`` if facet loops exist and all are triangles."""
    if not getattr(mesh, "facet_vertex_loops", None):
        return False
    tri_rows, tri_facets = mesh.triangle_row_cache()
    return tri_rows is not None and len(tri_facets) == len(mesh.facets)


def calculate_surface_energy(mesh: Mesh, global_params) -> float:
    """Compute the total surface energy for all facets.

    This is the energy-only path, so we batch over facets using the cached
    vertex loops and a single positions array where possible.
    """
    if not mesh.facets:
        return 0.0

    # Fast path: all facets are triangles with cached vertex loops.
    if _all_facets_are_triangles(mesh):
        positions = mesh.positions_view()
        tri_rows, tri_facets = mesh.triangle_row_cache()
        gammas = np.empty(len(tri_facets), dtype=float)

        for idx, fid in enumerate(tri_facets):
            facet = mesh.facets[fid]
            gammas[idx] = facet.options.get(
                "surface_tension",
                global_params.get("surface_tension"),
            )

        tri_pos = positions[tri_rows]  # (n_facets, 3, 3)
        v0 = tri_pos[:, 0, :]
        v1 = tri_pos[:, 1, :]
        v2 = tri_pos[:, 2, :]
        cross = _fast_cross(v1 - v0, v2 - v0)
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
    return float(
        np.dot(np.asarray(gammas, dtype=float), np.asarray(areas, dtype=float))
    )


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

    # Fast energy‑only path: reuse the highly vectorised surface energy
    # routine instead of looping per facet.
    if not compute_gradient:
        E = calculate_surface_energy(mesh, global_params)
        logger.debug(f"Computed surface energy (energy‑only path): {E}")
        return E, {}

    E = 0.0

    # If we have a pure triangle mesh with cached vertex loops, use a fully
    # batched path that mirrors :func:`calculate_surface_energy` but also
    # accumulates gradients in a single vectorised pass.
    if _all_facets_are_triangles(mesh):
        E, grad = _batched_surface_energy_and_gradient_triangles(
            mesh, global_params, param_resolver
        )
        return E, grad

    # Fallback: per-facet computation using shared position array.
    positions = mesh.positions_view()
    vertex_ids = mesh.vertex_ids
    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros((len(vertex_ids), 3))

    for facet in mesh.facets.values():
        surface_tension = param_resolver.get(facet, "surface_tension")
        if surface_tension is None:
            surface_tension = global_params.get("surface_tension")

        area, area_gradient = facet.compute_area_and_gradient(
            mesh, positions=positions, index_map=idx_map
        )
        E += surface_tension * area

        for vertex_index, gradient_vector in area_gradient.items():
            row = idx_map[vertex_index]
            grad_arr[row] += surface_tension * gradient_vector

    grad = {vid: grad_arr[row] for vid, row in idx_map.items()}
    # Avoid constructing large string representations of gradients unless
    # debug logging is actually enabled.
    if logger.isEnabledFor(10):
        logger.debug("Computed surface energy: %.6f", E)
    return E, grad


def _batched_surface_energy_and_gradient_triangles(
    mesh: Mesh,
    global_params,
    param_resolver,
) -> Tuple[float, Dict[int, np.ndarray]]:
    """Vectorised surface energy and gradient for pure triangle meshes.

    Assumes that ``mesh.facet_vertex_loops`` is populated and that every
    facet loop has length three.
    """
    positions = mesh.positions_view()
    tri_rows_arr, tri_facets = mesh.triangle_row_cache()

    gammas_arr = np.empty(len(tri_facets), dtype=float)

    for idx, fid in enumerate(tri_facets):
        facet = mesh.facets[fid]
        surface_tension = param_resolver.get(facet, "surface_tension")
        if surface_tension is None:
            surface_tension = global_params.get("surface_tension")
        gammas_arr[idx] = surface_tension

    tri_pos = positions[tri_rows_arr]  # (nF, 3, 3)
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]

    e1 = v1 - v0
    e2 = v2 - v0
    n = _fast_cross(e1, e2)
    A = np.linalg.norm(n, axis=1)
    mask = A >= 1e-12
    if not np.any(mask):
        # Degenerate mesh; return zero energy/gradient.
        grad_zero = {vid: np.zeros(3) for vid in mesh.vertices.keys()}
        return 0.0, grad_zero

    n_hat = n[mask] / A[mask][:, None]
    areas = 0.5 * A[mask]
    gammas_masked = gammas_arr[mask]

    # Energy: sum_i gamma_i * area_i
    E = float(np.dot(gammas_masked, areas))

    # Gradient per triangle, following the same formulas as
    # Facet.compute_area_and_gradient specialised to the triangle case.
    v0m = v0[mask]
    v1m = v1[mask]
    v2m = v2[mask]

    g0 = 0.5 * _fast_cross(v1m - v0m, n_hat)  # dA/dv0
    g1 = 0.5 * _fast_cross(v2m - v1m, n_hat)  # dA/dv1
    g2 = 0.5 * _fast_cross(v0m - v2m, n_hat)  # dA/dv2

    # Scale by per-facet surface tension.
    scale = gammas_masked[:, None]
    g0 *= scale
    g1 *= scale
    g2 *= scale

    # Accumulate into per-vertex gradient array.
    n_vertices = len(mesh.vertex_ids)
    grad_arr = np.zeros((n_vertices, 3), dtype=float)
    tri_rows_masked = tri_rows_arr[mask]

    i0 = tri_rows_masked[:, 0]
    i1 = tri_rows_masked[:, 1]
    i2 = tri_rows_masked[:, 2]

    np.add.at(grad_arr, i0, g0)
    np.add.at(grad_arr, i1, g1)
    np.add.at(grad_arr, i2, g2)

    vertex_ids = mesh.vertex_ids
    grad = {vid: grad_arr[i] for i, vid in enumerate(vertex_ids)}
    return E, grad
