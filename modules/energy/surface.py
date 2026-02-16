# modules/surface.py
# Here goes energy functions relevant for area of facets

import logging
import os
from typing import Dict

import numpy as np

from fortran_kernels.loader import get_surface_energy_kernel
from geometry.entities import Mesh, _fast_cross

logger = logging.getLogger("membrane_solver")


def _call_fortran_surface_kernel(
    kernel,
    *,
    nv: int,
    nf: int,
    pos_f: np.ndarray,
    tri_f: np.ndarray,
    gamma: np.ndarray,
    grad_f: np.ndarray,
    zero_based: int,
) -> float:
    """Call the Fortran kernel across f2py signature variants."""
    try:
        # Canonical f2py wrapper signature:
        #   e = surface_energy_and_gradient(pos, tri, gamma, grad, zero_based, [nv, nf])
        energy = kernel(pos_f, tri_f, gamma, grad_f, zero_based)
        return float(energy)
    except (TypeError, ValueError):
        pass

    try:
        # Some f2py builds preserve the Fortran argument order.
        energy = kernel(nv, nf, pos_f, tri_f, gamma, grad_f, zero_based)
        return float(energy)
    except (TypeError, ValueError):
        pass

    try:
        # Others infer nv/nf from array shapes and accept keywords.
        energy = kernel(
            pos_f, tri_f, gamma, grad_f, nv=nv, nf=nf, zero_based=zero_based
        )
        return float(energy)
    except (TypeError, ValueError):
        pass

    energy = kernel(pos_f, tri_f, gamma, grad_f, zero_based=zero_based)
    return float(energy)


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
        cached_rows, _ = mesh.triangle_row_cache()

        if cached_rows is not None and len(cached_rows) == len(mesh.facets):
            gammas = mesh.get_facet_parameter_array("surface_tension")
            areas = mesh.triangle_areas()
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


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray,
) -> float:
    """Vectorised surface energy and gradient calculation writing to array."""

    # Fast path: pure triangle mesh with cached loops
    if _all_facets_are_triangles(mesh):
        tri_rows_arr, tri_facets = mesh.triangle_row_cache()
        gammas_arr = mesh.get_facet_parameter_array("surface_tension")

        kernel_spec = get_surface_energy_kernel()
        if kernel_spec is not None:
            kernel = kernel_spec.func
            expects_transpose = kernel_spec.expects_transpose
            nv = positions.shape[0]
            nf = tri_rows_arr.shape[0]
            strict = os.environ.get("MEMBRANE_FORTRAN_STRICT_NOCOPY") in {
                "1",
                "true",
                "TRUE",
            }
            if expects_transpose:
                pos_f = positions.T
                tri_f = tri_rows_arr.T
                grad_f = np.zeros((3, nv), dtype=np.float64, order="F")
            else:
                pos_f = positions
                tri_f = tri_rows_arr
                grad_f = grad_arr
            gamma = np.asarray(gammas_arr)

            if strict:
                if (
                    pos_f.dtype != np.float64
                    or tri_f.dtype != np.int32
                    or gamma.dtype != np.float64
                    or grad_f.dtype != np.float64
                ):
                    raise TypeError(
                        "Fortran surface kernel requires float64 positions/gamma/gradient and int32 tri_rows."
                    )
                if not (
                    pos_f.flags["F_CONTIGUOUS"]
                    and tri_f.flags["F_CONTIGUOUS"]
                    and grad_f.flags["F_CONTIGUOUS"]
                    and gamma.flags["C_CONTIGUOUS"]
                ):
                    raise ValueError(
                        "Fortran surface kernel requires F-contiguous positions/tri_rows/gradient and C-contiguous gamma (to avoid hidden copies)."
                    )
            else:
                gamma = np.ascontiguousarray(gamma, dtype=np.float64)
                pos_f = np.asfortranarray(pos_f, dtype=np.float64)
                tri_f = np.asfortranarray(tri_f, dtype=np.int32)

            try:
                energy = _call_fortran_surface_kernel(
                    kernel,
                    nv=nv,
                    nf=nf,
                    pos_f=pos_f,
                    tri_f=tri_f,
                    gamma=gamma,
                    grad_f=grad_f,
                    zero_based=1,
                )
                if expects_transpose:
                    grad_arr += grad_f.T
                return energy
            except Exception as exc:
                logger.warning(
                    "Fortran surface kernel failed; falling back to NumPy (rebuild may be required): %s",
                    exc,
                )

        tri_pos = positions[tri_rows_arr]
        v0 = tri_pos[:, 0, :]
        v1 = tri_pos[:, 1, :]
        v2 = tri_pos[:, 2, :]

        e1 = v1 - v0
        e2 = v2 - v0
        n = _fast_cross(e1, e2)
        A = np.linalg.norm(n, axis=1)
        mask = A >= 1e-12

        if not np.any(mask):
            return 0.0

        n_hat = n[mask] / A[mask][:, None]
        areas = 0.5 * A[mask]
        gammas_masked = gammas_arr[mask]

        E = float(np.dot(gammas_masked, areas))

        v0m = v0[mask]
        v1m = v1[mask]
        v2m = v2[mask]

        # Correct area gradients: perpendicular to opposite edge
        # grad(A) wrt v0 = 0.5 * (v1 - v2) x n_hat
        g0 = 0.5 * _fast_cross(v1m - v2m, n_hat)
        g1 = 0.5 * _fast_cross(v2m - v0m, n_hat)
        g2 = 0.5 * _fast_cross(v0m - v1m, n_hat)

        scale = gammas_masked[:, None]
        g0 *= scale
        g1 *= scale
        g2 *= scale

        tri_rows_masked = tri_rows_arr[mask]
        np.add.at(grad_arr, tri_rows_masked[:, 0], g0)
        np.add.at(grad_arr, tri_rows_masked[:, 1], g1)
        np.add.at(grad_arr, tri_rows_masked[:, 2], g2)

        return E

    # Fallback for non-triangle meshes
    E = 0.0
    for facet in mesh.facets.values():
        surface_tension = param_resolver.get(facet, "surface_tension")
        if surface_tension is None:
            surface_tension = global_params.get("surface_tension")

        area, area_gradient = facet.compute_area_and_gradient(
            mesh, positions=positions, index_map=index_map
        )
        E += surface_tension * area

        for vertex_index, gradient_vector in area_gradient.items():
            row = index_map.get(vertex_index)
            if row is not None:
                grad_arr[row] += surface_tension * gradient_vector
    return E


def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> tuple[float, Dict[int, np.ndarray]]:
    """Compute surface energy and optionally its gradient."""

    if not compute_gradient:
        E = calculate_surface_energy(mesh, global_params)
        logger.debug(f"Computed surface energy (energy‑only path): {E}")
        return E, {}

    # Use the array-based backend
    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    n_vertices = len(mesh.vertex_ids)
    grad_arr = np.zeros((n_vertices, 3), dtype=float)

    E = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
    )

    # Convert back to dict
    grad = {vid: grad_arr[row] for vid, row in idx_map.items()}

    if logger.isEnabledFor(10):
        logger.debug("Computed surface energy: %.6f", E)
    return E, grad
