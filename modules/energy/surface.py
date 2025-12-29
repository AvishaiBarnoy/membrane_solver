# modules/surface.py
# Here goes energy functions relevant for area of facets

import logging
import os
from typing import Dict

import numpy as np

from geometry.entities import Mesh, _fast_cross

logger = logging.getLogger("membrane_solver")

_FORTRAN_SURFACE_KERNEL = None


def _get_fortran_surface_kernel():
    """Return the f2py-wrapped Fortran surface kernel if available.

    The expected build is something like:
      `python -m numpy.f2py -c -m surface_energy fortran_kernels/surface_energy.f90`

    Depending on f2py/version, the callable may be exposed at either
    `surface_energy.surface_energy_mod.surface_energy_and_gradient` or as a
    top-level function.
    """
    global _FORTRAN_SURFACE_KERNEL
    if _FORTRAN_SURFACE_KERNEL is not None:
        return _FORTRAN_SURFACE_KERNEL

    if os.environ.get("MEMBRANE_DISABLE_FORTRAN_SURFACE") in {"1", "true", "TRUE"}:
        _FORTRAN_SURFACE_KERNEL = False
        return _FORTRAN_SURFACE_KERNEL

    candidates = []
    try:
        import fortran_kernels.surface_energy as fe  # type: ignore

        candidates.append(fe)
    except Exception:
        pass

    try:
        import surface_energy as fe  # type: ignore

        candidates.append(fe)
    except Exception:
        pass

    for mod in candidates:
        fn = getattr(mod, "surface_energy_and_gradient", None)
        if not callable(fn):
            submod = getattr(mod, "surface_energy_mod", None)
            fn = (
                getattr(submod, "surface_energy_and_gradient", None) if submod else None
            )

        if not callable(fn):
            continue

        doc = getattr(fn, "__doc__", "") or ""
        # f2py wrapper doc includes bounds like "(nv,3)" or "(3,nv)".
        expects_transpose = "bounds (3,nv)" in doc or "bounds (3, nv)" in doc
        _FORTRAN_SURFACE_KERNEL = (fn, expects_transpose)
        return _FORTRAN_SURFACE_KERNEL

    _FORTRAN_SURFACE_KERNEL = False
    return _FORTRAN_SURFACE_KERNEL


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
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row

        cached_rows, cached_facets = mesh.triangle_row_cache()

        if cached_rows is not None and len(cached_rows) == len(mesh.facets):
            tri_rows = cached_rows
            n_facets = len(cached_rows)
            gammas = np.empty(n_facets, dtype=float)
            st_default = global_params.get("surface_tension")

            for i, fid in enumerate(cached_facets):
                facet = mesh.facets[fid]
                gammas[i] = facet.options.get("surface_tension", st_default)
        else:
            # Fallback if cache mismatch
            n_facets = len(mesh.facets)
            tri_rows = np.empty((n_facets, 3), dtype=int)
            gammas = np.empty(n_facets, dtype=float)

            for idx, facet in enumerate(mesh.facets.values()):
                loop = mesh.facet_vertex_loops[facet.index]
                tri_rows[idx, 0] = index_map[int(loop[0])]
                tri_rows[idx, 1] = index_map[int(loop[1])]
                tri_rows[idx, 2] = index_map[int(loop[2])]
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
        gammas_arr = np.empty(len(tri_facets), dtype=float)

        for idx, fid in enumerate(tri_facets):
            facet = mesh.facets[fid]
            surface_tension = param_resolver.get(facet, "surface_tension")
            if surface_tension is None:
                surface_tension = global_params.get("surface_tension")
            gammas_arr[idx] = surface_tension

        kernel_info = _get_fortran_surface_kernel()
        if kernel_info:
            kernel, expects_transpose = kernel_info
            nv = positions.shape[0]
            nf = tri_rows_arr.shape[0]
            gamma = np.ascontiguousarray(gammas_arr, dtype=np.float64)
            if expects_transpose:
                pos_f = np.asfortranarray(positions.T, dtype=np.float64)
                tri_f = np.asfortranarray(tri_rows_arr.T, dtype=np.int32)
                grad_f = np.zeros((3, nv), dtype=np.float64, order="F")
            else:
                pos_f = np.asfortranarray(positions, dtype=np.float64)
                tri_f = np.asfortranarray(tri_rows_arr, dtype=np.int32)
                grad_f = grad_arr

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
                global _FORTRAN_SURFACE_KERNEL

                _FORTRAN_SURFACE_KERNEL = False
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
