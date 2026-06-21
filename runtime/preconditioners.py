"""Jacobi preconditioners for tilt relaxation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from core.parameters.resolver import ParameterResolver
    from geometry.entities import Mesh
    from runtime.energy_context import EnergyContext


def build_tilt_cg_preconditioner(
    mesh: Mesh,
    param_resolver: ParameterResolver,
    energy_context: EnergyContext,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    fixed_mask: np.ndarray,
) -> np.ndarray:
    """Return a Jacobi preconditioner for the single-tilt CG solve."""
    n_vertices = len(mesh.vertex_ids)
    diag = np.zeros(n_vertices, dtype=float)

    k_tilt = float(param_resolver.get(None, "tilt_rigidity") or 0.0)
    if k_tilt != 0.0:
        tri_rows, _ = energy_context.geometry.triangle_rows(mesh)
        if tri_rows is not None and len(tri_rows) > 0:
            areas = energy_context.geometry.triangle_areas(mesh, positions)
            if areas is not None and len(areas) == len(tri_rows):
                vertex_areas = np.zeros(n_vertices, dtype=float)
                area_thirds = areas / 3.0
                np.add.at(vertex_areas, tri_rows[:, 0], area_thirds)
                np.add.at(vertex_areas, tri_rows[:, 1], area_thirds)
                np.add.at(vertex_areas, tri_rows[:, 2], area_thirds)
                diag += k_tilt * vertex_areas

    k_smooth = float(param_resolver.get(None, "tilt_smoothness_rigidity") or 0.0)
    if k_smooth != 0.0:
        from geometry.curvature import compute_curvature_data

        _k_vecs, _areas, weights, tri_rows = compute_curvature_data(
            mesh, positions, index_map
        )
        if weights is not None and tri_rows is not None and len(tri_rows) > 0:
            c0 = weights[:, 0]
            c1 = weights[:, 1]
            c2 = weights[:, 2]
            factor = 0.5 * k_smooth
            np.add.at(diag, tri_rows[:, 0], factor * (c1 + c2))
            np.add.at(diag, tri_rows[:, 1], factor * (c2 + c0))
            np.add.at(diag, tri_rows[:, 2], factor * (c0 + c1))

    diag = np.where(diag > 1e-12, diag, 1.0)
    diag[fixed_mask] = 1.0
    return 1.0 / diag


def build_leaflet_tilt_cg_preconditioner(
    mesh: Mesh,
    param_resolver: ParameterResolver,
    energy_context: EnergyContext,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    fixed_mask_in: np.ndarray,
    fixed_mask_out: np.ndarray,
    tilt_vertex_areas_in: np.ndarray | None = None,
    tilt_vertex_areas_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Jacobi preconditioners for leaflet tilt CG solves."""
    n_vertices = len(mesh.vertex_ids)
    diag_in = np.zeros(n_vertices, dtype=float)
    diag_out = np.zeros(n_vertices, dtype=float)

    k_in = float(param_resolver.get(None, "tilt_modulus_in") or 0.0)
    k_out = float(param_resolver.get(None, "tilt_modulus_out") or 0.0)
    if k_in != 0.0 or k_out != 0.0:
        vertex_areas_in = (
            np.asarray(tilt_vertex_areas_in, dtype=float)
            if tilt_vertex_areas_in is not None
            else None
        )
        vertex_areas_out = (
            np.asarray(tilt_vertex_areas_out, dtype=float)
            if tilt_vertex_areas_out is not None
            else None
        )
        if vertex_areas_in is None or vertex_areas_out is None:
            tri_rows, _ = energy_context.geometry.triangle_rows(mesh)
            if tri_rows is not None and len(tri_rows) > 0:
                areas = energy_context.geometry.triangle_areas(mesh, positions)
                if areas is not None and len(areas) == len(tri_rows):
                    vertex_areas = np.zeros(n_vertices, dtype=float)
                    area_thirds = areas / 3.0
                    np.add.at(vertex_areas, tri_rows[:, 0], area_thirds)
                    np.add.at(vertex_areas, tri_rows[:, 1], area_thirds)
                    np.add.at(vertex_areas, tri_rows[:, 2], area_thirds)
                    if vertex_areas_in is None:
                        vertex_areas_in = vertex_areas
                    if vertex_areas_out is None:
                        vertex_areas_out = vertex_areas
        if vertex_areas_in is not None and k_in != 0.0:
            diag_in += k_in * vertex_areas_in
        if vertex_areas_out is not None and k_out != 0.0:
            diag_out += k_out * vertex_areas_out

    k_smooth_in = float(
        param_resolver.get(None, "bending_modulus_in")
        or param_resolver.get(None, "bending_modulus")
        or 0.0
    )
    k_smooth_out = float(
        param_resolver.get(None, "bending_modulus_out")
        or param_resolver.get(None, "bending_modulus")
        or 0.0
    )
    if k_smooth_in != 0.0 or k_smooth_out != 0.0:
        from geometry.curvature import compute_curvature_data

        _k_vecs, _areas, weights, tri_rows = compute_curvature_data(
            mesh, positions, index_map
        )
        if weights is not None and tri_rows is not None and len(tri_rows) > 0:
            c0 = weights[:, 0]
            c1 = weights[:, 1]
            c2 = weights[:, 2]
            if k_smooth_in != 0.0:
                factor_in = 0.5 * k_smooth_in
                np.add.at(diag_in, tri_rows[:, 0], factor_in * (c1 + c2))
                np.add.at(diag_in, tri_rows[:, 1], factor_in * (c2 + c0))
                np.add.at(diag_in, tri_rows[:, 2], factor_in * (c0 + c1))
            if k_smooth_out != 0.0:
                factor_out = 0.5 * k_smooth_out
                np.add.at(diag_out, tri_rows[:, 0], factor_out * (c1 + c2))
                np.add.at(diag_out, tri_rows[:, 1], factor_out * (c2 + c0))
                np.add.at(diag_out, tri_rows[:, 2], factor_out * (c0 + c1))

    diag_in = np.where(diag_in > 1e-12, diag_in, 1.0)
    diag_out = np.where(diag_out > 1e-12, diag_out, 1.0)
    diag_in[fixed_mask_in] = 1.0
    diag_out[fixed_mask_out] = 1.0
    return 1.0 / diag_in, 1.0 / diag_out
