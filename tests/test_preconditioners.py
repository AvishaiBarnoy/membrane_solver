"""Focused unit tests for tilt preconditioner builders."""

from __future__ import annotations

import numpy as np

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.geom_io import parse_geometry
from runtime.energy_context import EnergyContext
from runtime.preconditioners import (
    build_leaflet_tilt_cg_preconditioner,
    build_tilt_cg_preconditioner,
)


def _build_simple_mesh():
    """Build a 4-vertex, 2-triangle mesh (a simple square patch)."""
    # 0---1
    # | \ |
    # 3---2
    data = {
        "vertices": {
            0: [0.0, 0.0, 0.0],
            1: [1.0, 0.0, 0.0],
            2: [1.0, 1.0, 0.0],
            3: [0.0, 1.0, 0.0],
        },
        "edges": {
            1: [0, 1],
            2: [1, 2],
            3: [2, 3],
            4: [3, 0],
            5: [0, 2],
        },
        "faces": {
            0: [1, 2, "r5"],
            1: [3, 4, -5],
        },
        "energy_modules": ["bending_tilt"],
        "global_parameters": {},
        "instructions": [],
    }
    mesh = parse_geometry(data)
    return mesh


def test_build_tilt_cg_preconditioner_pure_identity():
    mesh = _build_simple_mesh()
    params = GlobalParameters()
    resolver = ParameterResolver(params)
    context = EnergyContext()
    context.ensure_for_mesh(mesh)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    fixed_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)

    # With no rigidity, should return identity (all ones)
    M_inv = build_tilt_cg_preconditioner(
        mesh,
        resolver,
        context,
        positions=positions,
        index_map=index_map,
        fixed_mask=fixed_mask,
    )

    np.testing.assert_allclose(M_inv, 1.0)


def test_build_tilt_cg_preconditioner_with_rigidity():
    mesh = _build_simple_mesh()
    params = GlobalParameters()
    params.set("tilt_rigidity", 1.0)
    resolver = ParameterResolver(params)
    context = EnergyContext()
    context.ensure_for_mesh(mesh)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    fixed_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)

    M_inv = build_tilt_cg_preconditioner(
        mesh,
        resolver,
        context,
        positions=positions,
        index_map=index_map,
        fixed_mask=fixed_mask,
    )

    # Jacobi preconditioner is 1/diag.
    # For small vertex areas, diag < 1.0, so M_inv > 1.0.
    assert np.all(M_inv >= 1.0)
    assert not np.allclose(M_inv, 1.0)


def test_build_leaflet_tilt_cg_preconditioner_with_rigidity():
    mesh = _build_simple_mesh()
    params = GlobalParameters()
    params.set("tilt_modulus_in", 1.0)
    params.set("tilt_modulus_out", 2.0)
    resolver = ParameterResolver(params)
    context = EnergyContext()
    context.ensure_for_mesh(mesh)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    fixed_mask_in = np.zeros(len(mesh.vertex_ids), dtype=bool)
    fixed_mask_out = np.zeros(len(mesh.vertex_ids), dtype=bool)

    M_inv_in, M_inv_out = build_leaflet_tilt_cg_preconditioner(
        mesh,
        resolver,
        context,
        positions=positions,
        index_map=index_map,
        fixed_mask_in=fixed_mask_in,
        fixed_mask_out=fixed_mask_out,
    )

    # Should be different since moduli are different
    assert not np.allclose(M_inv_in, M_inv_out)

    # Modulus out is higher (2.0) than in (1.0), so diag_out > diag_in, so M_inv_out < M_inv_in
    # We only check elements where M_inv_in > 1.0 (i.e. diag > 1e-12)
    mask = M_inv_in > 1.0
    assert np.any(mask), f"M_inv_in was {M_inv_in}"
    np.testing.assert_array_less(M_inv_out[mask], M_inv_in[mask])


def test_build_leaflet_tilt_cg_preconditioner_uses_provided_leaflet_areas():
    mesh = _build_simple_mesh()
    params = GlobalParameters()
    params.set("tilt_modulus_in", 1.0)
    params.set("tilt_modulus_out", 1.0)
    resolver = ParameterResolver(params)
    context = EnergyContext()
    context.ensure_for_mesh(mesh)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    fixed_mask_in = np.zeros(len(mesh.vertex_ids), dtype=bool)
    fixed_mask_out = np.zeros(len(mesh.vertex_ids), dtype=bool)
    custom_in = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
    custom_out = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=float)

    M_inv_in, M_inv_out = build_leaflet_tilt_cg_preconditioner(
        mesh,
        resolver,
        context,
        positions=positions,
        index_map=index_map,
        fixed_mask_in=fixed_mask_in,
        fixed_mask_out=fixed_mask_out,
        tilt_vertex_areas_in=custom_in,
        tilt_vertex_areas_out=custom_out,
    )

    np.testing.assert_allclose(M_inv_in, 1.0 / custom_in)
    np.testing.assert_allclose(M_inv_out, 1.0 / custom_out)
