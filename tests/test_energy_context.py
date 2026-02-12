import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_context import EnergyContext
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _build_mesh():
    return parse_geometry(
        load_data("tests/fixtures/kozlov_free_disk_coarse_refinable.yaml")
    )


def test_triangle_rows_cache_survives_mesh_version_change() -> None:
    mesh = _build_mesh()
    ctx = EnergyContext()
    ctx.ensure_for_mesh(mesh)
    rows1, facets1 = ctx.geometry.triangle_rows(mesh)

    mesh.increment_version()
    ctx.ensure_for_mesh(mesh)
    rows2, facets2 = ctx.geometry.triangle_rows(mesh)

    if rows1 is None:
        assert rows2 is None
    else:
        assert rows2 is rows1
        assert np.array_equal(rows2, rows1)
    assert facets2 == facets1


def test_triangle_rows_cache_refreshes_on_facet_loop_change() -> None:
    mesh = _build_mesh()
    ctx = EnergyContext()
    ctx.ensure_for_mesh(mesh)
    rows1, facets1 = ctx.geometry.triangle_rows(mesh)

    mesh.build_facet_vertex_loops()
    ctx.ensure_for_mesh(mesh)
    rows2, facets2 = ctx.geometry.triangle_rows(mesh)

    assert facets2 == facets1
    if rows1 is not None and rows2 is not None:
        assert rows2 is not rows1
        assert np.array_equal(rows2, rows1)


def test_minimizer_energy_context_reuses_and_rebinds() -> None:
    mesh = _build_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )

    ctx1 = minim.energy_context()
    p1, idx1 = ctx1.geometry.soa_views(mesh)
    ctx2 = minim.energy_context()
    assert ctx2 is ctx1
    p2, idx2 = ctx2.geometry.soa_views(mesh)
    assert p2 is p1
    assert idx2 == idx1

    mesh.increment_version()
    mesh.vertices[int(mesh.vertex_ids[0])].position += np.array([0.1, 0.0, 0.0])
    ctx3 = minim.energy_context()
    assert ctx3 is ctx1
    p3, idx3 = ctx3.geometry.soa_views(mesh)
    assert p3.shape == p2.shape
    assert idx3 == idx2


def test_geometry_cache_triangle_rows_match_mesh_cache() -> None:
    mesh = _build_mesh()
    mesh_rows, mesh_facets = mesh.triangle_row_cache()

    ctx = EnergyContext()
    ctx.ensure_for_mesh(mesh)
    ctx_rows, ctx_facets = ctx.geometry.triangle_rows(mesh)

    if mesh_rows is None:
        assert ctx_rows is None
    else:
        assert ctx_rows is not None
        assert ctx_rows.shape == mesh_rows.shape
        assert (ctx_rows == mesh_rows).all()
    assert ctx_facets == mesh_facets


def test_energy_context_scratch_array_reused_and_zeroed() -> None:
    mesh = _build_mesh()
    ctx = EnergyContext()
    ctx.ensure_for_mesh(mesh)

    arr1 = ctx.scratch_array("grad", shape=(4, 3))
    arr1[0, 0] = 12.0
    arr2 = ctx.scratch_array("grad", shape=(4, 3))
    assert arr2 is arr1
    assert float(arr2.sum()) == 0.0
