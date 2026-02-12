import os
import sys

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


def test_geometry_cache_invalidates_on_mesh_version_change() -> None:
    mesh = _build_mesh()
    ctx = EnergyContext()
    ctx.ensure_for_mesh(mesh)
    ctx.geometry.set("probe", 123)
    assert ctx.geometry.get("probe") == 123

    mesh.increment_version()
    ctx.ensure_for_mesh(mesh)
    assert ctx.geometry.get("probe") is None
    assert ctx.geometry.is_valid_for(mesh)


def test_geometry_cache_invalidates_on_topology_change() -> None:
    mesh = _build_mesh()
    ctx = EnergyContext()
    ctx.ensure_for_mesh(mesh)
    ctx.geometry.set("probe", 123)
    assert ctx.geometry.get("probe") == 123

    mesh.increment_topology_version()
    ctx.ensure_for_mesh(mesh)
    assert ctx.geometry.get("probe") is None
    assert ctx.geometry.is_valid_for(mesh)


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
    ctx1.geometry.set("probe", 123)
    ctx2 = minim.energy_context()
    assert ctx2 is ctx1
    assert ctx2.geometry.get("probe") == 123

    mesh.increment_version()
    ctx3 = minim.energy_context()
    assert ctx3 is ctx1
    assert ctx3.geometry.get("probe") is None


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
