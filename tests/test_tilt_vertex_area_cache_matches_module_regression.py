import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from modules.energy import tilt_in, tilt_out  # noqa: E402
from modules.energy.leaflet_presence import (  # noqa: E402
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


@pytest.mark.regression
def test_tilt_vertex_area_cache_matches_tilt_in_module_energy_and_gradient() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    gp = mesh.global_parameters
    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )

    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    assert tri_rows is not None and len(tri_rows) > 0

    vertex_areas = minim._tilt_vertex_areas_from_triangles(
        n_vertices=len(mesh.vertex_ids),
        tri_rows=tri_rows,
        positions=positions,
    )

    rng = np.random.default_rng(0)
    tin = 1.0e-2 * rng.standard_normal(size=mesh.tilts_in_view().shape)

    grad_dummy = np.zeros_like(positions)
    tilt_grad_mod = np.zeros_like(positions)
    e_mod = tilt_in.compute_energy_and_gradient_array(
        mesh,
        gp,
        minim.param_resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad_dummy,
        tilts_in=tin,
        tilt_in_grad_arr=tilt_grad_mod,
    )

    k_tilt = float(gp.get("tilt_modulus_in") or 0.0)
    sq = np.einsum("ij,ij->i", tin, tin)
    e_fast = float(0.5 * k_tilt * np.sum(sq * vertex_areas))
    tilt_grad_fast = k_tilt * tin * vertex_areas[:, None]

    assert float(e_mod) == pytest.approx(e_fast, rel=1e-12, abs=1e-12)
    assert tilt_grad_mod == pytest.approx(tilt_grad_fast, rel=1e-12, abs=1e-12)


@pytest.mark.regression
def test_tilt_vertex_area_cache_matches_tilt_out_module_energy_and_gradient() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    gp = mesh.global_parameters
    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )

    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    assert tri_rows is not None and len(tri_rows) > 0

    absent_mask = leaflet_absent_vertex_mask(mesh, gp, leaflet="out")
    keep = leaflet_present_triangle_mask(mesh, tri_rows, absent_vertex_mask=absent_mask)
    tri_rows_out = tri_rows[keep] if keep.size else tri_rows

    vertex_areas_out = minim._tilt_vertex_areas_from_triangles(
        n_vertices=len(mesh.vertex_ids),
        tri_rows=tri_rows_out,
        positions=positions,
    )

    rng = np.random.default_rng(1)
    tout = 1.0e-2 * rng.standard_normal(size=mesh.tilts_out_view().shape)

    grad_dummy = np.zeros_like(positions)
    tilt_grad_mod = np.zeros_like(positions)
    e_mod = tilt_out.compute_energy_and_gradient_array(
        mesh,
        gp,
        minim.param_resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad_dummy,
        tilts_out=tout,
        tilt_out_grad_arr=tilt_grad_mod,
    )

    k_tilt = float(gp.get("tilt_modulus_out") or 0.0)
    sq = np.einsum("ij,ij->i", tout, tout)
    e_fast = float(0.5 * k_tilt * np.sum(sq * vertex_areas_out))
    tilt_grad_fast = k_tilt * tout * vertex_areas_out[:, None]

    assert float(e_mod) == pytest.approx(e_fast, rel=1e-12, abs=1e-12)
    assert tilt_grad_mod == pytest.approx(tilt_grad_fast, rel=1e-12, abs=1e-12)
