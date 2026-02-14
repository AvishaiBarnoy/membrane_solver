import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.resolver import ParameterResolver  # noqa: E402
from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from modules.energy.leaflet_presence import leaflet_absent_vertex_mask  # noqa: E402
from modules.energy.tilt_out import compute_energy_and_gradient_array  # noqa: E402


def _load_mesh():
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    return mesh, mesh.global_parameters


@pytest.mark.regression
def test_leaflet_absent_vertex_mask_reuses_cached_array_per_version() -> None:
    mesh, gp = _load_mesh()
    mask1 = leaflet_absent_vertex_mask(mesh, gp, leaflet="out")
    mask2 = leaflet_absent_vertex_mask(mesh, gp, leaflet="out")
    assert mask2 is mask1
    assert np.array_equal(mask2, mask1)


@pytest.mark.regression
def test_leaflet_absent_vertex_mask_invalidation_on_mesh_and_param_change() -> None:
    mesh, gp = _load_mesh()
    mask1 = leaflet_absent_vertex_mask(mesh, gp, leaflet="out")

    mesh.increment_version()
    mask2 = leaflet_absent_vertex_mask(mesh, gp, leaflet="out")
    assert mask2 is not mask1
    assert np.array_equal(mask2, mask1)

    gp.set("leaflet_out_absent_presets", ["__no_matching_preset__"])
    mask3 = leaflet_absent_vertex_mask(mesh, gp, leaflet="out")
    assert mask3 is not mask2
    assert np.count_nonzero(mask3) == 0


@pytest.mark.regression
def test_tilt_out_energy_gradient_unchanged_with_cached_absent_mask() -> None:
    mesh, gp = _load_mesh()
    resolver = ParameterResolver(gp)
    positions = mesh.positions_view()
    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions)
    tilts_out = mesh.tilts_out_view().copy()

    e1 = compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad_arr,
        tilts_out=tilts_out,
        tilt_out_grad_arr=tilt_grad_arr,
    )
    grad1 = tilt_grad_arr.copy()

    grad_arr.fill(0.0)
    tilt_grad_arr.fill(0.0)
    e2 = compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad_arr,
        tilts_out=tilts_out,
        tilt_out_grad_arr=tilt_grad_arr,
    )
    grad2 = tilt_grad_arr.copy()

    assert float(e2) == pytest.approx(float(e1), rel=1e-12, abs=1e-12)
    assert grad2 == pytest.approx(grad1, rel=1e-12, abs=1e-12)
