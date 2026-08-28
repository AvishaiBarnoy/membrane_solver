import numpy as np
import pytest

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from tests.minimizer_test_utils import build_minimizer as _build_minimizer


def _theory_parity_minimizer():
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    mesh.global_parameters.set("tilt_thetaB_optimize", False)
    return mesh, _build_minimizer(mesh, tol=1e-8)


@pytest.mark.regression
def test_tilt_only_energy_buffer_reuse_is_deterministic() -> None:
    """Guard against stale-buffer reuse affecting energy results."""
    mesh, minim = _theory_parity_minimizer()
    positions = mesh.positions_view()
    rng = np.random.default_rng(1)
    tin = 1e-2 * rng.standard_normal(size=mesh.tilts_in_view().shape)
    tout = 1e-2 * rng.standard_normal(size=mesh.tilts_out_view().shape)

    grad_dummy = np.ones_like(positions) * 123.0
    e1 = minim._compute_tilt_dependent_energy_with_leaflet_tilts(
        positions=positions,
        tilts_in=tin,
        tilts_out=tout,
        grad_dummy=grad_dummy,
    )
    grad_dummy[:] = -999.0
    e2 = minim._compute_tilt_dependent_energy_with_leaflet_tilts(
        positions=positions,
        tilts_in=tin,
        tilts_out=tout,
        grad_dummy=grad_dummy,
    )

    assert e1 == pytest.approx(e2, rel=0.0, abs=0.0)


@pytest.mark.regression
def test_tilt_only_energy_delta_matches_full_energy_delta_with_frozen_positions() -> (
    None
):
    """Tilt relaxation compares energies at fixed positions; deltas must match."""
    mesh, minim = _theory_parity_minimizer()

    positions = mesh.positions_view()
    tin0 = np.zeros_like(mesh.tilts_in_view())
    tout0 = np.zeros_like(mesh.tilts_out_view())
    rng = np.random.default_rng(0)
    tin1 = 1e-2 * rng.standard_normal(size=tin0.shape)
    tout1 = 1e-2 * rng.standard_normal(size=tout0.shape)

    e_full0 = minim._compute_energy_array_with_leaflet_tilts(
        positions=positions, tilts_in=tin0, tilts_out=tout0
    )
    e_full1 = minim._compute_energy_array_with_leaflet_tilts(
        positions=positions, tilts_in=tin1, tilts_out=tout1
    )
    e_tilt0 = minim._compute_tilt_dependent_energy_with_leaflet_tilts(
        positions=positions, tilts_in=tin0, tilts_out=tout0
    )
    e_tilt1 = minim._compute_tilt_dependent_energy_with_leaflet_tilts(
        positions=positions, tilts_in=tin1, tilts_out=tout1
    )

    # Shape-only terms are constant for fixed positions, so energy differences
    # must match closely.
    assert (e_full1 - e_full0) == pytest.approx(
        (e_tilt1 - e_tilt0), rel=1e-10, abs=1e-10
    )


@pytest.mark.regression
def test_single_tilt_energy_delta_matches_full_energy_delta_with_frozen_positions() -> (
    None
):
    """Single-tilt relaxation compares energies at fixed positions; deltas match."""
    mesh = parse_geometry(load_data("meshes/tilt_benchmarks/kh_pure_curl_free.yaml"))
    mesh.global_parameters.set("surface_tension", 1.0)
    if "surface" not in mesh.energy_modules:
        mesh.energy_modules.append("surface")

    minim = _build_minimizer(mesh, tol=1e-8)

    positions = mesh.positions_view()
    t0 = np.zeros_like(mesh.tilts_view())
    rng = np.random.default_rng(0)
    t1 = 1e-2 * rng.standard_normal(size=t0.shape)

    mesh.set_tilts_from_array(t0)
    e_full0 = minim.compute_energy()
    mesh.set_tilts_from_array(t1)
    e_full1 = minim.compute_energy()

    e_tilt0 = minim._compute_energy_array_with_tilts(positions=positions, tilts=t0)
    e_tilt1 = minim._compute_energy_array_with_tilts(positions=positions, tilts=t1)

    assert (e_full1 - e_full0) == pytest.approx(
        (e_tilt1 - e_tilt0), rel=1e-10, abs=1e-10
    )
