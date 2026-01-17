import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.refinement import refine_triangle_mesh
from runtime.steppers.gradient_descent import GradientDescent


def _relax_leaflet_tilts(mesh, *, inner_steps: int) -> Minimizer:
    mesh.global_parameters.update(
        {
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.05,
            "tilt_inner_steps": inner_steps,
            "tilt_tol": 1e-12,
        }
    )
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.increment_version()
    mesh._positions_cache = None
    mesh._positions_cache_version = -1
    mesh.project_tilts_to_tangent()
    minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode="nested")
    return minim


def _ring_mean(mags: np.ndarray, radii: np.ndarray, r0: float) -> float:
    idx = np.where(np.isclose(radii, r0, atol=1e-6))[0]
    assert idx.size > 0
    return float(mags[idx].mean())


def test_kozlov_annulus_flat_hard_source_decay() -> None:
    """Milestone-B E2E: tilt decays from an inner rim source on a flat annulus.

    The benchmark mesh clamps |t_in|=1 on the inner rim (r=1) and clamps t_in=0
    at the far field boundary (r=3). With `tilt_smoothness_in` + `tilt_in` and
    k_s=k_t=1, the relaxed solution should exhibit a monotone decay in |t_in|
    between rings.
    """

    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_annulus_flat_hard_source.yaml")
    )
    _relax_leaflet_tilts(mesh, inner_steps=800)

    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    mags = np.linalg.norm(mesh.tilts_in_view(), axis=1)

    m1 = _ring_mean(mags, radii, 1.0)
    m2 = _ring_mean(mags, radii, 2.0)
    m3 = _ring_mean(mags, radii, 3.0)

    assert m1 == pytest.approx(1.0, abs=2e-6)
    assert m3 == pytest.approx(0.0, abs=2e-6)
    assert m1 > m2 > m3
    assert m2 < 0.55


def test_kozlov_annulus_rotation_invariance() -> None:
    """Milestone-B regression: rotating the annulus does not change energy."""

    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_annulus_flat_hard_source.yaml")
    )
    mesh_rot = mesh.copy()

    theta = np.deg2rad(22.5)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    for v in mesh_rot.vertices.values():
        v.position = (R @ np.asarray(v.position, dtype=float)).astype(float)
        v.tilt_in = (R @ np.asarray(v.tilt_in, dtype=float)).astype(float)

    mesh_rot.increment_version()
    mesh_rot._positions_cache = None
    mesh_rot.touch_tilts_in()
    mesh_rot.build_connectivity_maps()
    mesh_rot.build_facet_vertex_loops()

    e0 = float(_relax_leaflet_tilts(mesh, inner_steps=800).compute_energy())
    e1 = float(_relax_leaflet_tilts(mesh_rot, inner_steps=800).compute_energy())

    assert e0 == pytest.approx(e1, rel=5e-6, abs=5e-6)


def test_kozlov_annulus_energy_decreases_under_refinement() -> None:
    """Milestone-B regression: refined annulus relaxes to a lower energy."""

    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_annulus_flat_hard_source.yaml")
    )
    e0 = float(_relax_leaflet_tilts(mesh, inner_steps=1200).compute_energy())
    mesh = refine_triangle_mesh(mesh)
    e1 = float(_relax_leaflet_tilts(mesh, inner_steps=1200).compute_energy())
    mesh = refine_triangle_mesh(mesh)
    e2 = float(_relax_leaflet_tilts(mesh, inner_steps=1200).compute_energy())

    assert e0 > e1 > e2
    assert 0.0 < e2 < e0


def test_kozlov_annulus_coupling_tracking() -> None:
    """Milestone-B: tilt_out tracks tilt_in with strong coupling."""
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_annulus_flat_hard_source.yaml")
    )

    # Enable coupling energy and relax both leaflets
    mesh.energy_modules.append("tilt_coupling")
    mesh.global_parameters.update(
        {
            "tilt_coupling_modulus": 10.0,
            "tilt_coupling_mode": "difference",
            "tilt_smoothness_out": 1.0,
            "tilt_out": 1.0,
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.05,
            "tilt_inner_steps": 1000,
            "tilt_tol": 1e-12,
        }
    )

    # Note: mesh loaded has tilt_fixed_in=True on rims, but tilt_fixed_out is False
    # everywhere by default (unless specified). We want tilt_out to be free to track.

    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode="nested")

    # Check that tilt_out has tracked tilt_in (difference should be small)
    t_in = mesh.tilts_in_view()
    t_out = mesh.tilts_out_view()
    diff = np.linalg.norm(t_in - t_out, axis=1)

    # With strong coupling (k_c=10 vs k_s=1), difference should be small
    assert np.mean(diff) < 0.1
    # tilt_in max is ~1.0, so tilt_out should be significant
    assert np.max(np.linalg.norm(t_out, axis=1)) > 0.9
