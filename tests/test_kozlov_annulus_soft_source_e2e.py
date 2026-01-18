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


def _relax(mesh, *, inner_steps: int) -> Minimizer:
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


def test_kozlov_annulus_soft_source_drives_tilt_decay() -> None:
    """E2E: soft rim source generates nonzero boundary tilt and decay to far field."""
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_annulus_flat_soft_source.yaml")
    )
    _relax(mesh, inner_steps=1200)

    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    mags = np.linalg.norm(mesh.tilts_in_view(), axis=1)

    m1 = _ring_mean(mags, radii, 1.0)
    m2 = _ring_mean(mags, radii, 2.0)
    m3 = _ring_mean(mags, radii, 3.0)

    assert m1 > 0.15
    assert m1 > m2 > m3
    assert m3 == pytest.approx(0.0, abs=2e-6)


def test_kozlov_annulus_soft_source_rotation_invariance() -> None:
    """Regression: rotation about z preserves the relaxed energy."""
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_annulus_flat_soft_source.yaml")
    )
    mesh_rot = mesh.copy()

    theta = np.deg2rad(22.5)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    for v in mesh_rot.vertices.values():
        v.position = (R @ np.asarray(v.position, dtype=float)).astype(float)

    mesh_rot.increment_version()
    mesh_rot._positions_cache = None
    mesh_rot.build_connectivity_maps()
    mesh_rot.build_facet_vertex_loops()

    e0 = float(_relax(mesh, inner_steps=1200).compute_energy())
    e1 = float(_relax(mesh_rot, inner_steps=1200).compute_energy())

    assert e0 == pytest.approx(e1, rel=5e-6, abs=5e-6)


def test_kozlov_annulus_soft_source_energy_decreases_under_refinement() -> None:
    """Regression: as we refine, the relaxed energy should decrease."""
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_annulus_flat_soft_source.yaml")
    )
    e0 = float(_relax(mesh, inner_steps=1600).compute_energy())
    mesh = refine_triangle_mesh(mesh)
    e1 = float(_relax(mesh, inner_steps=1600).compute_energy())
    mesh = refine_triangle_mesh(mesh)
    e2 = float(_relax(mesh, inner_steps=1600).compute_energy())

    assert e0 > e1 > e2
    assert np.isfinite(e2)
