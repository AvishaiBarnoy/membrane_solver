import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _build_minimizer(mesh) -> Minimizer:
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def _outer_rim_rows(mesh) -> list[int]:
    return [
        mesh.vertex_index_to_row[int(vid)]
        for vid in mesh.vertex_ids
        if mesh.vertices[int(vid)].options.get("pin_to_circle_group") == "outer"
    ]


def test_milestone_c_soft_source_generates_curvature_and_outer_tilt() -> None:
    """Milestone C: soft rim source + bending_tilt generates curvature; outer leaflet responds."""
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_annulus_milestone_c_soft_source.yaml")
    )

    # Break the up/down symmetry to make the relaxed shape deterministic.
    mesh.increment_version()
    mesh._positions_cache = None
    mesh._positions_cache_version = -1
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    mid_row = int(np.argmin(np.abs(radii - 2.0)))
    mesh.vertices[int(mesh.vertex_ids[mid_row])].position[2] = 1e-3
    mesh.increment_version()
    mesh._positions_cache = None
    mesh._positions_cache_version = -1

    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=60)

    positions = mesh.positions_view()
    z = positions[:, 2]

    outer_rows = _outer_rim_rows(mesh)
    assert len(outer_rows) > 0
    assert float(np.max(np.abs(z[outer_rows]))) < 1e-8
    assert float(np.max(np.abs(z))) > 2e-4

    # Without explicit `tilt_coupling`, the outer leaflet can still become nonzero
    # because `bending_tilt_out` couples it to the shared shape curvature.
    t_out = mesh.tilts_out_view()
    assert float(np.max(np.linalg.norm(t_out, axis=1))) > 5e-4


def test_milestone_c_without_bending_tilt_out_keeps_outer_tilt_zeroish() -> None:
    """Milestone C control: removing bending_tilt_out leaves tilt_out near its zero init."""
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_annulus_milestone_c_soft_source.yaml")
    )
    mesh.energy_modules = [m for m in mesh.energy_modules if m != "bending_tilt_out"]

    mesh.increment_version()
    mesh._positions_cache = None
    mesh._positions_cache_version = -1
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    mid_row = int(np.argmin(np.abs(radii - 2.0)))
    mesh.vertices[int(mesh.vertex_ids[mid_row])].position[2] = 1e-3
    mesh.increment_version()
    mesh._positions_cache = None
    mesh._positions_cache_version = -1

    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=60)

    t_out = mesh.tilts_out_view()
    assert float(np.max(np.linalg.norm(t_out, axis=1))) < 5e-5
