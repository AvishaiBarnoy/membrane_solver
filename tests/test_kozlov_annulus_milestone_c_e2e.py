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


def _break_up_down_symmetry(
    mesh, *, z_bump: float = 1e-3, target_radius: float = 2.0
) -> None:
    """Perturb a mid-radius vertex in z to avoid the flat saddle-point solution."""
    mesh.increment_version()
    mesh._positions_cache = None
    mesh._positions_cache_version = -1
    positions = mesh.positions_view()
    radii = np.linalg.norm(positions[:, :2], axis=1)
    mid_row = int(np.argmin(np.abs(radii - float(target_radius))))
    mesh.vertices[int(mesh.vertex_ids[mid_row])].position[2] = float(z_bump)
    mesh.increment_version()
    mesh._positions_cache = None
    mesh._positions_cache_version = -1


def _mean_z_midband(mesh, *, r_min: float = 1.4, r_max: float = 2.6) -> float:
    """Return mean z over a mid-radius band (exclude pinned rims)."""
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    mask = (r > float(r_min)) & (r < float(r_max))
    if not np.any(mask):
        return 0.0
    return float(np.mean(positions[mask, 2]))


def test_milestone_c_soft_source_generates_curvature_and_outer_tilt() -> None:
    """Milestone C: soft rim source + bending_tilt generates curvature; outer leaflet responds."""
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_annulus_milestone_c_soft_source.yaml")
    )
    mesh.global_parameters.set("tilt_inner_steps", 20)
    mesh.global_parameters.set("tilt_tol", 1e-8)

    # Break the up/down symmetry to make the relaxed shape deterministic.
    _break_up_down_symmetry(mesh)

    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=50)

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
    mesh.global_parameters.set("tilt_inner_steps", 20)
    mesh.global_parameters.set("tilt_tol", 1e-8)

    _break_up_down_symmetry(mesh)

    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=50)

    t_out = mesh.tilts_out_view()
    assert float(np.max(np.linalg.norm(t_out, axis=1))) < 5e-5


def test_milestone_c_swapping_source_leaflet_flips_curvature_direction() -> None:
    """Milestone C sign test: putting the same source on the other leaflet flips invagination."""
    mesh_in = parse_geometry(
        load_data("meshes/caveolin/kozlov_annulus_milestone_c_soft_source.yaml")
    )
    mesh_in.global_parameters.set("tilt_inner_steps", 20)
    mesh_in.global_parameters.set("tilt_tol", 1e-8)
    _break_up_down_symmetry(mesh_in)

    minim_in = _build_minimizer(mesh_in)
    minim_in.minimize(n_steps=50)
    mean_z_in = _mean_z_midband(mesh_in)

    mesh_out = parse_geometry(
        load_data("meshes/caveolin/kozlov_annulus_milestone_c_soft_source.yaml")
    )
    mesh_out.global_parameters.set("tilt_inner_steps", 20)
    mesh_out.global_parameters.set("tilt_tol", 1e-8)
    mesh_out.energy_modules = [
        m for m in mesh_out.energy_modules if m != "tilt_rim_source_in"
    ] + ["tilt_rim_source_out"]
    mesh_out.global_parameters.set(
        "tilt_rim_source_group_out",
        mesh_out.global_parameters.get("tilt_rim_source_group_in"),
    )
    mesh_out.global_parameters.set(
        "tilt_rim_source_strength_out",
        mesh_out.global_parameters.get("tilt_rim_source_strength_in"),
    )
    _break_up_down_symmetry(mesh_out)

    minim_out = _build_minimizer(mesh_out)
    minim_out.minimize(n_steps=30)
    mean_z_out = _mean_z_midband(mesh_out)

    assert mean_z_in < -1e-6
    assert mean_z_out > 1e-6
    assert mean_z_in * mean_z_out < 0.0
