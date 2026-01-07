import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.curvature import compute_curvature_fields
from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.diagnostics.gauss_bonnet import extract_boundary_loops, find_boundary_edges
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.refinement import refine_triangle_mesh
from runtime.steppers.gradient_descent import GradientDescent


def _interior_mean_curvature_stats(mesh) -> tuple[float, float, float]:
    mesh.build_facet_vertex_loops()
    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    fields = compute_curvature_fields(mesh, positions, idx_map)

    boundary = set(mesh.boundary_vertex_ids or [])
    interior_rows = [
        idx_map[int(vid)] for vid in mesh.vertex_ids if int(vid) not in boundary
    ]
    if not interior_rows:
        return 0.0, 0.0, 0.0

    h = np.asarray(fields.mean_curvature[interior_rows], dtype=float)
    return float(h.mean()), float(np.percentile(h, 95)), float(h.max())


def _minimize_surface(mesh, *, steps: int, step_size: float) -> None:
    mesh.global_parameters.set("volume_constraint_mode", "none")
    mesh.energy_modules = ["surface"]
    mesh.constraint_modules = []

    minimizer = Minimizer(
        mesh=mesh,
        global_params=mesh.global_parameters,
        stepper=GradientDescent(max_iter=20),
        energy_manager=EnergyModuleManager(mesh.energy_modules),
        constraint_manager=ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        step_size=float(step_size),
        tol=1e-12,
    )
    minimizer.minimize(n_steps=int(steps))


def test_catenoid_like_surface_has_small_mean_curvature_after_relaxation():
    """Catenoid is a minimal surface, so interior mean curvature should be small."""
    mesh = parse_geometry(load_data("meshes/catenoid.json"))

    mesh = refine_triangle_mesh(mesh)
    _minimize_surface(mesh, steps=100, step_size=1e-2)

    h_mean, h_p95, h_max = _interior_mean_curvature_stats(mesh)
    assert h_mean < 0.2
    assert h_p95 < 0.25
    assert h_max < 0.3

    # Refinement invariance: after refinement + a short relax, curvature remains bounded.
    refined = refine_triangle_mesh(mesh)
    _minimize_surface(refined, steps=50, step_size=1e-2)

    h_mean2, h_p95_2, h_max2 = _interior_mean_curvature_stats(refined)
    assert h_mean2 < 0.35
    assert h_p95_2 < 0.6
    assert h_max2 < 0.8

    boundary_edges = find_boundary_edges(refined)
    loops = extract_boundary_loops(refined, boundary_edges)
    assert len(loops) == 2
