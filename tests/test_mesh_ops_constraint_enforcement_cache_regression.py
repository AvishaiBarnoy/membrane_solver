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


def _is_pin_to_circle(opts: dict) -> bool:
    constraints = opts.get("constraints")
    if constraints is None:
        return False
    if isinstance(constraints, str):
        return constraints == "pin_to_circle"
    if isinstance(constraints, list):
        return "pin_to_circle" in constraints
    return False


@pytest.mark.regression
def test_mesh_ops_constraint_enforcement_invalidates_positions_cache():
    """Refine creates chord midpoints; enforcing pin_to_circle must update SoA caches.

    This guards a subtle failure mode: if constraint enforcement changes vertex
    positions but the mesh version is not bumped, `mesh.positions_view()` can
    return a stale cached array and downstream energy/diagnostics will see the
    pre-enforcement geometry.
    """
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_free_disk_coarse_refinable.yaml")
    )
    refined = refine_triangle_mesh(mesh)

    # Seed the positions cache (simulating downstream code building SoA views).
    refined.build_position_cache()
    pos_before = refined.positions_view().copy()

    # Build a minimal minimizer that can call enforce_constraints_after_mesh_ops.
    gp = refined.global_parameters
    stepper = GradientDescent()
    em = EnergyModuleManager(getattr(refined, "energy_modules", []) or [])
    cm = ConstraintModuleManager(getattr(refined, "constraint_modules", []) or [])
    minimizer = Minimizer(refined, gp, stepper, em, cm, quiet=True)

    # Identify disk-ring pin_to_circle vertices by options (post-refine).
    disk_rows = []
    radii = []
    for vid in refined.vertex_ids:
        v = refined.vertices[int(vid)]
        opts = getattr(v, "options", None) or {}
        if not _is_pin_to_circle(opts):
            continue
        if opts.get("pin_to_circle_group") != "disk":
            continue
        r = float(opts.get("pin_to_circle_radius"))
        disk_rows.append(int(refined.vertex_index_to_row[int(vid)]))
        radii.append(r)

    assert disk_rows, "Expected refined mesh to contain disk pin_to_circle vertices."
    R = float(np.median(np.asarray(radii, dtype=float)))
    disk_rows = np.asarray(disk_rows, dtype=int)

    # Before enforcement, some refined ring vertices are chord midpoints (r < R).
    r_before = np.linalg.norm(pos_before[disk_rows, :2], axis=1)
    assert np.max(np.abs(r_before - R)) > 1e-6

    # Enforce constraints and verify positions_view reflects the new geometry.
    minimizer.enforce_constraints_after_mesh_ops(refined)
    pos_after = refined.positions_view().copy()
    r_after = np.linalg.norm(pos_after[disk_rows, :2], axis=1)

    assert np.max(np.abs(r_after - R)) < 1e-10
