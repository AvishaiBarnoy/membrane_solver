import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


def _disk_ring_mask(mesh) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") == "disk":
            rows.append(int(mesh.vertex_index_to_row[int(vid)]))
    mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    mask[np.asarray(rows, dtype=int)] = True
    return mask


@pytest.mark.regression
def test_leaflet_tilt_relaxation_enforces_thetaB_boundary_before_gradient_eval() -> (
    None
):
    """Regression: thetaB boundary constraints must be applied even from zero tilts."""
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    gp = mesh.global_parameters

    # Make tilt relaxation rely on the boundary condition (not on spurious base-term curvature).
    gp.set("bending_tilt_base_term_boundary_group_in", "disk")

    gp.set("tilt_thetaB_optimize", False)
    gp.set("tilt_thetaB_value", 0.03)

    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 1)
    gp.set("tilt_tol", 0.0)
    gp.set("tilt_kkt_projection_during_relaxation", False)

    # Start from a fully zero tilt field.
    mesh.set_tilts_in_from_array(np.zeros_like(mesh.tilts_in_view()))
    mesh.set_tilts_out_from_array(np.zeros_like(mesh.tilts_out_view()))

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-8,
    )

    minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode="coupled")

    mask = _disk_ring_mask(mesh)
    assert int(np.sum(mask)) > 0
    norms = np.linalg.norm(mesh.tilts_in_view()[mask], axis=1)
    assert float(np.median(norms)) > 1e-6
