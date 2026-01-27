import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.executor import execute_command_line
from geometry.curvature import compute_curvature_fields
from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


@dataclass
class _Context:
    mesh: object
    minimizer: object
    stepper: object


def test_single_leaflet_curvature_induction() -> None:
    """Regression: single-leaflet rim contact drives curvature + tilt_out.

    This is a "did we induce the right qualitative response?" test:
    - A nonzero curvature field appears in the outer membrane.
    - The distal drive (tilt_in) induces a nonzero proximal response (tilt_out).
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "meshes",
        "caveolin",
        "kozlov_1disk_3d_tensionless_single_leaflet_soft_source.yaml",
    )
    mesh = parse_geometry(load_data(path))

    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    ctx = _Context(mesh=mesh, minimizer=minim, stepper=minim.stepper)

    execute_command_line(ctx, "induction_quick")

    mesh = ctx.mesh
    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    tilt_in_mag = np.linalg.norm(tilts_in, axis=1)
    tilt_out_mag = np.linalg.norm(tilts_out, axis=1)

    boundary_vids = getattr(mesh, "boundary_vertex_ids", None) or []
    boundary_rows = {mesh.vertex_index_to_row[vid] for vid in boundary_vids}
    interior_rows = np.array(
        [row for row in range(len(mesh.vertex_ids)) if row not in boundary_rows],
        dtype=int,
    )

    assert np.percentile(tilt_in_mag[interior_rows], 90) > 1e-3
    assert np.percentile(tilt_out_mag[interior_rows], 90) > 1e-5

    curvature = compute_curvature_fields(
        mesh, mesh.positions_view(), mesh.vertex_index_to_row
    ).mean_curvature
    assert np.percentile(curvature[interior_rows], 90) > 1e-4
