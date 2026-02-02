import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


def test_kozlov_free_disk_disk_region_is_pinned_flat() -> None:
    """Regression: disk-covered region stays flat (theory assumption)."""
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "meshes",
        "caveolin",
        "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml",
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

    # A few steps is enough to ensure constraints are applied.
    minim.minimize(n_steps=3)

    positions = mesh.positions_view()
    disk_rows = [
        mesh.vertex_index_to_row[int(vid)]
        for vid in mesh.vertex_ids
        if mesh.vertices[int(vid)].options.get("preset") == "disk"
    ]
    assert disk_rows
    z = positions[np.asarray(disk_rows, dtype=int), 2]
    assert float(np.ptp(z)) < 1e-10
