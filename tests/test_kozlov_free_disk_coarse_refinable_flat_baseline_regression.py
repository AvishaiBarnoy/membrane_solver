import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.curvature import compute_angle_defects  # noqa: E402
from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.interface_validation import validate_disk_interface_topology  # noqa: E402
from runtime.refinement import refine_triangle_mesh  # noqa: E402


@pytest.mark.regression
def test_kozlov_free_disk_coarse_refinable_flat_has_no_angle_defect_baseline() -> None:
    """Regression: refined disk-edge ring should not create spurious intrinsic curvature in a flat state."""
    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kozlov_free_disk_coarse_refinable.yaml",
    )
    mesh = parse_geometry(load_data(path))

    # Force a flat reference state.
    for v in mesh.vertices.values():
        v.position[2] = 0.0
    mesh.increment_version()

    # Refine a couple times to stress midpoint creation.
    for _ in range(2):
        mesh = refine_triangle_mesh(mesh)

    # Enforce geometric constraints (projects pinned rings back onto circles/planes).
    cm = ConstraintModuleManager(mesh.constraint_modules)
    cm.enforce_all(mesh, context="mesh_operation", global_params=mesh.global_parameters)
    mesh.increment_version()

    # Disk interface topology must remain valid.
    validate_disk_interface_topology(mesh, mesh.global_parameters)

    mesh.build_position_cache()
    pos = mesh.positions_view()
    idx = mesh.vertex_index_to_row
    defects = compute_angle_defects(mesh, pos, idx)

    # In a valid planar triangulation, interior vertices should have ~0 defect.
    assert float(np.max(np.abs(defects))) < 1e-10
