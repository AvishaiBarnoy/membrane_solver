def test_kozlov_free_disk_disk_region_is_pinned_flat() -> None:
    """Regression: slide-mode pin_to_plane keeps a patch planar (but allows translation)."""
    import os
    import sys

    import numpy as np

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from geometry.geom_io import parse_geometry  # noqa: E402
    from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402

    # Minimal patch: 4 vertices, 2 triangles. One vertex starts out of plane.
    data = {
        "global_parameters": {
            "pin_to_plane_normal": [0.0, 0.0, 1.0],
            "pin_to_plane_point": [0.0, 0.0, 0.0],
        },
        "constraint_modules": ["pin_to_plane"],
        "definitions": {
            "disk": {
                "constraints": ["pin_to_plane"],
                "pin_to_plane_mode": "slide",
                "pin_to_plane_group": "disk",
            }
        },
        "vertices": [
            [0.0, 0.0, 0.0, {"preset": "disk"}],
            [1.0, 0.0, 0.0, {"preset": "disk"}],
            [0.0, 1.0, 0.1, {"preset": "disk"}],
            [1.0, 1.0, 0.0, {"preset": "disk"}],
        ],
        "edges": [[0, 1], [1, 3], [3, 2], [2, 0], [0, 3]],
        "faces": [[0, 1, 4], [4, 3, 2]],
    }
    mesh = parse_geometry(data)

    cm = ConstraintModuleManager(mesh.constraint_modules)
    cm.enforce_all(mesh, context="minimize", global_params=mesh.global_parameters)

    positions = mesh.positions_view()
    z = positions[:, 2]

    # Planarity: all vertices share the same z after projection.
    assert float(np.ptp(z)) < 1e-12
