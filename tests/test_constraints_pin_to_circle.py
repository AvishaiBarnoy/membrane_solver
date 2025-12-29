import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager


def test_pin_to_circle_projects_vertices_and_edges() -> None:
    radius = 2.0
    data = {
        "vertices": [
            [3.0, 0.0, 5.0, {"constraints": ["pin_to_circle"]}],
            [0.0, 3.0, -1.0],  # via edge constraint
            [0.0, 0.0, 0.0],  # via edge constraint
            [0.0, 0.0, 0.0],  # untouched
        ],
        "edges": [
            [1, 2, {"constraints": ["pin_to_circle"]}],
        ],
        "faces": [],
        "global_parameters": {
            "pin_to_circle_normal": [0.0, 0.0, 1.0],
            "pin_to_circle_point": [0.0, 0.0, 0.0],
            "pin_to_circle_radius": radius,
        },
    }
    mesh = parse_geometry(data)
    manager = ConstraintModuleManager(mesh.constraint_modules)
    manager.enforce_all(mesh)

    center = np.array([0.0, 0.0, 0.0])
    for vidx in (0, 1, 2):
        pos = mesh.vertices[vidx].position
        assert np.isclose(pos[2], 0.0)
        assert np.isclose(np.linalg.norm(pos - center), radius)

    assert np.allclose(mesh.vertices[3].position, np.array([0.0, 0.0, 0.0]))


def test_pin_to_circle_fit_allows_rim_to_move() -> None:
    radius = 2.0
    center = np.array([10.0, 5.0, -3.0], dtype=float)
    normal = np.array([0.0, 1.0, 0.0], dtype=float)  # y=const plane

    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    points = np.stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + np.zeros_like(angles),
            center[2] + radius * np.sin(angles),
        ],
        axis=1,
    )

    # Add small noise so the fitter has something to correct.
    rng = np.random.default_rng(123)
    points = points + 0.02 * rng.normal(size=points.shape)

    data = {
        "vertices": [
            [
                float(p[0]),
                float(p[1]),
                float(p[2]),
                {
                    "constraints": ["pin_to_circle"],
                    "pin_to_circle_group": "rim",
                },
            ]
            for p in points
        ],
        "edges": [[0, 1]],
        "faces": [],
        "global_parameters": {
            "pin_to_circle_mode": "fit",
            "pin_to_circle_normal": normal.tolist(),
            "pin_to_circle_radius": radius,
        },
    }
    mesh = parse_geometry(data)
    manager = ConstraintModuleManager(mesh.constraint_modules)
    manager.enforce_all(mesh)

    # The circle is not pinned to the origin; it should remain near the input center.
    y_mean = float(np.mean([v.position[1] for v in mesh.vertices.values()]))
    assert np.isclose(y_mean, center[1], atol=0.1)

    # All constrained points should be at the requested radius around the fitted center.
    # Use the module's internal fitter so the test stays consistent with implementation.
    from modules.constraints import pin_to_circle as mod

    pts = np.array([v.position for v in mesh.vertices.values()], dtype=float)
    fitted = mod._fit_circle_in_plane(pts, normal, radius_fixed=radius)
    assert fitted is not None
    fitted_center, fitted_radius = fitted
    assert np.isclose(fitted_radius, radius, atol=1e-6)

    distances = np.linalg.norm(pts - fitted_center[None, :], axis=1)
    assert np.allclose(distances, radius, atol=1e-6)
