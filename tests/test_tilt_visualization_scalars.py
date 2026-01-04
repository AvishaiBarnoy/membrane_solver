import os
import sys

import matplotlib

# Use a non-interactive backend suitable for testing.
matplotlib.use("Agg")

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from visualization.plotting import triangle_tilt_divergence, triangle_tilt_magnitudes


def _single_triangle_data(tilts):
    return {
        "vertices": [
            [0.0, 0.0, 0.0, {"tilt": list(tilts[0])}],
            [1.0, 0.0, 0.0, {"tilt": list(tilts[1])}],
            [0.0, 1.0, 0.0, {"tilt": list(tilts[2])}],
        ],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "global_parameters": {"surface_tension": 0.0, "volume_constraint_mode": "none"},
        "instructions": [],
    }


def test_triangle_tilt_magnitudes_returns_mean_vertex_magnitude():
    tilts = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mesh = parse_geometry(_single_triangle_data(tilts))

    values, facet_ids = triangle_tilt_magnitudes(mesh)

    assert facet_ids == [0]
    assert values.shape == (1,)
    assert values[0] == pytest.approx(1.0 / 3.0)


def test_triangle_tilt_divergence_constant_field_is_zero():
    tilts = np.array([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
    mesh = parse_geometry(_single_triangle_data(tilts))

    values, facet_ids = triangle_tilt_divergence(mesh)

    assert facet_ids == [0]
    assert values.shape == (1,)
    assert values[0] == pytest.approx(0.0)


def test_triangle_tilt_divergence_xy_field_matches_two():
    # On the unit right triangle, the nodal field t=(x,y,0) has constant div(t)=2.
    tilts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = parse_geometry(_single_triangle_data(tilts))

    values, _ = triangle_tilt_divergence(mesh)

    assert values[0] == pytest.approx(2.0)
