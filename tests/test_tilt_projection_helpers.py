import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.minimizer import Minimizer
from runtime.tilt_projection import (
    project_tilts_axisymmetric_about_center,
    project_tilts_to_tangent_array,
)


def test_project_tilts_to_tangent_array_matches_minimizer_static():
    tilts = np.array([[1.0, 2.0, 3.0], [-1.0, 4.0, -2.0]], dtype=float, order="F")
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=float, order="F")
    got = project_tilts_to_tangent_array(tilts, normals)
    ref = Minimizer._project_tilts_to_tangent_array(tilts, normals)
    assert np.allclose(got, ref)


def test_project_tilts_axisymmetric_about_center_matches_minimizer_static():
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
        order="F",
    )
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (3, 1))
    center = np.array([0.0, 0.0, 0.0], dtype=float)
    axis = np.array([0.0, 0.0, 1.0], dtype=float)
    tilts = np.array(
        [[1.0, 2.0, 0.0], [-4.0, 3.0, 0.0], [5.0, 6.0, 0.0]],
        dtype=float,
        order="F",
    )
    fixed_mask = np.array([False, True, False], dtype=bool)

    got = project_tilts_axisymmetric_about_center(
        positions=positions,
        tilts=tilts,
        normals=normals,
        center=center,
        axis=axis,
        fixed_mask=fixed_mask,
    )
    ref = Minimizer._project_tilts_axisymmetric_about_center(
        positions=positions,
        tilts=tilts,
        normals=normals,
        center=center,
        axis=axis,
        fixed_mask=fixed_mask,
    )
    assert np.allclose(got, ref)
