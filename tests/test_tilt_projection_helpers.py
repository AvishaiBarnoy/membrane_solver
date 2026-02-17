import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.minimizer import Minimizer
from runtime.tilt_projection import (
    build_leaflet_trial_tilts,
    project_leaflet_tilts_with_optional_axisymmetry,
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


def test_project_leaflet_tilts_with_optional_axisymmetry_enabled_and_disabled():
    positions = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
        order="F",
    )
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (2, 1))
    tilts_in = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]], dtype=float, order="F")
    tilts_out = np.array([[-1.0, -2.0, 0.0], [-3.0, -4.0, 0.0]], dtype=float, order="F")

    gp_off = {"tilt_axisymmetric_about_thetaB_center": False}
    out_in, out_out = project_leaflet_tilts_with_optional_axisymmetry(
        global_params=gp_off,
        positions=positions,
        normals=normals,
        tilts_in=tilts_in,
        tilts_out=tilts_out,
    )
    assert out_in is tilts_in
    assert out_out is tilts_out

    gp_on = {
        "tilt_axisymmetric_about_thetaB_center": True,
        "tilt_thetaB_center": [0.0, 0.0, 0.0],
        "tilt_thetaB_normal": [0.0, 0.0, 1.0],
    }
    proj_in, proj_out = project_leaflet_tilts_with_optional_axisymmetry(
        global_params=gp_on,
        positions=positions,
        normals=normals,
        tilts_in=tilts_in,
        tilts_out=tilts_out,
        fixed_mask_in=np.array([False, True], dtype=bool),
        fixed_mask_out=np.array([False, False], dtype=bool),
    )
    assert np.allclose(proj_in[1], tilts_in[1])
    assert np.allclose(proj_out[:, 2], np.zeros(2))


def test_build_leaflet_trial_tilts_projects_and_restores_fixed_values():
    base_in = np.array([[1.0, 2.0, 3.0], [2.0, 0.0, -1.0]], dtype=float, order="F")
    base_out = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=float, order="F")
    delta_in = np.array([[0.5, -0.5, 0.0], [1.0, -1.0, 2.0]], dtype=float, order="F")
    delta_out = np.array([[-1.0, 0.0, 1.0], [0.5, 0.5, -2.0]], dtype=float, order="F")
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=float, order="F")
    fixed_mask_in = np.array([False, True], dtype=bool)
    fixed_mask_out = np.array([True, False], dtype=bool)
    fixed_vals_in = np.array([[9.0, 9.0, 9.0]], dtype=float)
    fixed_vals_out = np.array([[-7.0, -7.0, -7.0]], dtype=float)

    trial_in, trial_out = build_leaflet_trial_tilts(
        base_in=base_in,
        base_out=base_out,
        delta_in=delta_in,
        delta_out=delta_out,
        normals=normals,
        fixed_mask_in=fixed_mask_in,
        fixed_mask_out=fixed_mask_out,
        fixed_vals_in=fixed_vals_in,
        fixed_vals_out=fixed_vals_out,
    )

    assert np.allclose(trial_in[1], fixed_vals_in[0])
    assert np.allclose(trial_out[0], fixed_vals_out[0])
    assert abs(float(np.dot(trial_in[0], normals[0]))) < 1e-12
    assert abs(float(np.dot(trial_out[1], normals[1]))) < 1e-12
