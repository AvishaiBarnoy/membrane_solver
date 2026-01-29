import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry


def _single_tilt_mesh() -> dict:
    return {
        "vertices": {
            0: [0.0, 0.0, 0.0, {"tilt": [1.0, 0.0]}],
            1: [1.0, 0.0, 0.0, {"tilt": [0.0, 1.0]}],
            2: [0.0, 1.0, 0.0, {"tilt": [0.0, 0.0]}],
        },
        "edges": {1: [0, 1], 2: [1, 2], 3: [2, 0]},
        "faces": {0: [1, 2, 3]},
        "energy_modules": ["tilt"],
        "global_parameters": {"tilt_rigidity": 1.0},
        "instructions": [],
    }


def _leaflet_tilt_mesh() -> dict:
    return {
        "vertices": {
            0: [0.0, 0.0, 0.0, {"tilt_in": [1.0, 0.0], "tilt_out": [0.5, 0.2]}],
            1: [1.0, 0.0, 0.0, {"tilt_in": [0.2, 0.1], "tilt_out": [0.1, -0.2]}],
            2: [0.0, 1.0, 0.0, {"tilt_in": [0.0, 0.0], "tilt_out": [0.0, 0.0]}],
        },
        "edges": {1: [0, 1], 2: [1, 2], 3: [2, 0]},
        "faces": {0: [1, 2, 3]},
        "energy_modules": ["tilt_in", "tilt_out"],
        "global_parameters": {"tilt_modulus_in": 1.0, "tilt_modulus_out": 1.0},
        "instructions": [],
    }


def test_tilt_soa_roundtrip() -> None:
    mesh = parse_geometry(_single_tilt_mesh())
    row = mesh.vertex_index_to_row[1]

    tilts = mesh.tilts_view()
    tilts[row] = np.array([0.2, 0.4, 0.0])
    np.testing.assert_allclose(mesh.vertices[1].tilt, np.array([0.2, 0.4, 0.0]))

    mesh.vertices[1].tilt = np.array([0.6, 0.1, 0.0])
    np.testing.assert_allclose(mesh.tilts_view()[row], np.array([0.6, 0.1, 0.0]))


def test_leaflet_tilt_soa_roundtrip() -> None:
    mesh = parse_geometry(_leaflet_tilt_mesh())
    row = mesh.vertex_index_to_row[0]

    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    tilts_in[row] = np.array([0.3, 0.2, 0.0])
    tilts_out[row] = np.array([0.1, -0.3, 0.0])
    np.testing.assert_allclose(mesh.vertices[0].tilt_in, np.array([0.3, 0.2, 0.0]))
    np.testing.assert_allclose(mesh.vertices[0].tilt_out, np.array([0.1, -0.3, 0.0]))

    mesh.vertices[0].tilt_in = np.array([0.9, 0.0, 0.0])
    mesh.vertices[0].tilt_out = np.array([0.0, 0.7, 0.0])
    np.testing.assert_allclose(mesh.tilts_in_view()[row], np.array([0.9, 0.0, 0.0]))
    np.testing.assert_allclose(mesh.tilts_out_view()[row], np.array([0.0, 0.7, 0.0]))
