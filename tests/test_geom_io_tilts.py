import numpy as np

from geometry.geom_io import parse_geometry


def test_parse_geometry_loads_tilts_in_out():
    data = {
        "vertices": [
            [0.0, 0.0, 0.0, {"tilt_in": [1.0, 2.0, 3.0], "tilt_out": [0.1, 0.2, 0.3]}],
            [1.0, 0.0, 0.0, {"tilt_in": [0.5, -0.5], "tilt_out": [1.0, 0.0]}],
            [0.0, 1.0, 0.0, {}],
        ],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "global_parameters": {},
    }

    mesh = parse_geometry(data)
    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()

    # Tilts are projected onto the local tangent plane; for this triangle the
    # normal is +z, so z-components are removed.
    np.testing.assert_allclose(tilts_in[0], np.array([1.0, 2.0, 0.0]))
    np.testing.assert_allclose(tilts_out[0], np.array([0.1, 0.2, 0.0]))
    np.testing.assert_allclose(tilts_in[1], np.array([0.5, -0.5, 0.0]))
    np.testing.assert_allclose(tilts_out[1], np.array([1.0, 0.0, 0.0]))
