import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.minimizer_helpers import (
    capture_diagnostic_state,
    get_cached_tilt_fixed_mask,
    restore_diagnostic_state,
)


def test_get_cached_tilt_fixed_mask_uses_versions_and_vertex_order():
    mesh = parse_geometry(
        {
            "vertices": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, {"tilt_fixed": True}],
                [0.0, 1.0, 0.0],
            ],
            "edges": [[0, 1], [1, 2]],
        }
    )
    mesh.build_position_cache()

    mask0, flags0, verts0 = get_cached_tilt_fixed_mask(
        mesh=mesh,
        flag_attr="tilt_fixed",
        cached_mask=None,
        cached_flags_version=-1,
        cached_vertex_version=-1,
    )
    assert np.array_equal(mask0, np.array([False, True, False], dtype=bool))

    mask1, flags1, verts1 = get_cached_tilt_fixed_mask(
        mesh=mesh,
        flag_attr="tilt_fixed",
        cached_mask=mask0,
        cached_flags_version=flags0,
        cached_vertex_version=verts0,
    )
    assert mask1 is mask0
    assert flags1 == flags0
    assert verts1 == verts0


def test_capture_restore_diagnostic_state_single_tilt():
    mesh = parse_geometry(
        {
            "global_parameters": {"surface_tension": 2.0, "test_key": 5},
            "vertices": [
                [0.0, 0.0, 0.0, {"tilt": [1.0, 0.0]}],
                [1.0, 0.0, 0.0],
            ],
            "edges": [[0, 1]],
        }
    )
    gp = mesh.global_parameters

    snapshot = capture_diagnostic_state(mesh, gp, uses_leaflet_tilts=False)
    mesh.set_tilts_from_array(np.full_like(mesh.tilts_view(), 7.0))
    gp.set("surface_tension", 9.0)
    gp.set("new_key", 42)

    restore_diagnostic_state(mesh, gp, snapshot)

    assert np.allclose(mesh.tilts_view(), snapshot["tilts"])
    assert gp.get("surface_tension") == 2.0
    assert gp.get("test_key") == 5
    assert gp.get("new_key") is None


def test_capture_restore_diagnostic_state_leaflet_tilts():
    mesh = parse_geometry(
        {
            "global_parameters": {"surface_tension": 1.0},
            "energy_modules": ["tilt_in", "tilt_out"],
            "vertices": [
                [
                    0.0,
                    0.0,
                    0.0,
                    {"tilt_in": [1.0, 0.0], "tilt_out": [0.0, 1.0]},
                ],
                [1.0, 0.0, 0.0],
            ],
            "edges": [[0, 1]],
        }
    )
    gp = mesh.global_parameters

    snapshot = capture_diagnostic_state(mesh, gp, uses_leaflet_tilts=True)
    mesh.set_tilts_in_from_array(np.full_like(mesh.tilts_in_view(), 3.0))
    mesh.set_tilts_out_from_array(np.full_like(mesh.tilts_out_view(), -3.0))
    gp.set("surface_tension", 4.0)

    restore_diagnostic_state(mesh, gp, snapshot)

    assert np.allclose(mesh.tilts_in_view(), snapshot["tilts_in"])
    assert np.allclose(mesh.tilts_out_view(), snapshot["tilts_out"])
    assert gp.get("surface_tension") == 1.0
