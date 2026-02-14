import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402


@pytest.mark.regression
def test_set_tilts_leaflets_sync_vertex_views_without_copy_drift() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    mesh.build_position_cache()
    n = len(mesh.vertex_ids)
    rng = np.random.default_rng(42)

    tin = rng.normal(size=(n, 3))
    tout = rng.normal(size=(n, 3))
    mesh.set_tilts_in_from_array(tin)
    mesh.set_tilts_out_from_array(tout)

    np.testing.assert_allclose(mesh.tilts_in_view(), tin, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(mesh.tilts_out_view(), tout, rtol=0.0, atol=0.0)

    sample_rows = [0, n // 2, n - 1]
    for row in sample_rows:
        vid = int(mesh.vertex_ids[row])
        vertex = mesh.vertices[vid]
        np.testing.assert_allclose(vertex.tilt_in, tin[row], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(vertex.tilt_out, tout[row], rtol=0.0, atol=0.0)

    # Mutating cache-backed arrays should remain visible through vertex accessors.
    mesh.tilts_in_view()[sample_rows[1], :] = np.array([9.0, -7.0, 5.0], dtype=float)
    mesh.tilts_out_view()[sample_rows[2], :] = np.array([-3.0, 4.0, -8.0], dtype=float)
    mid_vid = int(mesh.vertex_ids[sample_rows[1]])
    last_vid = int(mesh.vertex_ids[sample_rows[2]])
    np.testing.assert_allclose(
        mesh.vertices[mid_vid].tilt_in, np.array([9.0, -7.0, 5.0], dtype=float)
    )
    np.testing.assert_allclose(
        mesh.vertices[last_vid].tilt_out, np.array([-3.0, 4.0, -8.0], dtype=float)
    )
