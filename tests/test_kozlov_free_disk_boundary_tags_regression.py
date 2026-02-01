import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402


def test_kozlov_free_disk_uses_disk_edge_as_thetaB_boundary_ring() -> None:
    """Regression: thetaB / disk boundary ring is tagged at the disk edge.

    The analytical benchmark in `docs/tex/1_disk_3d.tex` uses the disk edge
    radius R = 7/15 in simulation units. The thetaB contact module and the
    rim-slope matching constraint both key off `rim_slope_match_group: disk`,
    so that group must sit on the outermost disk ring (r=R), not an interior
    disk ring.
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "meshes",
        "caveolin",
        "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml",
    )
    mesh = parse_geometry(load_data(path))
    positions = mesh.positions_view()

    disk_rows = [
        mesh.vertex_index_to_row[int(vid)]
        for vid in mesh.vertex_ids
        if mesh.vertices[int(vid)].options.get("rim_slope_match_group") == "disk"
    ]
    assert len(disk_rows) == 12

    r = np.linalg.norm(positions[np.asarray(disk_rows, dtype=int), :2], axis=1)
    assert np.allclose(r, 7.0 / 15.0, atol=1e-12, rtol=0.0)


# Keep an explicit marker even though conftest auto-applies "regression"
# based on filename; this helps when running a single file directly.
pytestmark = pytest.mark.regression
