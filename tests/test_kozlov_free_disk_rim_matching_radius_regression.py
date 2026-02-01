import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402


def test_kozlov_free_disk_rim_matching_groups_use_disk_edge_and_unpinned_outer_ring() -> (
    None
):
    """Regression: rim matching uses the disk edge (R=7/15) and a free outer ring.

    For theory parity with `docs/tex/1_disk_3d.tex`, the rim matching constraint
    (`rim_slope_match_out`) should treat the disk edge (r=R) as the rim, and
    estimate phi(R) from a nearby outer ring. That intermediate outer ring
    should *not* be pinned to a circle (only the disk boundary and far-field
    boundary are geometric pin circles).
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "meshes",
        "caveolin",
        "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml",
    )
    mesh = parse_geometry(load_data(path))

    assert mesh.global_parameters.get("rim_slope_match_group") == "disk"
    assert mesh.global_parameters.get("rim_slope_match_outer_group") == "outer"
    assert mesh.global_parameters.get("rim_slope_match_disk_group") == "disk"

    positions = mesh.positions_view()

    disk_vids = [
        int(vid)
        for vid in mesh.vertex_ids
        if mesh.vertices[int(vid)].options.get("rim_slope_match_group") == "disk"
    ]
    outer_vids = [
        int(vid)
        for vid in mesh.vertex_ids
        if mesh.vertices[int(vid)].options.get("rim_slope_match_group") == "outer"
    ]

    assert len(disk_vids) == 12
    assert len(outer_vids) == 12

    disk_rows = np.asarray([mesh.vertex_index_to_row[v] for v in disk_vids], dtype=int)
    outer_rows = np.asarray(
        [mesh.vertex_index_to_row[v] for v in outer_vids], dtype=int
    )

    disk_r = np.linalg.norm(positions[disk_rows, :2], axis=1)
    outer_r = np.linalg.norm(positions[outer_rows, :2], axis=1)

    assert np.allclose(disk_r, 7.0 / 15.0, atol=1e-12, rtol=0.0)
    assert np.allclose(outer_r, 1.0, atol=1e-12, rtol=0.0)

    # Disk edge must be pinned to a circle to represent a rigid disk boundary.
    assert {
        tuple(mesh.vertices[v].options.get("constraints") or []) for v in disk_vids
    } == {("pin_to_circle",)}

    # The intermediate outer ring is a sampling ring (free membrane), so it must not
    # be circle-pinned.
    assert {
        tuple(mesh.vertices[v].options.get("constraints") or []) for v in outer_vids
    } == {()}
