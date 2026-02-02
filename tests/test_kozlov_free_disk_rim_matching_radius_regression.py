def test_kozlov_free_disk_rim_matching_groups_use_disk_edge_and_unpinned_outer_ring() -> (
    None
):
    """Regression: option merging produces expected rim-matching groups/constraints.

    This test is intentionally mesh-independent (no `meshes/` YAML load). It
    checks the invariants relied upon by the Kozlov free-disk configuration:
    - the disk-edge ring is both circle-pinned and planar,
    - an intermediate outer ring is tagged for rim matching but left unpinned.
    """
    import os
    import sys

    import numpy as np

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from geometry.geom_io import parse_geometry  # noqa: E402

    data = {
        "global_parameters": {
            "rim_slope_match_group": "disk",
            "rim_slope_match_outer_group": "outer",
            "rim_slope_match_disk_group": "disk",
        },
        "definitions": {
            "disk_edge": {
                "rim_slope_match_group": "disk",
                "constraints": ["pin_to_plane", "pin_to_circle"],
            },
            "outer_ring": {
                "rim_slope_match_group": "outer",
                "constraints": [],
            },
        },
        "vertices": [],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
    }

    # 12-gon disk edge at r=7/15.
    for k in range(12):
        theta = 2.0 * np.pi * k / 12.0
        x = (7.0 / 15.0) * float(np.cos(theta))
        y = (7.0 / 15.0) * float(np.sin(theta))
        data["vertices"].append([x, y, 0.0, {"preset": "disk_edge"}])

    # 12-gon outer sampling ring at r=1.
    for k in range(12):
        theta = 2.0 * np.pi * k / 12.0
        x = 1.0 * float(np.cos(theta))
        y = 1.0 * float(np.sin(theta))
        data["vertices"].append([x, y, 0.0, {"preset": "outer_ring"}])

    mesh = parse_geometry(data)

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

    # Disk edge is pinned by both plane and circle (order-insensitive).
    disk_constraints = {
        tuple(sorted(mesh.vertices[v].options.get("constraints") or []))
        for v in disk_vids
    }
    assert disk_constraints == {("pin_to_circle", "pin_to_plane")}

    # Outer ring is unpinned.
    outer_constraints = {
        tuple(sorted(mesh.vertices[v].options.get("constraints") or []))
        for v in outer_vids
    }
    assert outer_constraints == {()}
