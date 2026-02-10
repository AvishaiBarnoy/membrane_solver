import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.refinement import (  # noqa: E402
    refine_polygonal_facets,
    refine_triangle_mesh,
)


def test_kozlov_free_disk_uses_shared_rim_ring_at_disk_edge() -> None:
    """Regression: the disk/membrane interface ring sits at r=R and is shared.

    The analytical benchmark in `docs/tex/1_disk_3d.tex` uses the disk edge
    radius R = 7/15 in simulation units. The Kozlov free-disk mesh should use
    a *single* ring at r=R that is both the disk boundary and the rim-matching
    ring (no extra rim ring outside the disk).
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

    rim_vids = [
        int(vid)
        for vid in mesh.vertex_ids
        if mesh.vertices[int(vid)].options.get("rim_slope_match_group") == "rim"
    ]
    assert len(rim_vids) == 24

    rim_rows = np.asarray([mesh.vertex_index_to_row[v] for v in rim_vids], dtype=int)
    r_disk = np.linalg.norm(positions[rim_rows, :2], axis=1)
    assert np.allclose(r_disk, 7.0 / 15.0, atol=1e-12, rtol=0.0)

    # Shared ring: rim-matching vertices should belong to the disk preset.
    rim_presets = {mesh.vertices[int(vid)].options.get("preset") for vid in rim_vids}
    assert rim_presets == {"disk"}


def test_kozlov_free_disk_refinement_keeps_disk_vertices_in_rigid_group() -> None:
    """Disk preset vertices must remain tagged for rigid-disk constraint."""
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "meshes",
        "caveolin",
        "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml",
    )
    mesh = parse_geometry(load_data(path))
    refined = refine_triangle_mesh(refine_polygonal_facets(mesh))

    disk_vertices = [
        v
        for v in refined.vertices.values()
        if (getattr(v, "options", None) or {}).get("preset") == "disk"
    ]
    assert disk_vertices

    for v in disk_vertices:
        opts = getattr(v, "options", None) or {}
        assert opts.get("rigid_disk_group") == "disk"


# Keep an explicit marker even though conftest auto-applies "regression"
# based on filename; this helps when running a single file directly.
pytestmark = pytest.mark.regression
