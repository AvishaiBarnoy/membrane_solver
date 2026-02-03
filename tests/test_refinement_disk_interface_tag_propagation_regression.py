import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.refinement import refine_triangle_mesh


@pytest.mark.regression
def test_refine_propagates_disk_interface_tags_to_midpoints():
    """Disk boundary tags must survive refinement so interface constraints act on all ring vertices."""
    mesh = parse_geometry(
        load_data("meshes/caveolin/kozlov_free_disk_coarse_refinable.yaml")
    )
    refined = refine_triangle_mesh(mesh)

    disk_edge = [
        v
        for v in refined.vertices.values()
        if (getattr(v, "options", None) or {}).get("preset") == "disk_edge"
    ]
    assert len(disk_edge) >= 8

    # After refinement, all disk_edge vertices should carry the disk interface group tags.
    for v in disk_edge:
        opts = getattr(v, "options", None) or {}
        assert opts.get("rim_slope_match_group") == "disk"
        assert opts.get("tilt_thetaB_group_in") == "disk"
