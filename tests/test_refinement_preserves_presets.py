import os
import sys

sys.path.insert(0, os.getcwd())

from geometry.geom_io import load_data, parse_geometry
from runtime.refinement import refine_triangle_mesh


def test_refine_triangle_mesh_preserves_presets():
    data = load_data("tests/fixtures/kozlov_free_disk_coarse_refinable.yaml")
    mesh = parse_geometry(data)

    refined = refine_triangle_mesh(mesh)
    assert refined.vertices

    preset_count = 0
    for vertex in refined.vertices.values():
        opts = vertex.options or {}
        if "preset" in opts:
            preset_count += 1

    assert preset_count == len(refined.vertices)
