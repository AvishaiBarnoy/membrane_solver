import os
import sys

sys.path.insert(0, os.getcwd())

from geometry.geom_io import load_data, parse_geometry, save_geometry


def test_save_geometry_preserves_vertex_presets(tmp_path):
    base = load_data("tests/fixtures/kozlov_free_disk_coarse_refinable.yaml")
    mesh = parse_geometry(base)

    out_path = tmp_path / "saved.yaml"
    save_geometry(mesh, str(out_path), compact=True)

    out = load_data(out_path)
    vertices = out.get("vertices") or []
    assert vertices

    preset_count = 0
    for v in vertices:
        if isinstance(v, list) and v and isinstance(v[-1], dict):
            if "preset" in v[-1]:
                preset_count += 1

    assert preset_count == len(vertices)
