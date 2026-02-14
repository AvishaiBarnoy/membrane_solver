import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from modules.energy.bending_tilt_leaflet import _collect_group_rows  # noqa: E402


def test_collect_group_rows_cache_hit_and_invalidation() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    mesh.build_position_cache()
    index_map = mesh.vertex_index_to_row

    rows = _collect_group_rows(mesh, group="disk", index_map=index_map)
    assert rows.size > 0

    cache = getattr(mesh, "_bending_tilt_group_rows_cache")
    sentinel = np.array([123456], dtype=int)
    cache["rows"] = sentinel

    # Same mesh vertex-id version + group should hit cache.
    cached_rows = _collect_group_rows(mesh, group="disk", index_map=index_map)
    assert cached_rows is sentinel

    # Simulate topology/version change and verify recomputation occurs.
    mesh._vertex_ids_version += 1
    refreshed_rows = _collect_group_rows(mesh, group="disk", index_map=index_map)
    assert refreshed_rows is not sentinel
