import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from modules.energy.bending_tilt_leaflet import _interior_mask_leaflet  # noqa: E402


def _load_mesh():
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_free_disk_coarse_refinable.yaml")
    )
    mesh.build_position_cache()
    return mesh, mesh.global_parameters


def test_interior_mask_cache_reuses_and_invalidates_on_params_and_version() -> None:
    mesh, gp = _load_mesh()
    idx = mesh.vertex_index_to_row

    m1 = _interior_mask_leaflet(mesh, gp, cache_tag="out", index_map=idx)
    m2 = _interior_mask_leaflet(mesh, gp, cache_tag="out", index_map=idx)
    assert m2 is m1

    gp.set("bending_tilt_base_term_boundary_group_out", "disk")
    m3 = _interior_mask_leaflet(mesh, gp, cache_tag="out", index_map=idx)
    assert m3 is not m2
    assert np.count_nonzero(m3) <= np.count_nonzero(m2)

    mesh.increment_topology_version()
    m4 = _interior_mask_leaflet(mesh, gp, cache_tag="out", index_map=idx)
    assert m4 is not m3
