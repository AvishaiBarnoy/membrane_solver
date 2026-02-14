import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from modules.constraints.rim_slope_match_out import _build_matching_data  # noqa: E402
from modules.constraints.tilt_thetaB_boundary_in import (
    _boundary_directions,  # noqa: E402
)


def _load_mesh():
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_free_disk_coarse_refinable.yaml")
    )
    mesh.build_position_cache()
    return mesh, mesh.global_parameters


def test_boundary_directions_cache_reuse_and_invalidate() -> None:
    mesh, gp = _load_mesh()
    pos = mesh.positions_view()

    d1 = _boundary_directions(mesh, gp, positions=pos)
    d2 = _boundary_directions(mesh, gp, positions=pos)
    assert d1 is not None
    assert d2 is not None
    assert d2 is d1

    mesh.increment_version()
    d3 = _boundary_directions(mesh, gp, positions=pos)
    assert d3 is not None
    assert d3 is not d1
    assert np.array_equal(d3[0], d1[0])


def test_rim_matching_data_cache_reuse_and_invalidate() -> None:
    mesh, gp = _load_mesh()
    pos = mesh.positions_view()

    m1 = _build_matching_data(mesh, gp, pos)
    m2 = _build_matching_data(mesh, gp, pos)
    assert m1 is not None
    assert m2 is not None
    assert m2 is m1

    gp.set("rim_slope_match_center", [0.1, 0.0, 0.0])
    m3 = _build_matching_data(mesh, gp, pos)
    assert m3 is not None
    assert m3 is not m1
