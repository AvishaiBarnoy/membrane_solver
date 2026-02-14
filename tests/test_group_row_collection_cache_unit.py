import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from modules.constraints.rim_slope_match_out import (
    _collect_group_rows as _rim_rows,  # noqa: E402
)
from modules.constraints.tilt_thetaB_boundary_in import (  # noqa: E402
    _collect_group_rows as _theta_boundary_rows,
)
from modules.energy.tilt_thetaB_contact_in import (
    _collect_group_rows as _theta_contact_rows,  # noqa: E402
)


def _assert_cache_behavior(
    mesh,
    *,
    group: str,
    collector,
    cache_attr: str,
) -> None:
    rows = collector(mesh, group)
    assert rows.size > 0

    cache = getattr(mesh, cache_attr)
    sentinel = np.array([999_999], dtype=int)
    cache["rows"] = sentinel

    cached_rows = collector(mesh, group)
    assert cached_rows is sentinel

    mesh._vertex_ids_version += 1
    refreshed_rows = collector(mesh, group)
    assert refreshed_rows is not sentinel


def test_group_row_collectors_cache_and_invalidate() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    mesh.build_position_cache()

    _assert_cache_behavior(
        mesh,
        group="disk",
        collector=_theta_contact_rows,
        cache_attr="_tilt_thetaB_contact_group_rows_cache",
    )
    _assert_cache_behavior(
        mesh,
        group="disk",
        collector=_theta_boundary_rows,
        cache_attr="_tilt_thetaB_boundary_group_rows_cache",
    )
    _assert_cache_behavior(
        mesh,
        group="rim",
        collector=_rim_rows,
        cache_attr="_rim_slope_match_group_rows_cache",
    )
