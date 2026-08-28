import numpy as np
import pytest

from geometry.geom_io import parse_geometry
from tests.kozlov_test_utils import (
    build_1disk_profile_data,
)
from tests.kozlov_test_utils import (
    build_minimizer as _build_minimizer,
)
from tests.kozlov_test_utils import (
    collect_group_rows as _collect_group_rows,
)
from tests.kozlov_test_utils import (
    order_by_angle as _order_by_angle,
)
from tests.kozlov_test_utils import (
    outer_free_ring_rows as _outer_free_ring_rows,
)
from tests.kozlov_test_utils import (
    radial_unit_vectors as _radial_unit_vectors,
)

pytestmark = pytest.mark.e2e


def test_single_leaflet_profile_behavior() -> None:
    mesh = parse_geometry(build_1disk_profile_data(bilayer=False))
    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=60)

    positions = mesh.positions_view()
    z_span = float(np.ptp(positions[:, 2]))
    assert z_span > 1e-4

    disk_rows = _collect_group_rows(mesh, "tilt_disk_target_group_in", "disk")
    assert disk_rows.size

    rim_rows = _collect_group_rows(mesh, "rim_slope_match_group", "rim")
    outer_rows = _collect_group_rows(mesh, "rim_slope_match_group", "outer")
    disk_ring_rows = _collect_group_rows(mesh, "rim_slope_match_group", "disk")
    assert rim_rows.size and outer_rows.size and disk_ring_rows.size

    rim_rows = rim_rows[_order_by_angle(positions[rim_rows])]
    outer_rows = outer_rows[_order_by_angle(positions[outer_rows])]

    rim_pos = positions[rim_rows]
    outer_pos = positions[outer_rows]
    r_rim = np.linalg.norm(rim_pos[:, :2], axis=1)
    r_outer = np.linalg.norm(outer_pos[:, :2], axis=1)
    dr = np.maximum(r_outer - r_rim, 1e-6)
    phi = float(np.mean((outer_pos[:, 2] - rim_pos[:, 2]) / dr))
    assert abs(phi) > 1e-4

    r_disk = np.linalg.norm(positions[disk_rows, :2], axis=1)
    r_max = float(np.max(r_disk))
    r_hat_disk = _radial_unit_vectors(positions[disk_rows])
    theta_disk = np.einsum(
        "ij,ij->i",
        mesh.tilts_in_view()[disk_rows],
        r_hat_disk,
    )
    inner_band = theta_disk[r_disk < 0.4 * r_max]
    outer_band = theta_disk[r_disk > 0.8 * r_max]
    assert float(np.mean(outer_band)) > float(np.mean(inner_band))

    rim_r_hat = _radial_unit_vectors(rim_pos)
    theta_in_rim = float(
        np.mean(np.einsum("ij,ij->i", mesh.tilts_in_view()[rim_rows], rim_r_hat))
    )
    theta_out_rim = float(
        np.mean(np.einsum("ij,ij->i", mesh.tilts_out_view()[rim_rows], rim_r_hat))
    )
    disk_ring_r_hat = _radial_unit_vectors(positions[disk_ring_rows])
    theta_disk_ring = float(
        np.mean(
            np.einsum(
                "ij,ij->i",
                mesh.tilts_in_view()[disk_ring_rows],
                disk_ring_r_hat,
            )
        )
    )
    denom = max(abs(theta_in_rim), abs(theta_disk_ring - phi), 1e-6)
    assert abs(theta_in_rim - (theta_disk_ring - phi)) / denom < 0.6
    assert abs(theta_out_rim) > 1e-4

    free_rows = _outer_free_ring_rows(mesh, positions)
    assert free_rows.size
    free_r_hat = _radial_unit_vectors(positions[free_rows])
    theta_out_far = float(
        np.mean(
            np.abs(
                np.einsum(
                    "ij,ij->i",
                    mesh.tilts_out_view()[free_rows],
                    free_r_hat,
                )
            )
        )
    )
    assert theta_out_far < 0.7 * abs(theta_out_rim)
