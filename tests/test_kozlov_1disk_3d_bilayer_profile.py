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
    radial_unit_vectors as _radial_unit_vectors,
)

pytestmark = pytest.mark.e2e


def test_bilayer_profile_tilts_decay_in_outer_region() -> None:
    mesh = parse_geometry(build_1disk_profile_data(bilayer=True))
    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=60)

    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    rim_rows = _collect_group_rows(mesh, "rim_slope_match_group", "rim")
    assert rim_rows.size
    r_rim = float(np.mean(r[rim_rows]))

    # Exclude the rim itself; the asymmetric rim-matching condition allows
    # t_in != t_out at r=R even in the bilayer profile setup.
    mask = r >= (r_rim + 1e-3)
    rows = np.where(mask)[0]
    assert rows.size

    r_hat_outer = _radial_unit_vectors(positions[rows])
    theta_in_outer = np.einsum("ij,ij->i", mesh.tilts_in_view()[rows], r_hat_outer)
    theta_out_outer = np.einsum("ij,ij->i", mesh.tilts_out_view()[rows], r_hat_outer)

    inner_mask = r <= (r_rim + 1e-6)
    inner_rows = np.where(inner_mask)[0]
    assert inner_rows.size
    r_hat_inner = _radial_unit_vectors(positions[inner_rows])
    theta_in_inner = np.einsum(
        "ij,ij->i", mesh.tilts_in_view()[inner_rows], r_hat_inner
    )
    theta_out_inner = np.einsum(
        "ij,ij->i", mesh.tilts_out_view()[inner_rows], r_hat_inner
    )

    # Expect decay in both leaflets away from the disk.
    outer_in_p90 = float(np.quantile(np.abs(theta_in_outer), 0.9))
    outer_out_p90 = float(np.quantile(np.abs(theta_out_outer), 0.9))
    inner_in_p90 = float(np.quantile(np.abs(theta_in_inner), 0.9))
    inner_out_p90 = float(np.quantile(np.abs(theta_out_inner), 0.9))

    assert outer_in_p90 < 0.3 * (inner_in_p90 + 1e-12)
    assert outer_out_p90 < 0.3 * (inner_out_p90 + 1e-12)
