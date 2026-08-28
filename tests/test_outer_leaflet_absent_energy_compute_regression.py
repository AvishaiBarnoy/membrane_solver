import os

import numpy as np
import pytest

from commands.context import CommandContext  # noqa: E402
from commands.executor import execute_command_line  # noqa: E402
from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from tests.minimizer_test_utils import build_minimizer

PROFILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "meshes",
    "caveolin",
    "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml",
)


def _build_profile_case():
    mesh = parse_geometry(load_data(PROFILE_PATH))
    return mesh, build_minimizer(mesh)


@pytest.mark.regression
def test_energy_compute_does_not_crash_with_outer_leaflet_absent_disk() -> None:
    """Regression: outer-leaflet masking must not crash energy evaluation."""
    _mesh, minim = _build_profile_case()

    # This call previously crashed due to vertex normal computation using
    # full-triangle normals while passing a subset tri_rows.
    energy = float(minim.compute_energy())
    assert np.isfinite(energy)


@pytest.mark.regression
def test_profile_relax_light_state_energy_matches_breakdown_after_g1() -> None:
    """Scalar and breakdown energy must use the same post-command state."""
    mesh, minim = _build_profile_case()
    ctx = CommandContext(mesh=mesh, minimizer=minim, stepper=minim.stepper)

    for command in ("g10", "r", "V2", "t5e-3", "g8", "t2e-3", "g1"):
        execute_command_line(ctx, command)

    scalar_energy = float(minim.compute_energy())
    breakdown_total = float(sum(minim.compute_energy_breakdown().values()))

    assert scalar_energy == pytest.approx(breakdown_total, rel=1e-9, abs=1e-9)
