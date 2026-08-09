import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.free_disk_profile_fits import analyze_mesh_profiles  # noqa: E402


@pytest.mark.acceptance
def test_free_disk_two_stage_protocol_produces_fittable_radial_profiles(
    canonical_profile_protocol_result,
) -> None:
    mesh, theta_b = canonical_profile_protocol_result
    assert theta_b > 0.0

    report = analyze_mesh_profiles(mesh, bins=16, flip_tilt_out=True)

    outer_fit = report["outer_fit_K1"]
    curvature_fit = report["curvature_fit_K0"]
    height_fit = report["height_fit"]

    assert outer_fit is not None
    assert curvature_fit is not None
    assert height_fit is not None

    assert float(outer_fit["rel_rmse"]) < 0.01
    assert float(curvature_fit["rel_rmse"]) < 0.02
    assert float(height_fit["rel_rmse"]) < 0.30

    for fit in (outer_fit, curvature_fit, height_fit):
        residuals = fit["residual_bands"]
        assert residuals is not None
        assert residuals["dominant_band"] in {"near", "mid", "far"}
        for label in ("near", "mid", "far"):
            band = residuals["bands"][label]
            assert int(band["count"]) >= 0
