from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_kh_rim_fidelity import (  # noqa: E402
    run_flat_disk_kh_rim_fidelity,
)


@pytest.mark.regression
def test_flat_disk_kh_rim_fidelity_metrics_are_finite_and_bounded() -> None:
    report = run_flat_disk_kh_rim_fidelity(
        optimize_preset="kh_strict_fast",
        refine_level=1,
        rim_local_refine_steps=1,
        rim_local_refine_band_lambda=4.0,
    )
    assert report["meta"]["mode"] == "flat_disk_kh_rim_fidelity"
    assert report["meta"]["parameterization"] == "kh_physical"

    parity = report["parity"]
    assert np.isfinite(float(parity["theta_factor"]))
    assert np.isfinite(float(parity["energy_factor"]))
    assert bool(parity["meets_factor_2"])

    rim = report["rim_fidelity"]
    for key in (
        "jump_abs_median",
        "jump_abs_max",
        "jump_ratio",
        "rim_theta_error_abs_median",
        "rim_theta_error_abs_max",
        "inner_tphi_over_trad_median",
        "outer_tphi_over_trad_median",
    ):
        assert np.isfinite(float(rim[key])), key

    assert float(rim["jump_ratio"]) < 0.30
    assert float(rim["rim_theta_error_abs_median"]) <= 1e-12
    assert float(rim["rim_theta_error_abs_max"]) <= 1e-12
    assert float(rim["inner_tphi_over_trad_median"]) <= 0.03
    assert float(rim["outer_tphi_over_trad_median"]) <= 1.40
