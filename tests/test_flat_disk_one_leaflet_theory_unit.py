import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_one_leaflet_theory import (
    FlatDiskTheoryParams,
    compute_flat_disk_theory,
    quadratic_min_from_scan,
    tex_reference_params,
)


@pytest.mark.unit
def test_flat_disk_theory_exact_reference_values() -> None:
    params = tex_reference_params()
    result = compute_flat_disk_theory(params)

    assert result.lambda_value == pytest.approx(1.0 / 15.0, abs=1e-12)
    assert result.lambda_inverse == pytest.approx(15.0, abs=1e-12)
    assert result.lambda_radius == pytest.approx(7.0, abs=1e-9)
    assert result.coeff_A == pytest.approx(9869.8456, rel=1e-7, abs=1e-4)
    assert result.coeff_B == pytest.approx(12.5663706, rel=1e-8, abs=1e-8)
    assert result.theta_star == pytest.approx(6.366e-4, rel=5e-4, abs=1e-8)
    assert result.elastic_inner == pytest.approx(1.8559e-3, rel=5e-4, abs=1e-8)
    assert result.elastic_outer == pytest.approx(2.1440e-3, rel=5e-4, abs=1e-8)
    assert result.contact == pytest.approx(-7.9998e-3, rel=5e-4, abs=1e-8)
    assert result.total == pytest.approx(-3.9999e-3, rel=5e-4, abs=1e-8)


@pytest.mark.unit
def test_flat_disk_theory_invalid_parameter_guards() -> None:
    with pytest.raises(ValueError, match="kappa must be > 0"):
        compute_flat_disk_theory(
            FlatDiskTheoryParams(
                kappa=0.0, kappa_t=225.0, radius=0.4666666667, drive=1.0
            )
        )
    with pytest.raises(ValueError, match="kappa_t must be > 0"):
        compute_flat_disk_theory(
            FlatDiskTheoryParams(kappa=1.0, kappa_t=0.0, radius=0.4666666667, drive=1.0)
        )
    with pytest.raises(ValueError, match="radius must be > 0"):
        compute_flat_disk_theory(
            FlatDiskTheoryParams(kappa=1.0, kappa_t=225.0, radius=0.0, drive=1.0)
        )


@pytest.mark.unit
def test_theta_scan_quadratic_fit_recovers_known_minimum() -> None:
    theta = np.linspace(-0.5, 0.5, 9)
    energy = (4.0 * theta * theta) - (1.2 * theta) + 0.3

    fit = quadratic_min_from_scan(theta, energy)
    theta_star_exact = 1.2 / 8.0
    energy_star_exact = (
        (4.0 * theta_star_exact * theta_star_exact) - (1.2 * theta_star_exact) + 0.3
    )

    assert fit.coeff_a == pytest.approx(4.0, rel=1e-10, abs=1e-12)
    assert fit.coeff_b == pytest.approx(-1.2, rel=1e-10, abs=1e-12)
    assert fit.theta_star == pytest.approx(theta_star_exact, rel=1e-10, abs=1e-12)
    assert fit.energy_star == pytest.approx(energy_star_exact, rel=1e-10, abs=1e-12)
