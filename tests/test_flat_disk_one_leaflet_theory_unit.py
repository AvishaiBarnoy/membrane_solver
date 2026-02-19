import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_one_leaflet_theory import (
    FlatDiskTheoryParams,
    compute_flat_disk_theory,
    physical_to_dimensionless_theory_params,
    quadratic_min_from_scan,
    solver_mapping_from_theory,
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


@pytest.mark.unit
def test_physical_to_dimensionless_theory_params_tex_scaling() -> None:
    params = physical_to_dimensionless_theory_params(
        kappa_physical=10.0,
        kappa_t_physical=10.0,
        radius_physical=7.0,
        drive_physical=(2.0 / 0.7),
        length_scale=15.0,
    )

    assert params.kappa == pytest.approx(1.0, abs=1e-12)
    assert params.kappa_t == pytest.approx(225.0, abs=1e-12)
    assert params.radius == pytest.approx(7.0 / 15.0, abs=1e-12)
    assert params.drive == pytest.approx(4.285714286, rel=1e-9, abs=1e-12)


@pytest.mark.unit
def test_physical_to_dimensionless_theory_params_guards() -> None:
    with pytest.raises(ValueError, match="kappa_physical must be > 0"):
        physical_to_dimensionless_theory_params(
            kappa_physical=0.0,
            kappa_t_physical=10.0,
            radius_physical=7.0,
            drive_physical=1.0,
            length_scale=15.0,
        )

    with pytest.raises(ValueError, match="kappa_t_physical must be > 0"):
        physical_to_dimensionless_theory_params(
            kappa_physical=10.0,
            kappa_t_physical=0.0,
            radius_physical=7.0,
            drive_physical=1.0,
            length_scale=15.0,
        )

    with pytest.raises(ValueError, match="radius_physical must be > 0"):
        physical_to_dimensionless_theory_params(
            kappa_physical=10.0,
            kappa_t_physical=10.0,
            radius_physical=0.0,
            drive_physical=1.0,
            length_scale=15.0,
        )

    with pytest.raises(ValueError, match="length_scale must be > 0"):
        physical_to_dimensionless_theory_params(
            kappa_physical=10.0,
            kappa_t_physical=10.0,
            radius_physical=7.0,
            drive_physical=1.0,
            length_scale=0.0,
        )


@pytest.mark.unit
def test_solver_mapping_from_theory_supports_legacy_and_kh_physical() -> None:
    params = FlatDiskTheoryParams(
        kappa=1.0, kappa_t=225.0, radius=7.0 / 15.0, drive=1.0
    )

    legacy = solver_mapping_from_theory(params, parameterization="legacy")
    assert legacy["bending_modulus_in"] == pytest.approx(225.0)
    assert legacy["tilt_modulus_in"] == pytest.approx(50625.0)

    kh = solver_mapping_from_theory(params, parameterization="kh_physical")
    assert kh["bending_modulus_in"] == pytest.approx(1.0)
    assert kh["tilt_modulus_in"] == pytest.approx(225.0)

    with pytest.raises(ValueError, match="parameterization must be"):
        solver_mapping_from_theory(params, parameterization="unknown_mode")
