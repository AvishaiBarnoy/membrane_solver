import os
import sys

import numpy as np
import pytest
from scipy import special

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.flat_disk_one_leaflet_theory import (
    FlatDiskTheoryParams,
    build_kh_outer_finite_bvp_profile,
    compute_flat_disk_kh_physical_theory,
    compute_flat_disk_theory,
    evaluate_flat_disk_scalar_tex_profile,
    evaluate_flat_disk_vector_kh_profile,
    evaluate_flat_disk_vector_smoothness_profile,
    physical_to_dimensionless_theory_params,
    quadratic_min_from_scan,
    solve_kh_outer_finite_bvp_coefficients,
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


@pytest.mark.unit
def test_flat_disk_kh_physical_theory_reference_values() -> None:
    """Lock kh_physical lane reference to strict KH radial-integral theory."""
    params = physical_to_dimensionless_theory_params(
        kappa_physical=10.0,
        kappa_t_physical=10.0,
        radius_physical=7.0,
        drive_physical=(2.0 / 0.7),
        length_scale=15.0,
    )
    result = compute_flat_disk_kh_physical_theory(params)

    assert result.lambda_value == pytest.approx(1.0 / 15.0, abs=1e-12)
    assert result.lambda_inverse == pytest.approx(15.0, abs=1e-12)
    assert result.lambda_radius == pytest.approx(7.0, abs=1e-9)
    assert result.coeff_A == pytest.approx(44.32880989, rel=1e-7, abs=1e-8)
    assert result.theta_star == pytest.approx(0.1417404465, rel=5e-7, abs=1e-10)
    assert result.elastic_inner == pytest.approx(0.4773577391, rel=5e-7, abs=1e-10)
    assert result.elastic_outer == pytest.approx(0.4132237519, rel=5e-7, abs=1e-10)
    assert result.total == pytest.approx(-0.8905814910, rel=5e-7, abs=1e-10)


@pytest.mark.unit
def test_kh_outer_finite_bvp_converges_to_infinite_profile_at_large_rmax() -> None:
    params = physical_to_dimensionless_theory_params(
        kappa_physical=10.0,
        kappa_t_physical=10.0,
        radius_physical=7.0,
        drive_physical=(2.0 / 0.7),
        length_scale=15.0,
    )
    result = compute_flat_disk_kh_physical_theory(params)
    radius = float(result.radius)
    lam = float(result.lambda_value)
    theta = 0.138
    sample_r = radius + (2.5 * lam)

    t_finite, div_finite, _ = build_kh_outer_finite_bvp_profile(
        theta,
        radius=radius,
        lambda_value=lam,
        outer_r_max=radius + (20.0 * lam),
    )
    amp_inf = float(theta / special.kv(1, radius / lam))
    t_inf = float(amp_inf * special.kv(1, sample_r / lam))
    div_inf = float(-(amp_inf / lam) * special.kv(0, sample_r / lam))
    assert float(t_finite(sample_r)) == pytest.approx(t_inf, rel=1e-6, abs=1e-10)
    assert float(div_finite(sample_r)) == pytest.approx(div_inf, rel=1e-6, abs=1e-10)


@pytest.mark.unit
def test_kh_outer_finite_bvp_coefficients_are_stable_and_sign_consistent() -> None:
    params = physical_to_dimensionless_theory_params(
        kappa_physical=10.0,
        kappa_t_physical=10.0,
        radius_physical=7.0,
        drive_physical=(2.0 / 0.7),
        length_scale=15.0,
    )
    result = compute_flat_disk_kh_physical_theory(params)
    radius = float(result.radius)
    lam = float(result.lambda_value)
    theta = 0.138

    coeff_i1, coeff_k1 = solve_kh_outer_finite_bvp_coefficients(
        theta,
        radius=radius,
        lambda_value=lam,
        outer_r_max=radius + (10.0 * lam),
    )
    assert np.isfinite(coeff_i1)
    assert np.isfinite(coeff_k1)
    assert float(coeff_k1) > 0.0


@pytest.mark.unit
def test_vector_smoothness_profile_uses_n1_harmonic_reference() -> None:
    radius = 7.0 / 15.0
    theta_boundary = 6.366e-4
    radii = np.asarray([0.0, radius / 2.0, radius, 1.0], dtype=float)

    profile = evaluate_flat_disk_vector_smoothness_profile(
        radii,
        theta_boundary=theta_boundary,
        radius=radius,
    )

    assert profile[0] == pytest.approx(0.0, abs=1e-18)
    assert profile[1] == pytest.approx(0.5 * theta_boundary, rel=1e-12, abs=1e-18)
    assert profile[2] == pytest.approx(theta_boundary, rel=1e-12, abs=1e-18)
    assert profile[3] == pytest.approx(theta_boundary * radius, rel=1e-12, abs=1e-18)


@pytest.mark.unit
def test_vector_kh_profile_uses_i1_k1_amplitude_law() -> None:
    params = physical_to_dimensionless_theory_params(
        kappa_physical=10.0,
        kappa_t_physical=10.0,
        radius_physical=7.0,
        drive_physical=(2.0 / 0.7),
        length_scale=15.0,
    )
    theory = compute_flat_disk_kh_physical_theory(params)
    radius = float(theory.radius)
    lam = float(theory.lambda_value)
    theta_boundary = float(theory.theta_star)
    radii = np.asarray([radius / 2.0, radius, radius + (2.0 * lam)], dtype=float)

    profile = evaluate_flat_disk_vector_kh_profile(
        radii,
        theta_boundary=theta_boundary,
        radius=radius,
        lambda_value=lam,
    )

    assert profile[0] == pytest.approx(
        theta_boundary * special.iv(1, radii[0] / lam) / special.iv(1, radius / lam),
        rel=1e-12,
        abs=1e-18,
    )
    assert profile[1] == pytest.approx(theta_boundary, rel=1e-12, abs=1e-18)
    assert profile[2] == pytest.approx(
        theta_boundary * special.kv(1, radii[2] / lam) / special.kv(1, radius / lam),
        rel=1e-12,
        abs=1e-18,
    )


@pytest.mark.unit
def test_scalar_tex_profile_uses_i0_k0_amplitude_law() -> None:
    params = tex_reference_params()
    theory = compute_flat_disk_theory(params)
    radius = float(theory.radius)
    lam = float(theory.lambda_value)
    theta_boundary = float(theory.theta_star)
    radii = np.asarray([radius / 2.0, radius, 1.0], dtype=float)

    profile = evaluate_flat_disk_scalar_tex_profile(
        radii,
        theta_boundary=theta_boundary,
        radius=radius,
        lambda_value=lam,
    )

    assert profile[0] == pytest.approx(
        theta_boundary * special.iv(0, radii[0] / lam) / special.iv(0, radius / lam),
        rel=1e-12,
        abs=1e-18,
    )
    assert profile[1] == pytest.approx(theta_boundary, rel=1e-12, abs=1e-18)
    assert profile[2] == pytest.approx(
        theta_boundary * special.kv(0, radii[2] / lam) / special.kv(0, radius / lam),
        rel=1e-12,
        abs=1e-18,
    )
