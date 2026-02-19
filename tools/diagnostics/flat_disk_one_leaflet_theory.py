"""Theory helpers for the flat one-leaflet disk benchmark.

Reference:
    docs/tex/1_disk_flat.tex
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy import special


@dataclass(frozen=True)
class FlatDiskTheoryParams:
    """Input parameters for the flat one-leaflet benchmark theory."""

    kappa: float
    kappa_t: float
    radius: float
    drive: float


@dataclass(frozen=True)
class FlatDiskTheoryResult:
    """Closed-form theory outputs for the flat one-leaflet benchmark."""

    kappa: float
    kappa_t: float
    radius: float
    drive: float
    lambda_value: float
    lambda_inverse: float
    lambda_radius: float
    ratio_i1_i0: float
    ratio_k1_k0: float
    coeff_A: float
    coeff_B: float
    theta_star: float
    elastic_inner: float
    elastic_outer: float
    contact: float
    total: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON/YAML-serializable dict representation."""
        return dict(asdict(self))


@dataclass(frozen=True)
class QuadraticFitResult:
    """Result for a convex quadratic fit E(theta)=a*theta^2+b*theta+c."""

    coeff_a: float
    coeff_b: float
    coeff_c: float
    theta_star: float
    energy_star: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON/YAML-serializable dict representation."""
        return dict(asdict(self))


def tex_reference_params() -> FlatDiskTheoryParams:
    """Return the parameter set stated in docs/tex/1_disk_flat.tex."""
    return FlatDiskTheoryParams(
        kappa=1.0,
        kappa_t=225.0,
        radius=0.4666666667,
        drive=4.285714286,
    )


def physical_to_dimensionless_theory_params(
    *,
    kappa_physical: float,
    kappa_t_physical: float,
    radius_physical: float,
    drive_physical: float,
    length_scale: float,
) -> FlatDiskTheoryParams:
    """Convert physical parameters to the dimensionless theory parameterization.

    Scaling convention:
      - energy unit E0 = kappa_physical
      - length unit L0 = length_scale
    so:
      - kappa_dimless = 1
      - kappa_t_dimless = (kappa_t_physical * L0^2) / E0
      - radius_dimless = radius_physical / L0
      - drive_dimless = (drive_physical * L0) / E0
    """
    if float(kappa_physical) <= 0.0:
        raise ValueError("kappa_physical must be > 0.")
    if float(kappa_t_physical) <= 0.0:
        raise ValueError("kappa_t_physical must be > 0.")
    if float(radius_physical) <= 0.0:
        raise ValueError("radius_physical must be > 0.")
    if float(length_scale) <= 0.0:
        raise ValueError("length_scale must be > 0.")

    e0 = float(kappa_physical)
    l0 = float(length_scale)
    return FlatDiskTheoryParams(
        kappa=1.0,
        kappa_t=float((float(kappa_t_physical) * l0 * l0) / e0),
        radius=float(float(radius_physical) / l0),
        drive=float((float(drive_physical) * l0) / e0),
    )


def validate_theory_params(params: FlatDiskTheoryParams) -> None:
    """Validate physical parameters for the flat benchmark."""
    if float(params.kappa) <= 0.0:
        raise ValueError("kappa must be > 0 for flat-disk theory.")
    if float(params.kappa_t) <= 0.0:
        raise ValueError("kappa_t must be > 0 for flat-disk theory.")
    if float(params.radius) <= 0.0:
        raise ValueError("radius must be > 0 for flat-disk theory.")


def compute_flat_disk_theory(params: FlatDiskTheoryParams) -> FlatDiskTheoryResult:
    """Compute closed-form flat-disk theory values from docs/tex/1_disk_flat.tex."""
    validate_theory_params(params)

    kappa = float(params.kappa)
    kappa_t = float(params.kappa_t)
    radius = float(params.radius)
    drive = float(params.drive)

    lam = float(np.sqrt(kappa / kappa_t))
    lam_inv = float(1.0 / lam)
    x = float(radius / lam)
    if x <= 0.0:
        raise ValueError("lambda * radius must be positive.")

    i0 = float(special.iv(0, x))
    i1 = float(special.iv(1, x))
    k0 = float(special.kv(0, x))
    k1 = float(special.kv(1, x))
    if abs(i0) < 1e-18 or abs(k0) < 1e-18:
        raise ValueError("Invalid Bessel ratio denominator in theory evaluation.")

    ratio_i1_i0 = float(i1 / i0)
    ratio_k1_k0 = float(k1 / k0)

    coeff_A = float(np.pi * kappa_t * radius * lam_inv * (ratio_i1_i0 + ratio_k1_k0))
    coeff_B = float(2.0 * np.pi * radius * drive)
    if coeff_A <= 0.0:
        raise ValueError("Computed quadratic coefficient A must be positive.")

    theta_star = float(coeff_B / (2.0 * coeff_A))
    elastic_inner = float(
        np.pi * kappa_t * radius * lam_inv * theta_star**2 * ratio_i1_i0
    )
    elastic_outer = float(
        np.pi * kappa_t * radius * lam_inv * theta_star**2 * ratio_k1_k0
    )
    contact = float(-coeff_B * theta_star)
    total = float(elastic_inner + elastic_outer + contact)

    return FlatDiskTheoryResult(
        kappa=kappa,
        kappa_t=kappa_t,
        radius=radius,
        drive=drive,
        lambda_value=lam,
        lambda_inverse=lam_inv,
        lambda_radius=x,
        ratio_i1_i0=ratio_i1_i0,
        ratio_k1_k0=ratio_k1_k0,
        coeff_A=coeff_A,
        coeff_B=coeff_B,
        theta_star=theta_star,
        elastic_inner=elastic_inner,
        elastic_outer=elastic_outer,
        contact=contact,
        total=total,
    )


def compute_flat_disk_kh_physical_theory(
    params: FlatDiskTheoryParams,
) -> FlatDiskTheoryResult:
    """Compute strict KH flat-disk closed-form values.

    KH reference model (flat geometry, one leaflet, no curvature relaxation):
      f = 0.5 * kappa * (div t)^2 + 0.5 * kappa_t * |t|^2

    For the axially symmetric disk/outer-domain benchmark:
      E(theta_B) = A*theta_B^2 - B*theta_B
      A = pi * kappa * R * lambda^{-1} * (I1/I0 + K1/K0)
      B = 2*pi*R*drive
    """
    validate_theory_params(params)

    kappa = float(params.kappa)
    kappa_t = float(params.kappa_t)
    radius = float(params.radius)
    drive = float(params.drive)

    lam = float(np.sqrt(kappa / kappa_t))
    lam_inv = float(1.0 / lam)
    x = float(radius / lam)
    if x <= 0.0:
        raise ValueError("lambda * radius must be positive.")

    i0 = float(special.iv(0, x))
    i1 = float(special.iv(1, x))
    k0 = float(special.kv(0, x))
    k1 = float(special.kv(1, x))
    if abs(i0) < 1e-18 or abs(k0) < 1e-18:
        raise ValueError("Invalid Bessel ratio denominator in KH theory evaluation.")

    ratio_i1_i0 = float(i1 / i0)
    ratio_k1_k0 = float(k1 / k0)

    coeff_A = float(np.pi * kappa * radius * lam_inv * (ratio_i1_i0 + ratio_k1_k0))
    coeff_B = float(2.0 * np.pi * radius * drive)
    if coeff_A <= 0.0:
        raise ValueError("Computed quadratic coefficient A must be positive.")

    theta_star = float(coeff_B / (2.0 * coeff_A))
    elastic_inner = float(
        np.pi * kappa * radius * lam_inv * theta_star**2 * ratio_i1_i0
    )
    elastic_outer = float(
        np.pi * kappa * radius * lam_inv * theta_star**2 * ratio_k1_k0
    )
    contact = float(-coeff_B * theta_star)
    total = float(elastic_inner + elastic_outer + contact)

    return FlatDiskTheoryResult(
        kappa=kappa,
        kappa_t=kappa_t,
        radius=radius,
        drive=drive,
        lambda_value=lam,
        lambda_inverse=lam_inv,
        lambda_radius=x,
        ratio_i1_i0=ratio_i1_i0,
        ratio_k1_k0=ratio_k1_k0,
        coeff_A=coeff_A,
        coeff_B=coeff_B,
        theta_star=theta_star,
        elastic_inner=elastic_inner,
        elastic_outer=elastic_outer,
        contact=contact,
        total=total,
    )


def solver_mapping_from_theory(
    params: FlatDiskTheoryParams, *, parameterization: str = "legacy"
) -> dict[str, float]:
    """Map theory coefficients to solver coefficients.

    Supported parameterizations:
      - legacy:
          bending_modulus_in = kappa_t
          tilt_modulus_in = kappa_t^2 / kappa
      - kh_physical:
          bending_modulus_in = kappa
          tilt_modulus_in = kappa_t
    """
    validate_theory_params(params)
    kappa = float(params.kappa)
    kappa_t = float(params.kappa_t)
    mode = str(parameterization).lower()
    if mode == "legacy":
        return {
            "bending_modulus_in": float(kappa_t),
            "tilt_modulus_in": float((kappa_t * kappa_t) / kappa),
        }
    if mode == "kh_physical":
        return {
            "bending_modulus_in": float(kappa),
            "tilt_modulus_in": float(kappa_t),
        }
    raise ValueError("parameterization must be 'legacy' or 'kh_physical'.")


def quadratic_min_from_scan(
    theta_values: Sequence[float],
    energy_values: Sequence[float],
) -> QuadraticFitResult:
    """Fit a convex quadratic and return its minimizer."""
    theta = np.asarray(theta_values, dtype=float)
    energy = np.asarray(energy_values, dtype=float)
    if theta.ndim != 1 or energy.ndim != 1 or theta.size != energy.size:
        raise ValueError(
            "theta_values and energy_values must be 1D arrays of equal size."
        )
    if theta.size < 3:
        raise ValueError("Need at least 3 scan points for quadratic fit.")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(energy)):
        raise ValueError("Scan values must be finite.")

    coeff_a, coeff_b, coeff_c = np.polyfit(theta, energy, 2)
    coeff_a = float(coeff_a)
    coeff_b = float(coeff_b)
    coeff_c = float(coeff_c)
    if coeff_a <= 0.0:
        raise ValueError("Quadratic fit is not convex; scan bracket is not suitable.")

    theta_star = float(-coeff_b / (2.0 * coeff_a))
    if theta_star < float(np.min(theta)) or theta_star > float(np.max(theta)):
        raise ValueError(
            "Quadratic minimum is outside scan bracket; widen theta range."
        )
    energy_star = float(
        (coeff_a * theta_star * theta_star) + (coeff_b * theta_star) + coeff_c
    )

    return QuadraticFitResult(
        coeff_a=coeff_a,
        coeff_b=coeff_b,
        coeff_c=coeff_c,
        theta_star=theta_star,
        energy_star=energy_star,
    )


__all__ = [
    "FlatDiskTheoryParams",
    "FlatDiskTheoryResult",
    "QuadraticFitResult",
    "compute_flat_disk_theory",
    "compute_flat_disk_kh_physical_theory",
    "physical_to_dimensionless_theory_params",
    "quadratic_min_from_scan",
    "solver_mapping_from_theory",
    "tex_reference_params",
    "validate_theory_params",
]
