"""Theory helpers for the curved one-leaflet disk benchmark (1_disk_3d.tex).

This module implements the closed-form theory for a curved disk embedded in a
tensionless or finite-tension membrane, as derived in docs/1_disk_3d.tex.

Unlike the flat-disk theory, this model assumes the midplane shape relaxs to
minimize the coupled bending-tilt energy, leading to a "trumpet" shape at
zero tension.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import special


@dataclass(frozen=True)
class CurvedDiskTheoryParams:
    """Input parameters for the curved one-leaflet benchmark theory."""

    kappa: float
    kappa_t: float
    radius: float
    drive: float
    surface_tension: float = 0.0


@dataclass(frozen=True)
class CurvedDiskTheoryResult:
    """Theory outputs for the curved one-leaflet benchmark."""

    params: CurvedDiskTheoryParams
    lambda_value: float
    psi: float
    mu: float
    theta_star: float
    phi_star: float
    elastic_inner: float
    elastic_outer: float
    contact: float
    total: float

    # Coefficients for finite-domain BVP if needed
    coeff_A_eff: float
    coeff_B_eff: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_curved_disk_theory(
    params: CurvedDiskTheoryParams, r_max: float | None = None
) -> CurvedDiskTheoryResult:
    """Compute theory values based on docs/1_disk_3d.tex.

    If r_max is provided, finite-domain BVP corrections (matching t(r_max)=0, z(r_max)=0)
    should be applied. For now, we implement the infinite-domain limits as a baseline.
    """
    kappa = float(params.kappa)
    kappa_t = float(params.kappa_t)
    R = float(params.radius)
    drive = float(params.drive)
    gamma = float(params.surface_tension)

    lam = np.sqrt(kappa_t / kappa)  # Note: docs use lambda = sqrt(kappa_t/kappa)
    # The Bessel arguments in docs are lambda*r, so lambda has units 1/length.
    # docs say lambda = sqrt(kappa_t / kappa).
    # Let's re-verify: F_out = pi int r dr [ kappa (J-Dtheta)^2 + kappa (J+Dtheta)^2 + kappa_t (theta^2 + theta^2) ]
    # At gamma=0, theta_p = theta_d = theta.
    # J = 0 (shape) and theta satisfies r^2 theta'' + r theta' - (1 + lambda^2 r^2) theta = 0.
    # This implies lambda^2 = kappa_t / kappa.

    if gamma == 0:
        # Tensionless case
        mu = 1.0
        psi = 0.0

        # Bessel ratios for r=R
        x = lam * R
        i0_i1 = special.iv(0, x) / special.iv(1, x)
        k0_k1 = special.kv(0, x) / special.kv(1, x)

        # A_eff is the coefficient of theta_B^2 in F_tot
        # F_in,el = pi * kappa * R * lambda * (I0/I1) * theta_B^2
        # F_out,el = (pi/2) * kappa * R * lambda * (K0/K1) * theta_B^2
        # Note: docs/1_disk_3d.tex Eq 24, 25.

        coeff_A = np.pi * kappa * R * lam * (i0_i1 + 0.5 * k0_k1)
        coeff_B = 2 * np.pi * R * drive

        theta_star = coeff_B / (2 * coeff_A)
        phi_star = theta_star / 2.0

        elastic_inner = np.pi * kappa * R * lam * i0_i1 * theta_star**2
        elastic_outer = 0.5 * np.pi * kappa * R * lam * k0_k1 * theta_star**2
        contact = -coeff_B * theta_star
        total = elastic_inner + elastic_outer + contact

        return CurvedDiskTheoryResult(
            params=params,
            lambda_value=lam,
            psi=psi,
            mu=mu,
            theta_star=theta_star,
            phi_star=phi_star,
            elastic_inner=elastic_inner,
            elastic_outer=elastic_outer,
            contact=contact,
            total=total,
            coeff_A_eff=coeff_A,
            coeff_B_eff=coeff_B,
        )
    else:
        # Finite tension case
        # psi^2 = (gamma/kappa) * (kappa_t / (2*kappa_t + gamma))
        psi = np.sqrt((gamma / kappa) * (kappa_t / (2 * kappa_t + gamma)))
        mu = 1.0 - (gamma / (2 * kappa_t))

        x_lam = lam * R
        x_psi = psi * R

        i0_i1_lam = special.iv(0, x_lam) / special.iv(1, x_lam)
        k0_k1_lam = special.kv(0, x_lam) / special.kv(1, x_lam)
        k0_k1_psi = special.kv(0, x_psi) / special.kv(1, x_psi)

        # K_eff from Eq 41
        # K_eff = pi*kappa*R*lambda*(I0/I1) + 0.5*pi*kappa*R*lambda*(K0/K1) + (1/(4*mu^2))*pi*kappa*R*psi*(K0/K1_psi)

        term_in = np.pi * kappa * R * lam * i0_i1_lam
        term_out_tilt = 0.5 * np.pi * kappa * R * lam * k0_k1_lam
        term_out_shape = (1.0 / (4.0 * mu**2)) * np.pi * kappa * R * psi * k0_k1_psi

        coeff_A = term_in + term_out_tilt + term_out_shape
        coeff_B = 2 * np.pi * R * drive

        theta_star = coeff_B / (2 * coeff_A)
        phi_star = theta_star / (2 * mu)

        elastic_inner = term_in * theta_star**2
        elastic_outer = (term_out_tilt + term_out_shape) * theta_star**2
        contact = -coeff_B * theta_star
        total = elastic_inner + elastic_outer + contact

        return CurvedDiskTheoryResult(
            params=params,
            lambda_value=lam,
            psi=psi,
            mu=mu,
            theta_star=theta_star,
            phi_star=phi_star,
            elastic_inner=elastic_inner,
            elastic_outer=elastic_outer,
            contact=contact,
            total=total,
            coeff_A_eff=coeff_A,
            coeff_B_eff=coeff_B,
        )


def tex_reference_params() -> CurvedDiskTheoryParams:
    """Return the benchmark parameters from 1_disk_3d.tex Section 2.1."""
    return CurvedDiskTheoryParams(
        kappa=1.0, kappa_t=225.0, radius=7.0 / 15.0, drive=4.286, surface_tension=0.0
    )


if __name__ == "__main__":
    res = compute_curved_disk_theory(tex_reference_params())
    print("Theory Results (Tensionless):")
    print(f"  theta_star: {res.theta_star:.6f}")
    print(f"  phi_star:   {res.phi_star:.6f}")
    print(f"  Total Energy: {res.total:.6f}")
    print(
        f"  Inner: {res.elastic_inner:.6f}, Outer: {res.elastic_outer:.6f}, Contact: {res.contact:.6f}"
    )
