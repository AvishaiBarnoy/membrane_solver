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


def evaluate_tensionless_curved_disk_profiles(
    *,
    result: CurvedDiskTheoryResult,
    radii: np.ndarray,
    z_rim: float = 0.0,
) -> dict[str, np.ndarray]:
    """Evaluate the tensionless one-disc minimizing fields at radial samples."""
    if float(result.params.surface_tension) != 0.0:
        raise ValueError("This profile evaluator currently requires zero tension.")

    r = np.asarray(radii, dtype=float)
    if np.any(r < 0.0):
        raise ValueError("radii must be non-negative")
    radius = float(result.params.radius)
    lam = float(result.lambda_value)
    theta = float(result.theta_star)
    phi = float(result.phi_star)
    x_rim = lam * radius
    i1_rim = float(special.iv(1, x_rim))
    k1_rim = float(special.kv(1, x_rim))
    if abs(i1_rim) < 1.0e-18 or abs(k1_rim) < 1.0e-18:
        raise ValueError("Invalid Bessel normalization at the disc rim.")

    disk = r <= radius
    outer = r >= radius
    height = np.full_like(r, float(z_rim))
    slope = np.zeros_like(r)
    tilt_disk = np.zeros_like(r)
    tilt_outer = np.zeros_like(r)

    scaled = lam * r
    tilt_disk[disk] = (
        theta * np.asarray(special.iv(1, scaled[disk]), dtype=float) / i1_rim
    )
    if np.any(outer):
        outer_amplitude = (
            phi * np.asarray(special.kv(1, scaled[outer]), dtype=float) / k1_rim
        )
        tilt_outer[outer] = outer_amplitude
        slope[outer] = phi * radius / r[outer]
        height[outer] = float(z_rim) + phi * radius * np.log(r[outer] / radius)

    tilt_in = np.where(disk, tilt_disk, tilt_outer)
    return {
        "height": height,
        "slope": slope,
        "tilt_disk": tilt_disk,
        "tilt_outer": tilt_outer,
        "tilt_in": tilt_in,
        "tilt_out": tilt_outer.copy(),
    }


def _weighted_relative_l2(
    measured: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return a weighted relative L2 error with a stable zero-field fallback."""
    delta_norm = float(np.sqrt(np.sum(weights * (measured - reference) ** 2)))
    reference_norm = float(np.sqrt(np.sum(weights * reference**2)))
    if reference_norm <= 1.0e-15:
        return delta_norm
    return delta_norm / reference_norm


def compare_tensionless_curved_disk_profiles(
    *,
    result: CurvedDiskTheoryResult,
    radii: np.ndarray,
    height: np.ndarray,
    slope: np.ndarray,
    tilt_in_radial: np.ndarray,
    tilt_out_radial: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compare sampled numerical fields with the full tensionless solution.

    The absolute height is a gauge degree of freedom.  Its best weighted
    constant offset is therefore removed before evaluating the height error.
    """
    r = np.asarray(radii, dtype=float)
    measured = {
        "height": np.asarray(height, dtype=float),
        "slope": np.asarray(slope, dtype=float),
        "tilt_in": np.asarray(tilt_in_radial, dtype=float),
        "tilt_out": np.asarray(tilt_out_radial, dtype=float),
    }
    if any(values.shape != r.shape for values in measured.values()):
        raise ValueError("all sampled fields must have the same shape as radii")

    if weights is None:
        sample_weights = np.ones_like(r)
    else:
        sample_weights = np.asarray(weights, dtype=float)
        if sample_weights.shape != r.shape:
            raise ValueError("weights must have the same shape as radii")
        if np.any(sample_weights < 0.0):
            raise ValueError("weights must be non-negative")
    weight_sum = float(np.sum(sample_weights))
    if weight_sum <= 0.0:
        raise ValueError("weights must have a positive sum")

    reference = evaluate_tensionless_curved_disk_profiles(result=result, radii=r)
    height_offset = float(
        np.sum(sample_weights * (measured["height"] - reference["height"])) / weight_sum
    )
    height_aligned = measured["height"] - height_offset

    return {
        "height_gauge_offset": height_offset,
        "height_rel_l2": _weighted_relative_l2(
            height_aligned, reference["height"], sample_weights
        ),
        "slope_rel_l2": _weighted_relative_l2(
            measured["slope"], reference["slope"], sample_weights
        ),
        "tilt_in_rel_l2": _weighted_relative_l2(
            measured["tilt_in"], reference["tilt_in"], sample_weights
        ),
        "tilt_out_rel_l2": _weighted_relative_l2(
            measured["tilt_out"], reference["tilt_out"], sample_weights
        ),
    }


def axisymmetric_ring_topology_diagnostics(mesh) -> dict[str, object]:
    """Report radial backtracking in an axisymmetric ring mesh.

    Vertices are grouped by cylindrical radius and the resulting ring graph is
    traversed outward from the center.  A decreasing radius along that
    topological path identifies a folded annular band.
    """
    mesh.build_position_cache()
    positions = np.asarray(mesh.positions_view(), dtype=float)
    radii = np.linalg.norm(positions[:, :2], axis=1)
    rounded = np.round(radii, 9)
    unique_radii, ring_index = np.unique(rounded, return_inverse=True)
    index_map = mesh.vertex_index_to_row

    adjacency: list[set[int]] = [set() for _ in range(unique_radii.size)]
    for edge in mesh.edges.values():
        tail_row = index_map.get(int(edge.tail_index))
        head_row = index_map.get(int(edge.head_index))
        if tail_row is None or head_row is None:
            continue
        tail_ring = int(ring_index[int(tail_row)])
        head_ring = int(ring_index[int(head_row)])
        if tail_ring == head_ring:
            continue
        adjacency[tail_ring].add(head_ring)
        adjacency[head_ring].add(tail_ring)

    start = int(np.argmin(unique_radii))
    path = [start]
    previous: int | None = None
    current = start
    while True:
        candidates = sorted(
            adjacency[current] - ({previous} if previous is not None else set())
        )
        if len(candidates) != 1:
            break
        next_ring = int(candidates[0])
        if next_ring in path:
            break
        path.append(next_ring)
        previous, current = current, next_ring

    path_radii = np.asarray([unique_radii[idx] for idx in path], dtype=float)
    inversions = [
        {
            "inner_radius": float(path_radii[idx]),
            "outer_radius": float(path_radii[idx + 1]),
        }
        for idx in range(max(0, path_radii.size - 1))
        if path_radii[idx + 1] <= path_radii[idx]
    ]
    return {
        "ring_count": int(unique_radii.size),
        "path_ring_count": int(path_radii.size),
        "path_radii": path_radii.tolist(),
        "inversion_count": int(len(inversions)),
        "inversions": inversions,
        "is_monotone": bool(path_radii.size == unique_radii.size and not inversions),
    }


if __name__ == "__main__":
    res = compute_curved_disk_theory(tex_reference_params())
    print("Theory Results (Tensionless):")
    print(f"  theta_star: {res.theta_star:.6f}")
    print(f"  phi_star:   {res.phi_star:.6f}")
    print(f"  Total Energy: {res.total:.6f}")
    print(
        f"  Inner: {res.elastic_inner:.6f}, Outer: {res.elastic_outer:.6f}, Contact: {res.contact:.6f}"
    )
