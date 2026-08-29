"""Numerical region, band, and resolution metrics for the flat-disk KH audit."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from scipy import integrate, special

from tools.flat_disk_benchmark_metrics import (
    _rim_boundary_realization_metrics as _boundary_realization_metrics,  # noqa: F401
)


def _radial_frames(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return radius, radial unit vectors, and azimuthal unit vectors."""
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    phi_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    phi_hat[good, 0] = -positions[good, 1] / r[good]
    phi_hat[good, 1] = positions[good, 0] / r[good]
    return r, r_hat, phi_hat


def _triangle_centroid_radius(
    positions: np.ndarray,
    tri_rows: np.ndarray,
) -> np.ndarray:
    tri_cent = (
        positions[tri_rows[:, 0]]
        + positions[tri_rows[:, 1]]
        + positions[tri_rows[:, 2]]
    ) / 3.0
    return np.linalg.norm(tri_cent[:, :2], axis=1)


def _triangle_inside_fraction(
    positions: np.ndarray,
    tri_rows: np.ndarray,
    *,
    radius: float,
    subdivisions: int = 6,
) -> np.ndarray:
    """Return per-triangle inside fraction using deterministic subtriangle sampling."""
    tri_pos = positions[tri_rows]
    tri_r = np.linalg.norm(tri_pos[:, :, :2], axis=2)
    inside = tri_r <= float(radius)
    all_in = np.all(inside, axis=1)
    all_out = np.all(~inside, axis=1)

    frac = np.zeros(tri_rows.shape[0], dtype=float)
    frac[all_in] = 1.0
    boundary = ~(all_in | all_out)
    if not np.any(boundary):
        return frac

    n = max(int(subdivisions), 1)
    bary: list[tuple[float, float, float]] = []
    inv_n = 1.0 / float(n)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            bary.append((i * inv_n, j * inv_n, k * inv_n))
    w = np.asarray(bary, dtype=float)  # (P, 3)

    tri2 = tri_pos[boundary, :, :2]  # (M, 3, 2)
    pts = np.einsum("pj,mjd->mpd", w, tri2)
    rr = np.linalg.norm(pts, axis=2)
    frac[boundary] = np.mean(rr <= float(radius), axis=1)
    return frac


def _triangle_radial_interval_fraction(
    positions: np.ndarray,
    tri_rows: np.ndarray,
    *,
    r_min: float,
    r_max: float | None,
    subdivisions: int = 6,
) -> np.ndarray:
    """Return per-triangle fraction inside radial interval [r_min, r_max)."""
    if tri_rows is None or len(tri_rows) == 0:
        return np.zeros(0, dtype=float)
    tri_pos = positions[tri_rows]
    tri_r = np.linalg.norm(tri_pos[:, :, :2], axis=2)
    lo = float(max(r_min, 0.0))
    hi = None if r_max is None else float(max(r_max, lo))
    if hi is None:
        inside_v = tri_r >= lo
    else:
        inside_v = (tri_r >= lo) & (tri_r < hi)
    all_in = np.all(inside_v, axis=1)
    all_out = np.all(~inside_v, axis=1)

    frac = np.zeros(tri_rows.shape[0], dtype=float)
    frac[all_in] = 1.0
    boundary = ~(all_in | all_out)
    if not np.any(boundary):
        return frac

    n = max(int(subdivisions), 1)
    bary: list[tuple[float, float, float]] = []
    inv_n = 1.0 / float(n)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            bary.append((i * inv_n, j * inv_n, k * inv_n))
    w = np.asarray(bary, dtype=float)
    tri2 = tri_pos[boundary, :, :2]
    pts = np.einsum("pj,mjd->mpd", w, tri2)
    rr = np.linalg.norm(pts, axis=2)
    if hi is None:
        inside_pts = rr >= lo
    else:
        inside_pts = (rr >= lo) & (rr < hi)
    frac[boundary] = np.mean(inside_pts, axis=1)
    return frac


def _mesh_internal_triangle_terms(
    mesh,
    *,
    smoothness_model: str,
) -> dict[str, Any]:
    """Return shared per-triangle internal energy terms for region/band diagnostics."""
    from modules.energy import tilt_smoothness as tilt_smoothness_base

    gp = mesh.global_parameters
    positions = mesh.positions_view()
    tilts_in = mesh.tilts_in_view()
    area, g0, g1, g2, tri_rows = mesh.p1_triangle_shape_gradient_cache(
        positions=positions
    )
    if tri_rows is None or len(tri_rows) == 0:
        return {
            "positions": positions,
            "tri_rows": np.zeros((0, 3), dtype=int),
            "tilt_tri": np.zeros(0, dtype=float),
            "smooth_tri": np.zeros(0, dtype=float),
            "internal_tri": np.zeros(0, dtype=float),
        }

    tri_rows_arr = np.asarray(tri_rows, dtype=int)
    k_tilt = float(gp.get("tilt_modulus_in") or 0.0)
    tilt_sq = np.einsum("ij,ij->i", tilts_in, tilts_in)
    tri_tilt_sq_sum = tilt_sq[tri_rows_arr].sum(axis=1)
    tilt_tri = 0.5 * k_tilt * area * (tri_tilt_sq_sum / 3.0)

    mode = str(smoothness_model)
    if mode == "splay_twist":
        k_splay = gp.get("tilt_splay_modulus_in")
        if k_splay is None:
            k_splay = gp.get("bending_modulus_in")
        if k_splay is None:
            k_splay = gp.get("bending_modulus")
        k_splay_f = float(k_splay or 0.0)
        k_twist_f = float(gp.get("tilt_twist_modulus_in") or 0.0)
        t0 = tilts_in[tri_rows_arr[:, 0]]
        t1 = tilts_in[tri_rows_arr[:, 1]]
        t2 = tilts_in[tri_rows_arr[:, 2]]
        div_tri = (
            np.einsum("ij,ij->i", t0, g0)
            + np.einsum("ij,ij->i", t1, g1)
            + np.einsum("ij,ij->i", t2, g2)
        )
        n = mesh.triangle_normals(positions=positions)
        n_norm = np.linalg.norm(n, axis=1)
        n_hat = np.zeros_like(n)
        good = n_norm > 1e-20
        n_hat[good] = n[good] / n_norm[good, None]
        curl_vec = np.cross(g0, t0) + np.cross(g1, t1) + np.cross(g2, t2)
        curl_n = np.einsum("ij,ij->i", curl_vec, n_hat)
        smooth_tri = (
            0.5
            * area
            * ((k_splay_f * div_tri * div_tri) + (k_twist_f * curl_n * curl_n))
        )
    elif mode == "dirichlet":
        k_smooth = gp.get("bending_modulus_in")
        if k_smooth is None:
            k_smooth = gp.get("bending_modulus")
        k_smooth_f = float(k_smooth or 0.0)
        weights, smooth_tri_rows = tilt_smoothness_base._get_weights_and_tris(
            mesh,
            positions=positions,
            index_map=mesh.vertex_index_to_row,
        )
        if smooth_tri_rows is None:
            smooth_tri = np.zeros_like(tilt_tri)
        else:
            rows = np.asarray(smooth_tri_rows, dtype=int)
            c0 = weights[:, 0]
            c1 = weights[:, 1]
            c2 = weights[:, 2]
            t0 = tilts_in[rows[:, 0]]
            t1 = tilts_in[rows[:, 1]]
            t2 = tilts_in[rows[:, 2]]
            d12 = t1 - t2
            d20 = t2 - t0
            d01 = t0 - t1
            smooth_raw = (
                0.25
                * k_smooth_f
                * (
                    c0 * np.einsum("ij,ij->i", d12, d12)
                    + c1 * np.einsum("ij,ij->i", d20, d20)
                    + c2 * np.einsum("ij,ij->i", d01, d01)
                )
            )
            if rows.shape == tri_rows_arr.shape and np.array_equal(rows, tri_rows_arr):
                smooth_tri = smooth_raw
            else:
                smooth_tri = np.zeros(tri_rows_arr.shape[0], dtype=float)
                tri_lookup = {
                    tuple(sorted(int(v) for v in tri.tolist())): idx
                    for idx, tri in enumerate(tri_rows_arr)
                }
                for row_vals, value in zip(rows, smooth_raw):
                    idx = tri_lookup.get(
                        tuple(sorted(int(v) for v in row_vals.tolist()))
                    )
                    if idx is not None:
                        smooth_tri[idx] += float(value)
    else:
        raise ValueError("smoothness_model must be 'dirichlet' or 'splay_twist'.")

    return {
        "positions": positions,
        "tri_rows": tri_rows_arr,
        "tilt_tri": np.asarray(tilt_tri, dtype=float),
        "smooth_tri": np.asarray(smooth_tri, dtype=float),
        "internal_tri": np.asarray(tilt_tri + smooth_tri, dtype=float),
    }


def _mesh_internal_region_split(
    mesh,
    *,
    smoothness_model: str,
    radius: float,
    triangle_terms: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Split mesh internal energy into disk (r<R) and outer (r>R) regions."""
    terms = (
        _mesh_internal_triangle_terms(mesh, smoothness_model=smoothness_model)
        if triangle_terms is None
        else triangle_terms
    )
    tri_rows = np.asarray(terms["tri_rows"], dtype=int)
    if tri_rows.size == 0:
        return {
            "mesh_internal_disk": 0.0,
            "mesh_internal_outer": 0.0,
            "mesh_internal_total_from_regions": 0.0,
            "mesh_tilt_disk": 0.0,
            "mesh_tilt_outer": 0.0,
            "mesh_smooth_disk": 0.0,
            "mesh_smooth_outer": 0.0,
        }

    positions = np.asarray(terms["positions"], dtype=float)
    tilt_tri = np.asarray(terms["tilt_tri"], dtype=float)
    smooth_tri = np.asarray(terms["smooth_tri"], dtype=float)
    disk_frac = _triangle_inside_fraction(positions, tri_rows, radius=float(radius))
    outer_frac = 1.0 - disk_frac

    tilt_disk = float(np.sum(tilt_tri * disk_frac))
    tilt_outer = float(np.sum(tilt_tri * outer_frac))
    smooth_disk = float(np.sum(smooth_tri * disk_frac))
    smooth_outer = float(np.sum(smooth_tri * outer_frac))
    return {
        "mesh_internal_disk": float(tilt_disk + smooth_disk),
        "mesh_internal_outer": float(tilt_outer + smooth_outer),
        "mesh_internal_total_from_regions": float(
            tilt_disk + smooth_disk + tilt_outer + smooth_outer
        ),
        "mesh_tilt_disk": tilt_disk,
        "mesh_tilt_outer": tilt_outer,
        "mesh_smooth_disk": smooth_disk,
        "mesh_smooth_outer": smooth_outer,
    }


def _mesh_internal_band_split(
    mesh,
    *,
    smoothness_model: str,
    radius: float,
    lambda_value: float,
    rim_half_width_lambda: float = 1.0,
    outer_near_width_lambda: float = 4.0,
    partition_mode: str = "centroid",
    triangle_terms: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Split mesh internal energy into radial bands and report rim-band resolution."""
    terms = (
        _mesh_internal_triangle_terms(mesh, smoothness_model=smoothness_model)
        if triangle_terms is None
        else triangle_terms
    )
    positions = np.asarray(terms["positions"], dtype=float)
    tri_rows = np.asarray(terms["tri_rows"], dtype=int)
    if tri_rows.size == 0:
        return {
            "mesh_internal_disk_core": 0.0,
            "mesh_internal_rim_band": 0.0,
            "mesh_internal_outer_near": 0.0,
            "mesh_internal_outer_far": 0.0,
            "rim_band_tri_count": 0.0,
            "rim_band_h_over_lambda_median": float("nan"),
        }
    tilt_tri = np.asarray(terms["tilt_tri"], dtype=float)
    smooth_tri = np.asarray(terms["smooth_tri"], dtype=float)
    internal_tri = np.asarray(terms["internal_tri"], dtype=float)
    rim_w = float(rim_half_width_lambda) * float(lambda_value)
    outer_near_w = float(outer_near_width_lambda) * float(lambda_value)
    partition = str(partition_mode).strip().lower()
    if partition == "centroid":
        tri_r = _triangle_centroid_radius(positions, tri_rows)
        disk_core_w = (tri_r < (float(radius) - rim_w)).astype(float)
        rim_band_w = (np.abs(tri_r - float(radius)) <= rim_w).astype(float)
        outer_near_wg = (
            (tri_r > (float(radius) + rim_w))
            & (tri_r <= (float(radius) + outer_near_w))
        ).astype(float)
        outer_far_w = (tri_r > (float(radius) + outer_near_w)).astype(float)
    elif partition == "fractional":
        disk_core_w = _triangle_radial_interval_fraction(
            positions, tri_rows, r_min=0.0, r_max=(float(radius) - rim_w)
        )
        rim_band_w = _triangle_radial_interval_fraction(
            positions,
            tri_rows,
            r_min=(float(radius) - rim_w),
            r_max=(float(radius) + rim_w),
        )
        outer_near_wg = _triangle_radial_interval_fraction(
            positions,
            tri_rows,
            r_min=(float(radius) + rim_w),
            r_max=(float(radius) + outer_near_w),
        )
        outer_far_w = _triangle_radial_interval_fraction(
            positions,
            tri_rows,
            r_min=(float(radius) + outer_near_w),
            r_max=None,
        )
    else:
        raise ValueError("partition_mode must be 'centroid' or 'fractional'.")

    tri_pos = positions[tri_rows]
    e01 = np.linalg.norm(tri_pos[:, 0] - tri_pos[:, 1], axis=1)
    e12 = np.linalg.norm(tri_pos[:, 1] - tri_pos[:, 2], axis=1)
    e20 = np.linalg.norm(tri_pos[:, 2] - tri_pos[:, 0], axis=1)
    h_tri = np.maximum.reduce([e01, e12, e20])
    rim_h = h_tri[rim_band_w > 1e-12]
    rim_h_over_lambda = (
        float(np.median(rim_h) / max(float(lambda_value), 1e-18))
        if rim_h.size > 0
        else float("nan")
    )

    return {
        "mesh_internal_disk_core": float(np.sum(internal_tri * disk_core_w)),
        "mesh_internal_rim_band": float(np.sum(internal_tri * rim_band_w)),
        "mesh_internal_outer_near": float(np.sum(internal_tri * outer_near_wg)),
        "mesh_internal_outer_far": float(np.sum(internal_tri * outer_far_w)),
        "mesh_tilt_disk_core": float(np.sum(tilt_tri * disk_core_w)),
        "mesh_tilt_rim_band": float(np.sum(tilt_tri * rim_band_w)),
        "mesh_tilt_outer_near": float(np.sum(tilt_tri * outer_near_wg)),
        "mesh_tilt_outer_far": float(np.sum(tilt_tri * outer_far_w)),
        "mesh_smooth_disk_core": float(np.sum(smooth_tri * disk_core_w)),
        "mesh_smooth_rim_band": float(np.sum(smooth_tri * rim_band_w)),
        "mesh_smooth_outer_near": float(np.sum(smooth_tri * outer_near_wg)),
        "mesh_smooth_outer_far": float(np.sum(smooth_tri * outer_far_w)),
        "rim_band_tri_count": float(np.sum(rim_band_w)),
        "rim_band_h_over_lambda_median": rim_h_over_lambda,
    }


def _theory_term_band_split_uncached(
    *,
    theta: float,
    kappa: float,
    kappa_t: float,
    radius: float,
    lambda_value: float,
    rim_half_width_lambda: float,
    outer_near_width_lambda: float,
    outer_r_max: float | None = None,
    theory_outer_mode: str = "infinite",
) -> dict[str, Any]:
    """Compute KH theory tilt/splay term split across radial bands at fixed theta."""
    from tools.diagnostics.flat_disk_one_leaflet_theory import (
        build_kh_outer_finite_bvp_profile,
    )

    theta_f = float(theta)
    kappa_f = float(kappa)
    kappa_t_f = float(kappa_t)
    radius_f = float(radius)
    lam = float(lambda_value)
    outer_mode = str(theory_outer_mode).strip().lower()
    if outer_mode not in {"infinite", "finite_bvp"}:
        raise ValueError("theory_outer_mode must be 'infinite' or 'finite_bvp'.")

    x = radius_f / max(lam, 1e-18)
    i1_x = float(special.iv(1, x))
    k1_x = float(special.kv(1, x))
    if abs(i1_x) < 1e-18 or abs(k1_x) < 1e-18:
        raise ValueError("Invalid KH radial profile normalization in term audit.")

    a_inner = theta_f / i1_x
    b_outer = theta_f / k1_x

    def _t_inner(r: float) -> float:
        return float(a_inner * special.iv(1, r / lam))

    def _div_inner(r: float) -> float:
        return float((a_inner / lam) * special.iv(0, r / lam))

    if outer_mode == "infinite":

        def _t_outer(r: float) -> float:
            return float(b_outer * special.kv(1, r / lam))

        def _div_outer(r: float) -> float:
            return float(-(b_outer / lam) * special.kv(0, r / lam))

        r_max_outer = None if outer_r_max is None else float(outer_r_max)
    else:
        if outer_r_max is None or not np.isfinite(float(outer_r_max)):
            raise ValueError(
                "finite_bvp outer mode requires finite outer_r_max "
                f"(got outer_r_max={outer_r_max})."
            )
        r_max_outer = float(outer_r_max)
        _t_outer, _div_outer, _ = build_kh_outer_finite_bvp_profile(
            theta_f,
            radius=radius_f,
            lambda_value=lam,
            outer_r_max=r_max_outer,
        )

    def _integrate_term(
        fn, lo: float, hi: float, *, use_inf: bool = False, coeff: float
    ) -> float:
        lo_f = max(float(lo), 0.0)
        if use_inf:
            val, _ = integrate.quad(
                lambda rr: np.pi * coeff * rr * (fn(rr) ** 2),
                lo_f,
                np.inf,
                epsabs=1e-10,
                epsrel=1e-9,
                limit=300,
            )
            return float(val)
        hi_f = max(float(hi), lo_f)
        if hi_f <= lo_f:
            return 0.0
        val, _ = integrate.quad(
            lambda rr: np.pi * coeff * rr * (fn(rr) ** 2),
            lo_f,
            hi_f,
            epsabs=1e-10,
            epsrel=1e-9,
            limit=300,
        )
        return float(val)

    rim_w = max(0.0, float(rim_half_width_lambda) * lam)
    outer_near_w = max(0.0, float(outer_near_width_lambda) * lam)
    r_in_rim_start = max(0.0, radius_f - rim_w)
    r_out_rim_end = radius_f + rim_w
    r_outer_near_end = radius_f + outer_near_w

    tilt_disk_core = _integrate_term(
        _t_inner, 0.0, min(radius_f, r_in_rim_start), coeff=kappa_t_f
    )
    smooth_disk_core = _integrate_term(
        _div_inner, 0.0, min(radius_f, r_in_rim_start), coeff=kappa_f
    )

    tilt_rim_in = _integrate_term(_t_inner, r_in_rim_start, radius_f, coeff=kappa_t_f)
    smooth_rim_in = _integrate_term(_div_inner, r_in_rim_start, radius_f, coeff=kappa_f)
    tilt_rim_out = _integrate_term(
        _t_outer, radius_f, max(radius_f, r_out_rim_end), coeff=kappa_t_f
    )
    smooth_rim_out = _integrate_term(
        _div_outer, radius_f, max(radius_f, r_out_rim_end), coeff=kappa_f
    )
    tilt_rim_band = float(tilt_rim_in + tilt_rim_out)
    smooth_rim_band = float(smooth_rim_in + smooth_rim_out)

    tilt_outer_near = _integrate_term(
        _t_outer,
        max(radius_f, r_out_rim_end),
        max(radius_f, r_outer_near_end),
        coeff=kappa_t_f,
    )
    smooth_outer_near = _integrate_term(
        _div_outer,
        max(radius_f, r_out_rim_end),
        max(radius_f, r_outer_near_end),
        coeff=kappa_f,
    )
    r_far_start = max(radius_f, r_outer_near_end)
    if outer_mode == "finite_bvp":
        r_max = max(float(r_max_outer), radius_f)
        near_upper = min(max(radius_f, r_outer_near_end), r_max)
        tilt_outer_near = _integrate_term(
            _t_outer,
            max(radius_f, r_out_rim_end),
            near_upper,
            coeff=kappa_t_f,
        )
        smooth_outer_near = _integrate_term(
            _div_outer,
            max(radius_f, r_out_rim_end),
            near_upper,
            coeff=kappa_f,
        )
        tilt_outer_far = _integrate_term(_t_outer, r_far_start, r_max, coeff=kappa_t_f)
        smooth_outer_far = _integrate_term(
            _div_outer, r_far_start, r_max, coeff=kappa_f
        )
    else:
        r_max = None if outer_r_max is None else max(float(outer_r_max), r_far_start)
    if outer_mode == "infinite" and r_max is None:
        tilt_outer_far = _integrate_term(
            _t_outer, r_far_start, 0.0, use_inf=True, coeff=kappa_t_f
        )
        smooth_outer_far = _integrate_term(
            _div_outer,
            r_far_start,
            0.0,
            use_inf=True,
            coeff=kappa_f,
        )
    else:
        tilt_outer_far = _integrate_term(_t_outer, r_far_start, r_max, coeff=kappa_t_f)
        smooth_outer_far = _integrate_term(
            _div_outer, r_far_start, r_max, coeff=kappa_f
        )

    return {
        "theory_tilt_disk_core": tilt_disk_core,
        "theory_tilt_rim_band": tilt_rim_band,
        "theory_tilt_outer_near": tilt_outer_near,
        "theory_tilt_outer_far": tilt_outer_far,
        "theory_smooth_disk_core": smooth_disk_core,
        "theory_smooth_rim_band": smooth_rim_band,
        "theory_smooth_outer_near": smooth_outer_near,
        "theory_smooth_outer_far": smooth_outer_far,
        "theory_internal_disk_core": float(tilt_disk_core + smooth_disk_core),
        "theory_internal_rim_band": float(tilt_rim_band + smooth_rim_band),
        "theory_internal_outer_near": float(tilt_outer_near + smooth_outer_near),
        "theory_internal_outer_far": float(tilt_outer_far + smooth_outer_far),
        "theory_outer_r_max": float(r_max) if r_max is not None else float("inf"),
        "theory_outer_mode": str(outer_mode),
    }


_THEORY_BAND_SPLIT_FIELD_ORDER: tuple[str, ...] = (
    "theory_tilt_disk_core",
    "theory_tilt_rim_band",
    "theory_tilt_outer_near",
    "theory_tilt_outer_far",
    "theory_smooth_disk_core",
    "theory_smooth_rim_band",
    "theory_smooth_outer_near",
    "theory_smooth_outer_far",
    "theory_internal_disk_core",
    "theory_internal_rim_band",
    "theory_internal_outer_near",
    "theory_internal_outer_far",
    "theory_outer_r_max",
    "theory_outer_mode",
)


@lru_cache(maxsize=2048)
def _theory_term_band_split_cached(
    *,
    theta: float,
    kappa: float,
    kappa_t: float,
    radius: float,
    lambda_value: float,
    rim_half_width_lambda: float,
    outer_near_width_lambda: float,
    outer_r_max: float | None = None,
    theory_outer_mode: str = "infinite",
) -> tuple[object, ...]:
    result = _theory_term_band_split_uncached(
        theta=float(theta),
        kappa=float(kappa),
        kappa_t=float(kappa_t),
        radius=float(radius),
        lambda_value=float(lambda_value),
        rim_half_width_lambda=float(rim_half_width_lambda),
        outer_near_width_lambda=float(outer_near_width_lambda),
        outer_r_max=None if outer_r_max is None else float(outer_r_max),
        theory_outer_mode=str(theory_outer_mode),
    )
    return tuple(result[name] for name in _THEORY_BAND_SPLIT_FIELD_ORDER)


def _theory_term_band_split(
    *,
    theta: float,
    kappa: float,
    kappa_t: float,
    radius: float,
    lambda_value: float,
    rim_half_width_lambda: float,
    outer_near_width_lambda: float,
    outer_r_max: float | None = None,
    theory_outer_mode: str = "infinite",
) -> dict[str, Any]:
    """Compute KH theory tilt/splay term split across radial bands at fixed theta."""
    values = _theory_term_band_split_cached(
        theta=float(theta),
        kappa=float(kappa),
        kappa_t=float(kappa_t),
        radius=float(radius),
        lambda_value=float(lambda_value),
        rim_half_width_lambda=float(rim_half_width_lambda),
        outer_near_width_lambda=float(outer_near_width_lambda),
        outer_r_max=None if outer_r_max is None else float(outer_r_max),
        theory_outer_mode=str(theory_outer_mode),
    )
    return dict(zip(_THEORY_BAND_SPLIT_FIELD_ORDER, values, strict=True))


def _leakage_metrics(
    mesh,
    *,
    radius: float,
    lambda_value: float,
    rim_half_width_lambda: float,
    outer_near_width_lambda: float,
) -> dict[str, float]:
    """Report azimuthal (t_phi) leakage relative to radial component."""
    from tools.flat_disk_benchmark_metrics import _leakage_metrics as shared_metrics

    return shared_metrics(
        mesh,
        radius=float(radius),
        lambda_value=float(lambda_value),
        rim_half_width_lambda=float(rim_half_width_lambda),
        outer_near_width_lambda=float(outer_near_width_lambda),
    )


def _pearson_correlation(x_vals: Sequence[float], y_vals: Sequence[float]) -> float:
    x = np.asarray([float(v) for v in x_vals], dtype=float)
    y = np.asarray([float(v) for v in y_vals], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan")
    xg = x[mask]
    yg = y[mask]
    x0 = xg - float(np.mean(xg))
    y0 = yg - float(np.mean(yg))
    den = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if den <= 1e-18:
        return float("nan")
    return float(np.dot(x0, y0) / den)


def _band_anisotropy_metrics(
    mesh,
    *,
    radius: float,
    lambda_value: float,
    leakage: dict[str, float],
) -> dict[str, float]:
    """Report per-band anisotropy metrics and leakage correlations."""
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return {
            "disk_core_hmax_over_hmin_mean": float("nan"),
            "rim_band_hmax_over_hmin_mean": float("nan"),
            "outer_near_hmax_over_hmin_mean": float("nan"),
            "outer_far_hmax_over_hmin_mean": float("nan"),
            "disk_core_edge_orientation_spread": float("nan"),
            "rim_band_edge_orientation_spread": float("nan"),
            "outer_near_edge_orientation_spread": float("nan"),
            "outer_far_edge_orientation_spread": float("nan"),
            "corr_hmax_over_hmin_vs_tphi_over_trad": float("nan"),
            "corr_orientation_spread_vs_tphi_over_trad": float("nan"),
        }

    pos = mesh.positions_view()
    tri_pos = pos[tri_rows]
    e01 = tri_pos[:, 1] - tri_pos[:, 0]
    e12 = tri_pos[:, 2] - tri_pos[:, 1]
    e20 = tri_pos[:, 0] - tri_pos[:, 2]
    l01 = np.linalg.norm(e01, axis=1)
    l12 = np.linalg.norm(e12, axis=1)
    l20 = np.linalg.norm(e20, axis=1)
    lmax = np.maximum.reduce([l01, l12, l20])
    lmin = np.minimum.reduce([l01, l12, l20])
    tri_aspect = lmax / np.maximum(lmin, 1e-18)
    tri_cent = np.mean(tri_pos, axis=1)
    tri_r = np.linalg.norm(tri_cent[:, :2], axis=1)

    rim_w = max(0.0, float(lambda_value))
    outer_near_w = max(0.0, 4.0 * float(lambda_value))
    r_disk_core_end = max(0.0, float(radius) - rim_w)
    r_rim_end = max(r_disk_core_end, float(radius) + rim_w)
    r_outer_near_end = max(r_rim_end, float(radius) + outer_near_w)

    tri_masks = {
        "disk_core": tri_r < r_disk_core_end,
        "rim_band": (tri_r >= r_disk_core_end) & (tri_r <= r_rim_end),
        "outer_near": (tri_r > r_rim_end) & (tri_r <= r_outer_near_end),
        "outer_far": tri_r > r_outer_near_end,
    }

    def _mean_aspect(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(tri_aspect[mask]))

    all_edges = np.vstack(
        [tri_rows[:, [0, 1]], tri_rows[:, [1, 2]], tri_rows[:, [2, 0]]]
    )
    sorted_edges = np.sort(all_edges, axis=1)
    edges = np.unique(sorted_edges, axis=0)
    p0 = pos[edges[:, 0]]
    p1 = pos[edges[:, 1]]
    dxy = p1[:, :2] - p0[:, :2]
    edge_len = np.linalg.norm(dxy, axis=1)
    valid = edge_len > 1e-12
    angles = np.zeros(edges.shape[0], dtype=float)
    angles[valid] = np.arctan2(dxy[valid, 1], dxy[valid, 0])
    mid = 0.5 * (p0 + p1)
    r_mid = np.linalg.norm(mid[:, :2], axis=1)
    edge_masks = {
        "disk_core": valid & (r_mid < r_disk_core_end),
        "rim_band": valid & (r_mid >= r_disk_core_end) & (r_mid <= r_rim_end),
        "outer_near": valid & (r_mid > r_rim_end) & (r_mid <= r_outer_near_end),
        "outer_far": valid & (r_mid > r_outer_near_end),
    }

    def _orientation_spread(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        th2 = 2.0 * angles[mask]
        c = float(np.mean(np.cos(th2)))
        s = float(np.mean(np.sin(th2)))
        resultant = float(np.clip(np.hypot(c, s), 1e-12, 1.0))
        return float(np.sqrt(max(0.0, -2.0 * np.log(resultant))) / 2.0)

    aspect_vec = []
    orient_vec = []
    leak_vec = []
    by_band: dict[str, float] = {}
    for band in ("disk_core", "rim_band", "outer_near", "outer_far"):
        asp = _mean_aspect(tri_masks[band])
        ori = _orientation_spread(edge_masks[band])
        leak = float(leakage.get(f"{band}_tphi_over_trad_median", float("nan")))
        by_band[f"{band}_hmax_over_hmin_mean"] = asp
        by_band[f"{band}_edge_orientation_spread"] = ori
        aspect_vec.append(asp)
        orient_vec.append(ori)
        leak_vec.append(
            float(np.log(max(leak, 1e-18))) if np.isfinite(leak) else float("nan")
        )

    by_band["corr_hmax_over_hmin_vs_tphi_over_trad"] = _pearson_correlation(
        aspect_vec, leak_vec
    )
    by_band["corr_orientation_spread_vs_tphi_over_trad"] = _pearson_correlation(
        orient_vec, leak_vec
    )
    return by_band


def _radial_projected_band_diagnostics(
    mesh,
    *,
    smoothness_model: str,
    radius: float,
    lambda_value: float,
    theory_bands: dict[str, float],
) -> dict[str, float]:
    """Compare current field to radial-only projection t <- (t·r_hat) r_hat."""
    pos = mesh.positions_view()
    _, r_hat, _ = _radial_frames(pos)
    tilts_orig = mesh.tilts_in_view().copy(order="F")
    t_rad = np.einsum("ij,ij->i", tilts_orig, r_hat)
    tilts_proj = r_hat * t_rad[:, None]

    mesh.set_tilts_in_from_array(tilts_proj)
    mesh.project_tilts_to_tangent()
    try:
        proj_region = _mesh_internal_region_split(
            mesh,
            smoothness_model=smoothness_model,
            radius=float(radius),
        )
        proj_bands = _mesh_internal_band_split(
            mesh,
            smoothness_model=smoothness_model,
            radius=float(radius),
            lambda_value=float(lambda_value),
            rim_half_width_lambda=1.0,
            outer_near_width_lambda=4.0,
        )
    finally:
        mesh.set_tilts_in_from_array(tilts_orig)
        mesh.project_tilts_to_tangent()

    def _abs_err(mesh_key: str, th_key: str) -> float:
        return float(abs(float(proj_bands[mesh_key]) - float(theory_bands[th_key])))

    return {
        "proj_radial_mesh_internal": float(
            proj_region["mesh_internal_total_from_regions"]
        ),
        "proj_radial_mesh_internal_disk": float(proj_region["mesh_internal_disk"]),
        "proj_radial_mesh_internal_outer": float(proj_region["mesh_internal_outer"]),
        "proj_radial_mesh_internal_disk_core": float(
            proj_bands["mesh_internal_disk_core"]
        ),
        "proj_radial_mesh_internal_rim_band": float(
            proj_bands["mesh_internal_rim_band"]
        ),
        "proj_radial_mesh_internal_outer_near": float(
            proj_bands["mesh_internal_outer_near"]
        ),
        "proj_radial_mesh_internal_outer_far": float(
            proj_bands["mesh_internal_outer_far"]
        ),
        "proj_radial_internal_disk_core_abs_error": _abs_err(
            "mesh_internal_disk_core", "theory_internal_disk_core"
        ),
        "proj_radial_internal_rim_band_abs_error": _abs_err(
            "mesh_internal_rim_band", "theory_internal_rim_band"
        ),
        "proj_radial_internal_outer_near_abs_error": _abs_err(
            "mesh_internal_outer_near", "theory_internal_outer_near"
        ),
        "proj_radial_internal_outer_far_abs_error": _abs_err(
            "mesh_internal_outer_far", "theory_internal_outer_far"
        ),
    }


def _resolution_metrics(
    mesh, *, radius: float, lambda_value: float
) -> dict[str, float]:
    """Report rim edge-length scale relative to the decay length lambda."""
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return {
            "rim_edge_count": 0,
            "rim_edge_length_median": float("nan"),
            "rim_edge_length_max": float("nan"),
            "rim_h_over_lambda_median": float("nan"),
        }

    pos = mesh.positions_view()
    all_edges = np.vstack(
        [
            tri_rows[:, [0, 1]],
            tri_rows[:, [1, 2]],
            tri_rows[:, [2, 0]],
        ]
    )
    sorted_edges = np.sort(all_edges, axis=1)
    edges = np.unique(sorted_edges, axis=0)

    p0 = pos[edges[:, 0]]
    p1 = pos[edges[:, 1]]
    mid = 0.5 * (p0 + p1)
    mid_r = np.linalg.norm(mid[:, :2], axis=1)
    lengths = np.linalg.norm(p1 - p0, axis=1)
    rim_mask = (mid_r >= (0.9 * float(radius))) & (mid_r <= (1.1 * float(radius)))
    rim_lengths = lengths[rim_mask]
    if rim_lengths.size == 0:
        return {
            "rim_edge_count": 0,
            "rim_edge_length_median": float("nan"),
            "rim_edge_length_max": float("nan"),
            "rim_h_over_lambda_median": float("nan"),
        }

    h_med = float(np.median(rim_lengths))
    return {
        "rim_edge_count": int(rim_lengths.size),
        "rim_edge_length_median": h_med,
        "rim_edge_length_max": float(np.max(rim_lengths)),
        "rim_h_over_lambda_median": float(h_med / max(float(lambda_value), 1e-18)),
    }
