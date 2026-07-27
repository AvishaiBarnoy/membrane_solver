"""Recovered divergence logic for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh

from .bt_params import (
    _bending_tilt_in_update_mode,
    _inner_recovered_divergence_domain_mode,
    _use_inner_recovered_divergence,
)


def _inner_recovery_triangle_domains(
    mesh: Mesh,
    global_params,
    *,
    cache_tag: str,
    tri_rows: np.ndarray,
) -> np.ndarray | None:
    """Return disc-side/outer-side triangle labels for one-sided recovery."""
    if (
        str(cache_tag) != "in"
        or _inner_recovered_divergence_domain_mode(global_params)
        != "physical_disk_one_sided_v1"
    ):
        return None
    disk_group = str(global_params.get("rim_slope_match_disk_group") or "disk").strip()
    disk_rows = np.array(
        [
            str(
                (getattr(mesh.vertices[int(vid)], "options", {}) or {}).get(
                    "pin_to_circle_group"
                )
                or ""
            )
            == disk_group
            for vid in mesh.vertex_ids
        ],
        dtype=bool,
    )
    # Triangles wholly inside the rigid disc are domain 0. Triangles containing
    # any membrane vertex are domain 1 and share only a one-sided rim trace.
    return np.asarray(~np.all(disk_rows[tri_rows], axis=1), dtype=np.int32)


def _inner_bending_tilt_dE_ddiv(
    *,
    mesh: Mesh,
    global_params,
    cache_tag: str,
    kappa_tri: np.ndarray,
    base_tri: np.ndarray,
    div_term: np.ndarray,
    va0_eff: np.ndarray,
    va1_eff: np.ndarray,
    va2_eff: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Return inner divergence gradient contribution under benchmark modes."""
    mode = _bending_tilt_in_update_mode(global_params)
    stats = {
        "enabled": bool(mode != "off"),
        "mode": str(mode),
        "candidate_tri_count": 0,
        "capped_tri_count": 0,
        "rim_tri_count": 0,
        "cap_magnitude": 0.0,
        "cross_term_removed": False,
    }
    if str(cache_tag) != "in":
        return (
            (kappa_tri[:, 0] * (base_tri[:, 0] + div_term) * va0_eff)
            + (kappa_tri[:, 1] * (base_tri[:, 1] + div_term) * va1_eff)
            + (kappa_tri[:, 2] * (base_tri[:, 2] + div_term) * va2_eff),
            stats,
        )
    if mode == "radial_cross_term_off_v1":
        stats["cross_term_removed"] = True
        setattr(mesh, "_last_bending_tilt_in_update_mode_stats", stats)
        return (
            (kappa_tri[:, 0] * div_term * va0_eff)
            + (kappa_tri[:, 1] * div_term * va1_eff)
            + (kappa_tri[:, 2] * div_term * va2_eff)
        ), stats
    return (
        (kappa_tri[:, 0] * (base_tri[:, 0] + div_term) * va0_eff)
        + (kappa_tri[:, 1] * (base_tri[:, 1] + div_term) * va1_eff)
        + (kappa_tri[:, 2] * (base_tri[:, 2] + div_term) * va2_eff)
    ), stats


def _inner_recovered_divergence(
    *,
    global_params,
    cache_tag: str,
    tri_rows: np.ndarray,
    tri_area: np.ndarray,
    div_tri: np.ndarray,
    n_vertices: int,
    triangle_domains: np.ndarray | None = None,
    ctx=None,
    scratch_tag: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return divergence used for inner-leaflet evaluation.

    For the inner leaflet, recover a per-vertex divergence from surrounding
    triangle values using barycentric area weights, then average it back to
    triangles. Other leaflets keep the raw constant-per-triangle divergence.
    """
    div_tri = np.asarray(div_tri, dtype=float)
    if str(cache_tag) != "in" or div_tri.size == 0:
        return div_tri, None, None
    if not _use_inner_recovered_divergence(global_params, cache_tag=cache_tag):
        return div_tri, None, None

    tri_area = np.asarray(tri_area, dtype=float)
    domains = None
    storage_shape = (n_vertices,)
    if triangle_domains is not None:
        domains = np.asarray(triangle_domains, dtype=np.int32)
        if domains.shape != div_tri.shape or np.any(domains < 0):
            raise ValueError("triangle_domains must be nonnegative with shape (N_tri,)")
        storage_shape = (int(np.max(domains, initial=0)) + 1, n_vertices)
    w = tri_area / 3.0
    if ctx is not None:
        v_area = ctx.scratch_array(
            f"{scratch_tag}_v_area", shape=storage_shape, dtype=float
        )
        v_div_num = ctx.scratch_array(
            f"{scratch_tag}_v_div_num", shape=storage_shape, dtype=float
        )
        v_div = ctx.scratch_array(
            f"{scratch_tag}_v_div", shape=storage_shape, dtype=float
        )
        div_eval = ctx.scratch_array(
            f"{scratch_tag}_div_eval", shape=div_tri.shape, dtype=float
        )
        v_area.fill(0.0)
        v_div_num.fill(0.0)
        v_div.fill(0.0)
    else:
        v_area = np.zeros(storage_shape, dtype=float)
        v_div_num = np.zeros(storage_shape, dtype=float)
        v_div = np.zeros(storage_shape, dtype=float)
        div_eval = np.zeros_like(div_tri)

    if domains is None:
        for corner in range(3):
            np.add.at(v_area, tri_rows[:, corner], w)
            np.add.at(v_div_num, tri_rows[:, corner], w * div_tri)
    else:
        for corner in range(3):
            indices = (domains, tri_rows[:, corner])
            np.add.at(v_area, indices, w)
            np.add.at(v_div_num, indices, w * div_tri)

    good_v = v_area > 1.0e-20
    v_div[good_v] = v_div_num[good_v] / v_area[good_v]
    if domains is None:
        div_eval[:] = (
            v_div[tri_rows[:, 0]] + v_div[tri_rows[:, 1]] + v_div[tri_rows[:, 2]]
        ) / 3.0
    else:
        div_eval[:] = (
            v_div[domains, tri_rows[:, 0]]
            + v_div[domains, tri_rows[:, 1]]
            + v_div[domains, tri_rows[:, 2]]
        ) / 3.0
    return div_eval, v_div, v_area


def _inner_recovered_divergence_pullback(
    *,
    global_params,
    cache_tag: str,
    tri_rows: np.ndarray,
    tri_area: np.ndarray,
    coeff_div_eval: np.ndarray,
    v_area: np.ndarray | None,
    triangle_domains: np.ndarray | None = None,
    ctx=None,
    scratch_tag: str,
) -> np.ndarray:
    """Map dE/d(div_eval) back to raw triangle-divergence coefficients."""
    coeff_div_eval = np.asarray(coeff_div_eval, dtype=float)
    if str(cache_tag) != "in" or coeff_div_eval.size == 0:
        return coeff_div_eval
    if not _use_inner_recovered_divergence(global_params, cache_tag=cache_tag):
        return coeff_div_eval
    if v_area is None:
        raise ValueError("Recovered inner divergence requires vertex areas.")

    domains = (
        None
        if triangle_domains is None
        else np.asarray(triangle_domains, dtype=np.int32)
    )
    if v_area.ndim == 2 and (domains is None or domains.shape != coeff_div_eval.shape):
        raise ValueError("Domain-separated recovery requires triangle_domains.")
    storage_shape = v_area.shape
    if ctx is not None:
        v_grad = ctx.scratch_array(
            f"{scratch_tag}_v_grad", shape=storage_shape, dtype=float
        )
        inv_v_area = ctx.scratch_array(
            f"{scratch_tag}_inv_v_area", shape=storage_shape, dtype=float
        )
        coeff_div = ctx.scratch_array(
            f"{scratch_tag}_coeff_div", shape=coeff_div_eval.shape, dtype=float
        )
        v_grad.fill(0.0)
        inv_v_area.fill(0.0)
    else:
        v_grad = np.zeros(storage_shape, dtype=float)
        inv_v_area = np.zeros_like(v_area)
        coeff_div = np.zeros_like(coeff_div_eval)

    for corner in range(3):
        indices = tri_rows[:, corner]
        if domains is not None:
            indices = (domains, indices)
        np.add.at(v_grad, indices, coeff_div_eval / 3.0)
    good_v = v_area > 1.0e-20
    inv_v_area[good_v] = 1.0 / v_area[good_v]
    if domains is None:
        coeff_div[:] = (tri_area / 3.0) * (
            v_grad[tri_rows[:, 0]] * inv_v_area[tri_rows[:, 0]]
            + v_grad[tri_rows[:, 1]] * inv_v_area[tri_rows[:, 1]]
            + v_grad[tri_rows[:, 2]] * inv_v_area[tri_rows[:, 2]]
        )
    else:
        coeff_div[:] = (tri_area / 3.0) * (
            v_grad[domains, tri_rows[:, 0]] * inv_v_area[domains, tri_rows[:, 0]]
            + v_grad[domains, tri_rows[:, 1]] * inv_v_area[domains, tri_rows[:, 1]]
            + v_grad[domains, tri_rows[:, 2]] * inv_v_area[domains, tri_rows[:, 2]]
        )
    return coeff_div


def _inner_recovered_divergence_area_pullback(
    *,
    global_params,
    cache_tag: str,
    tri_rows: np.ndarray,
    div_tri: np.ndarray,
    coeff_div_eval: np.ndarray,
    v_div: np.ndarray | None,
    v_area: np.ndarray | None,
    triangle_domains: np.ndarray | None = None,
    ctx=None,
    scratch_tag: str,
) -> np.ndarray:
    """Map recovered-divergence sensitivity to triangle-area coefficients."""
    coeff_div_eval = np.asarray(coeff_div_eval, dtype=float)
    if str(cache_tag) != "in" or coeff_div_eval.size == 0:
        return np.zeros_like(coeff_div_eval)
    if not _use_inner_recovered_divergence(global_params, cache_tag=cache_tag):
        return np.zeros_like(coeff_div_eval)
    if v_div is None or v_area is None:
        raise ValueError("Recovered inner divergence requires vertex values and areas.")

    domains = (
        None
        if triangle_domains is None
        else np.asarray(triangle_domains, dtype=np.int32)
    )
    if v_area.ndim == 2 and (domains is None or domains.shape != coeff_div_eval.shape):
        raise ValueError("Domain-separated recovery requires triangle_domains.")
    storage_shape = v_area.shape
    if ctx is not None:
        v_grad = ctx.scratch_array(
            f"{scratch_tag}_area_v_grad", shape=storage_shape, dtype=float
        )
        inv_v_area = ctx.scratch_array(
            f"{scratch_tag}_area_inv_v_area", shape=storage_shape, dtype=float
        )
        coeff_area = ctx.scratch_array(
            f"{scratch_tag}_coeff_area", shape=coeff_div_eval.shape, dtype=float
        )
        v_grad.fill(0.0)
        inv_v_area.fill(0.0)
    else:
        v_grad = np.zeros(storage_shape, dtype=float)
        inv_v_area = np.zeros_like(v_area)
        coeff_area = np.zeros_like(coeff_div_eval)

    for corner in range(3):
        indices = tri_rows[:, corner]
        if domains is not None:
            indices = (domains, indices)
        np.add.at(v_grad, indices, coeff_div_eval / 3.0)
    good_v = v_area > 1.0e-20
    inv_v_area[good_v] = 1.0 / v_area[good_v]

    div_tri = np.asarray(div_tri, dtype=float)
    if domains is None:
        coeff_area[:] = (
            v_grad[tri_rows[:, 0]]
            * (div_tri - v_div[tri_rows[:, 0]])
            * inv_v_area[tri_rows[:, 0]]
            + v_grad[tri_rows[:, 1]]
            * (div_tri - v_div[tri_rows[:, 1]])
            * inv_v_area[tri_rows[:, 1]]
            + v_grad[tri_rows[:, 2]]
            * (div_tri - v_div[tri_rows[:, 2]])
            * inv_v_area[tri_rows[:, 2]]
        ) / 3.0
    else:
        coeff_area[:] = (
            v_grad[domains, tri_rows[:, 0]]
            * (div_tri - v_div[domains, tri_rows[:, 0]])
            * inv_v_area[domains, tri_rows[:, 0]]
            + v_grad[domains, tri_rows[:, 1]]
            * (div_tri - v_div[domains, tri_rows[:, 1]])
            * inv_v_area[domains, tri_rows[:, 1]]
            + v_grad[domains, tri_rows[:, 2]]
            * (div_tri - v_div[domains, tri_rows[:, 2]])
            * inv_v_area[domains, tri_rows[:, 2]]
        ) / 3.0
    return coeff_area
