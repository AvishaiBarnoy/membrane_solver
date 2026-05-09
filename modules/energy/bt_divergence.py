"""Recovered divergence logic for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh

from .bt_params import _bending_tilt_in_update_mode, _use_inner_recovered_divergence


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
    w = tri_area / 3.0
    if ctx is not None:
        v_area = ctx.scratch_array(
            f"{scratch_tag}_v_area", shape=(n_vertices,), dtype=float
        )
        v_div_num = ctx.scratch_array(
            f"{scratch_tag}_v_div_num", shape=(n_vertices,), dtype=float
        )
        v_div = ctx.scratch_array(
            f"{scratch_tag}_v_div", shape=(n_vertices,), dtype=float
        )
        div_eval = ctx.scratch_array(
            f"{scratch_tag}_div_eval", shape=div_tri.shape, dtype=float
        )
        v_area.fill(0.0)
        v_div_num.fill(0.0)
        v_div.fill(0.0)
    else:
        v_area = np.zeros(n_vertices, dtype=float)
        v_div_num = np.zeros(n_vertices, dtype=float)
        v_div = np.zeros(n_vertices, dtype=float)
        div_eval = np.zeros_like(div_tri)

    np.add.at(v_area, tri_rows[:, 0], w)
    np.add.at(v_area, tri_rows[:, 1], w)
    np.add.at(v_area, tri_rows[:, 2], w)
    np.add.at(v_div_num, tri_rows[:, 0], w * div_tri)
    np.add.at(v_div_num, tri_rows[:, 1], w * div_tri)
    np.add.at(v_div_num, tri_rows[:, 2], w * div_tri)

    good_v = v_area > 1.0e-20
    v_div[good_v] = v_div_num[good_v] / v_area[good_v]
    div_eval[:] = (
        v_div[tri_rows[:, 0]] + v_div[tri_rows[:, 1]] + v_div[tri_rows[:, 2]]
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

    n_vertices = int(v_area.shape[0])
    if ctx is not None:
        v_grad = ctx.scratch_array(
            f"{scratch_tag}_v_grad", shape=(n_vertices,), dtype=float
        )
        inv_v_area = ctx.scratch_array(
            f"{scratch_tag}_inv_v_area", shape=(n_vertices,), dtype=float
        )
        coeff_div = ctx.scratch_array(
            f"{scratch_tag}_coeff_div", shape=coeff_div_eval.shape, dtype=float
        )
        v_grad.fill(0.0)
        inv_v_area.fill(0.0)
    else:
        v_grad = np.zeros(n_vertices, dtype=float)
        inv_v_area = np.zeros_like(v_area)
        coeff_div = np.zeros_like(coeff_div_eval)

    np.add.at(v_grad, tri_rows[:, 0], coeff_div_eval / 3.0)
    np.add.at(v_grad, tri_rows[:, 1], coeff_div_eval / 3.0)
    np.add.at(v_grad, tri_rows[:, 2], coeff_div_eval / 3.0)
    good_v = v_area > 1.0e-20
    inv_v_area[good_v] = 1.0 / v_area[good_v]
    coeff_div[:] = (tri_area / 3.0) * (
        v_grad[tri_rows[:, 0]] * inv_v_area[tri_rows[:, 0]]
        + v_grad[tri_rows[:, 1]] * inv_v_area[tri_rows[:, 1]]
        + v_grad[tri_rows[:, 2]] * inv_v_area[tri_rows[:, 2]]
    )
    return coeff_div
