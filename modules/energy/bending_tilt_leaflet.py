"""Shared helpers for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.bending_derivatives import grad_triangle_area
from geometry.curvature import compute_curvature_data
from geometry.entities import Mesh
from geometry.tilt_operators import (
    _resolve_transport_model,
    compute_divergence_from_basis,
    p1_triangle_divergence,
)
from modules.energy.bending import (  # noqa: PLC0415
    _apply_beltrami_laplacian,
    _cached_cotan_gradients,
    _compute_effective_areas,
    _energy_model,
    _grad_cotan,
    _gradient_mode,
    _spontaneous_curvature,
    _vertex_normals,
)
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)
from modules.energy.scatter import scatter_triangle_scalar_to_vertices

_BASE_TERM_BOUNDARY_OPTION_KEYS = (
    # Most configs tag the disk interface ring via the rim-slope match group.
    "rim_slope_match_group",
    # Some configs use explicit thetaB group tags.
    "tilt_thetaB_group",
    "tilt_thetaB_group_in",
    "tilt_thetaB_group_out",
)

_ASSUME_J0_PRESETS_KEY = "bending_tilt_assume_J0_presets"


def _assume_J0_presets(global_params, *, cache_tag: str) -> tuple[str, ...]:
    """Optional config: presets for which the Helfrich base term is set to zero.

    This is a theory-mode knob used to match formulations that assume the
    geometric curvature contribution J (i.e. 2H - c0) vanishes on a tagged
    patch such as a rigid disk. It is off by default.
    """
    if global_params is None:
        return ()
    raw = global_params.get(f"{_ASSUME_J0_PRESETS_KEY}_{cache_tag}")
    if raw is None:
        raw = global_params.get(_ASSUME_J0_PRESETS_KEY)
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [raw]
    else:
        try:
            items = list(raw)
        except TypeError:
            items = [raw]

    presets: list[str] = []
    for item in items:
        name = str(item).strip()
        if name:
            presets.append(name)
    return tuple(presets)


def _assume_J0_radius_max(global_params, *, cache_tag: str) -> float | None:
    """Optional config: radial cap for theory-mode J0 suppression rows."""
    if global_params is None:
        return None
    raw = global_params.get(f"{_ASSUME_J0_PRESETS_KEY}_radius_max_{cache_tag}")
    if raw is None:
        raw = global_params.get(f"{_ASSUME_J0_PRESETS_KEY}_radius_max")
    if raw is None:
        return None
    radius_max = float(raw)
    if radius_max < 0.0:
        raise ValueError("bending_tilt_assume_J0_presets_radius_max must be >= 0.")
    return radius_max


def _assume_J0_center_xy(global_params) -> np.ndarray:
    """Return the xy center used for radial J0-suppression clipping."""
    if global_params is None:
        return np.zeros(2, dtype=float)
    raw = global_params.get("tilt_thetaB_center")
    if raw is None:
        raw = global_params.get("pin_to_circle_point")
    if raw is None:
        return np.zeros(2, dtype=float)
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size < 2:
        return np.zeros(2, dtype=float)
    return arr[:2].astype(float, copy=False)


def _base_term_region_mode(global_params) -> str:
    """Return optional benchmark-scoped base-term region mode."""
    if global_params is None:
        return "off"
    raw = global_params.get("bending_tilt_base_term_region_mode")
    mode = str(raw or "off").strip().lower()
    if mode not in {"off", "physical_disk_split_v1", "disk_only_base_term_v1"}:
        raise ValueError(
            "bending_tilt_base_term_region_mode must be 'off' or "
            "'physical_disk_split_v1' or 'disk_only_base_term_v1'."
        )
    return mode


def _base_term_region_radius(global_params) -> float | None:
    """Return physical disk radius used by base-term region modes."""
    if global_params is None:
        return None
    raw = global_params.get("bending_tilt_base_term_region_radius")
    if raw is None:
        return None
    radius = float(raw)
    if radius < 0.0:
        raise ValueError("bending_tilt_base_term_region_radius must be >= 0.")
    return radius


def _bending_tilt_in_update_mode(global_params) -> str:
    """Return optional benchmark-scoped inner bending-tilt update mode."""
    if global_params is None:
        return "off"
    raw = global_params.get("bending_tilt_in_update_mode")
    mode = str(raw or "off").strip().lower()
    if mode not in {
        "off",
        "outer_near_divergence_cap_v1",
        "radial_cross_term_off_v1",
    }:
        raise ValueError(
            "bending_tilt_in_update_mode must be 'off' or "
            "'outer_near_divergence_cap_v1' or 'radial_cross_term_off_v1'."
        )
    return mode


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


def _accumulate_leaflet_tilt_gradient(
    tilt_grad_arr: np.ndarray,
    tri_rows: np.ndarray,
    factor: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    *,
    ctx=None,
    scratch_tag: str,
) -> None:
    """Accumulate tilt gradients while reusing scratch buffers when available."""
    if ctx is None:
        np.add.at(tilt_grad_arr, tri_rows[:, 0], factor * g0)
        np.add.at(tilt_grad_arr, tri_rows[:, 1], factor * g1)
        np.add.at(tilt_grad_arr, tri_rows[:, 2], factor * g2)
        return

    scaled = ctx.scratch_array(scratch_tag, shape=g0.shape, dtype=g0.dtype)
    np.multiply(factor, g0, out=scaled)
    np.add.at(tilt_grad_arr, tri_rows[:, 0], scaled)
    np.multiply(factor, g1, out=scaled)
    np.add.at(tilt_grad_arr, tri_rows[:, 1], scaled)
    np.multiply(factor, g2, out=scaled)
    np.add.at(tilt_grad_arr, tri_rows[:, 2], scaled)


def _inner_recovered_divergence(
    *,
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


def _apply_inner_divergence_update_mode(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    tri_rows: np.ndarray,
    cache_tag: str,
    div_term: np.ndarray,
) -> np.ndarray:
    """Apply benchmark-only inner divergence cap beyond the rim."""
    stats_attr = "_last_bending_tilt_in_update_mode_stats"
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
        return div_term
    if mode == "off" or tri_rows.size == 0:
        setattr(mesh, stats_attr, stats)
        return div_term
    if mode == "radial_cross_term_off_v1":
        stats["cross_term_removed"] = True
        setattr(mesh, stats_attr, stats)
        return div_term

    radius = float(global_params.get("benchmark_disk_radius") or 0.0)
    lambda_value = float(global_params.get("benchmark_lambda_value") or 0.0)
    if radius <= 0.0 or lambda_value <= 0.0:
        setattr(mesh, stats_attr, stats)
        return div_term

    center = _assume_J0_center_xy(global_params)
    tri_xy = np.mean(positions[tri_rows, :2], axis=1)
    tri_radii = np.linalg.norm(tri_xy - center[None, :], axis=1)
    rim_w = float(lambda_value)
    near_w = 4.0 * float(lambda_value)
    rim_mask = np.abs(tri_radii - radius) <= rim_w
    outer_near_mask = (tri_radii > (radius + rim_w)) & (tri_radii <= (radius + near_w))
    stats["candidate_tri_count"] = int(np.sum(outer_near_mask))
    stats["rim_tri_count"] = int(np.sum(rim_mask))
    rim_mag = np.abs(div_term[rim_mask])
    if rim_mag.size == 0:
        setattr(mesh, stats_attr, stats)
        return div_term

    cap_magnitude = float(1.05 * np.median(rim_mag))
    stats["cap_magnitude"] = cap_magnitude
    if cap_magnitude <= 0.0 or not np.any(outer_near_mask):
        setattr(mesh, stats_attr, stats)
        return div_term

    updated = np.array(div_term, copy=True)
    hit = outer_near_mask & (np.abs(updated) > cap_magnitude)
    updated[hit] = np.sign(updated[hit]) * cap_magnitude
    stats["capped_tri_count"] = int(np.sum(hit))
    setattr(mesh, stats_attr, stats)
    return updated


def _collect_preset_rows(
    mesh: Mesh,
    *,
    presets: tuple[str, ...],
    cache_tag: str,
    index_map: Dict[int, int],
    radius_max: float | None = None,
    center_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Return vertex-row indices whose ``preset`` option is in ``presets``."""
    if not presets:
        return np.zeros(0, dtype=int)
    radius_key = None if radius_max is None else float(radius_max)
    center_key = None
    if center_xy is not None:
        center_arr = np.asarray(center_xy, dtype=float).reshape(-1)
        if center_arr.size >= 2:
            center_key = (float(center_arr[0]), float(center_arr[1]))
    cache_key = (mesh._vertex_ids_version, presets, radius_key, center_key)
    cache_attr = f"_bending_tilt_assume_J0_rows_cache_{cache_tag}"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["rows"]
    presets_set = set(presets)
    present: set[str] = set()
    center = (
        np.zeros(2, dtype=float)
        if center_xy is None
        else np.asarray(center_xy, dtype=float)
        .reshape(-1)[:2]
        .astype(float, copy=False)
    )
    radius_tol = 1.0e-12

    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        preset = str(opts.get("preset") or "")
        present.add(preset)
        if preset in presets_set:
            row = index_map.get(int(vid))
            if row is not None:
                if radius_max is not None:
                    pos = np.asarray(mesh.vertices[int(vid)].position, dtype=float)
                    radius = float(np.linalg.norm(pos[:2] - center))
                    if radius > float(radius_max) + radius_tol:
                        continue
                rows.append(int(row))

    unknown = presets_set - present
    if unknown:
        raise ValueError(
            "Unknown presets in bending_tilt_assume_J0_presets: "
            + ", ".join(sorted(unknown))
        )

    out = np.asarray(rows, dtype=int)
    setattr(mesh, cache_attr, {"key": cache_key, "rows": out})
    return out


def _collect_group_rows(
    mesh: Mesh, *, group: str, index_map: Dict[int, int]
) -> np.ndarray:
    """Return vertex-row indices whose options tag them as members of ``group``."""
    cache_key = (mesh._vertex_ids_version, str(group))
    cache_attr = "_bending_tilt_group_rows_cache"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["rows"]

    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if any(opts.get(key) == group for key in _BASE_TERM_BOUNDARY_OPTION_KEYS):
            row = index_map.get(int(vid))
            if row is not None:
                rows.append(int(row))
    out = np.asarray(rows, dtype=int)
    setattr(mesh, cache_attr, {"key": cache_key, "rows": out})
    return out


def _base_term_region_zero_rows(
    mesh: Mesh,
    global_params,
    *,
    cache_tag: str,
    index_map: Dict[int, int],
) -> np.ndarray:
    """Return extra rows zeroed by benchmark-scoped base-term region modes."""
    mode = _base_term_region_mode(global_params)
    if mode == "off":
        return np.zeros(0, dtype=int)
    radius = _base_term_region_radius(global_params)
    if radius is None:
        raise ValueError(
            "bending_tilt_base_term_region_radius is required when "
            "bending_tilt_base_term_region_mode is enabled."
        )
    if mode in {"physical_disk_split_v1", "disk_only_base_term_v1"}:
        if mode == "physical_disk_split_v1" and str(cache_tag) != "out":
            return np.zeros(0, dtype=int)
        if mode == "disk_only_base_term_v1" and str(cache_tag) != "in":
            return np.zeros(0, dtype=int)
        center = _assume_J0_center_xy(global_params)
        cache_key = (
            mesh._vertex_ids_version,
            str(cache_tag),
            str(mode),
            float(radius),
            float(center[0]),
            float(center[1]),
        )
        cache_attr = "_bending_tilt_base_term_region_rows_cache"
        cached = getattr(mesh, cache_attr, None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["rows"]
        rows: list[int] = []
        for vid in mesh.vertex_ids:
            row = index_map.get(int(vid))
            if row is None:
                continue
            pos = np.asarray(mesh.vertices[int(vid)].position, dtype=float)
            row_radius = float(np.linalg.norm(pos[:2] - center))
            if mode == "physical_disk_split_v1":
                if row_radius <= radius + 1.0e-12:
                    rows.append(int(row))
            else:
                if row_radius > radius + 1.0e-12:
                    rows.append(int(row))
        out = np.asarray(rows, dtype=int)
        setattr(mesh, cache_attr, {"key": cache_key, "rows": out})
        return out
    raise AssertionError("unreachable")


def _base_term_boundary_group(global_params, *, cache_tag: str) -> str | None:
    """Optional config: treat a tagged interface ring as a base-term boundary."""
    if global_params is None:
        return None
    raw = global_params.get(f"bending_tilt_base_term_boundary_group_{cache_tag}")
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _interior_mask_leaflet(
    mesh: Mesh,
    global_params,
    *,
    cache_tag: str,
    index_map: Dict[int, int],
) -> np.ndarray:
    """Return cached interior mask for base-term evaluation.

    Interior vertices are all non-boundary vertices, optionally excluding a
    configured interface group and preset list used for theory-mode matching.
    """
    group = _base_term_boundary_group(global_params, cache_tag=cache_tag)
    cache_key = (
        int(mesh._vertex_ids_version),
        int(mesh._topology_version),
        None if group is None else str(group),
    )
    cache_attr = f"_bending_tilt_interior_mask_cache_{cache_tag}"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["mask"]

    is_interior = np.ones(len(mesh.vertex_ids), dtype=bool)

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        if boundary_rows:
            is_interior[np.asarray(boundary_rows, dtype=int)] = False

    if group:
        rows = _collect_group_rows(mesh, group=group, index_map=index_map)
        if rows.size:
            is_interior[rows] = False

    setattr(mesh, cache_attr, {"key": cache_key, "mask": is_interior})
    return is_interior


def _resolve_bending_modulus(global_params, kappa_key: str) -> float:
    """Return the leaflet-specific bending modulus or the global default."""
    val = global_params.get(kappa_key)
    if val is None:
        val = global_params.get("bending_modulus", 0.0)
    return float(val or 0.0)


def _per_vertex_params_leaflet(
    mesh: Mesh,
    global_params,
    *,
    model: str,
    kappa_key: str,
    cache_tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-vertex (kappa, c0) arrays using a leaflet default modulus."""
    mesh.build_position_cache()

    n = len(mesh.vertex_ids)
    kappa_default = _resolve_bending_modulus(global_params, kappa_key)

    # Resolve leaflet-specific spontaneous curvature default
    c0_key = f"spontaneous_curvature_{cache_tag}"
    c0_default = global_params.get(c0_key)
    if c0_default is None:
        c0_default = (
            _spontaneous_curvature(global_params) if model == "helfrich" else 0.0
        )
    c0_default = float(c0_default or 0.0)

    cache_key = (
        mesh._vertex_ids_version,
        model,
        float(kappa_default),
        float(c0_default),
    )
    cache_attr = f"_bending_leaflet_param_cache_{cache_tag}"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["kappa"], cached["c0"]

    kappa = np.full(n, kappa_default, dtype=float)
    c0 = np.full(n, c0_default, dtype=float)

    override_rows_k: list[int] = []
    override_vals_k: list[float] = []
    override_rows_c0: list[int] = []
    override_vals_c0: list[float] = []

    for vid, vertex in mesh.vertices.items():
        row = mesh.vertex_index_to_row.get(int(vid))
        if row is None:
            continue
        opts = getattr(vertex, "options", None) or {}
        if kappa_key in opts:
            try:
                override_rows_k.append(row)
                override_vals_k.append(float(opts[kappa_key]))
            except (TypeError, ValueError):
                pass
        elif "bending_modulus" in opts:
            try:
                override_rows_k.append(row)
                override_vals_k.append(float(opts["bending_modulus"]))
            except (TypeError, ValueError):
                pass

        if model == "helfrich":
            # Per-vertex c0 resolution: leaflet-specific -> generic
            v_c0 = opts.get(c0_key)
            if v_c0 is None:
                v_c0 = opts.get("spontaneous_curvature")
            if v_c0 is None:
                v_c0 = opts.get("intrinsic_curvature")

            if v_c0 is not None:
                try:
                    override_rows_c0.append(row)
                    override_vals_c0.append(float(v_c0))
                except (TypeError, ValueError):
                    pass

    if override_rows_k:
        kappa[np.asarray(override_rows_k, dtype=int)] = np.asarray(
            override_vals_k, dtype=float
        )
    if model == "helfrich" and override_rows_c0:
        c0[np.asarray(override_rows_c0, dtype=int)] = np.asarray(
            override_vals_c0, dtype=float
        )

    setattr(mesh, cache_attr, {"key": cache_key, "kappa": kappa, "c0": c0})
    return kappa, c0


def _leaflet_triangle_payload(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    cache_tag: str,
    ctx=None,
) -> dict[str, np.ndarray | None]:
    """Return cached leaflet-masked triangle geometry for fixed positions."""
    mesh.build_position_cache()
    absent_mask = None
    absent_key = None
    if cache_tag in {"in", "out"}:
        absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet=cache_tag)
        absent_key = id(absent_mask)

    use_cache = mesh._geometry_cache_active(positions)
    cache_attr = f"_bending_tilt_leaflet_triangle_payload_cache_{cache_tag}"
    cache_key = (
        int(mesh._version),
        int(mesh._facet_loops_version),
        int(mesh._vertex_ids_version),
        id(positions),
        absent_key,
    )
    if use_cache:
        cached = getattr(mesh, cache_attr, None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["value"]

    k_vecs, vertex_areas_vor, weights_full, tri_rows_full = compute_curvature_data(
        mesh, positions, index_map
    )
    if tri_rows_full.size == 0:
        payload = {
            "k_vecs": k_vecs,
            "vertex_areas_vor": vertex_areas_vor,
            "weights_full": weights_full,
            "tri_rows_full": tri_rows_full,
            "weights": weights_full,
            "tri_rows": tri_rows_full,
            "tri_keep": np.zeros(0, dtype=bool),
            "g0": None,
            "g1": None,
            "g2": None,
        }
        if use_cache:
            setattr(mesh, cache_attr, {"key": cache_key, "value": payload})
        return payload

    tri_keep = np.zeros(0, dtype=bool)
    weights = weights_full
    tri_rows = tri_rows_full
    if absent_mask is not None:
        tri_keep = leaflet_present_triangle_mask(
            mesh, tri_rows_full, absent_vertex_mask=absent_mask
        )
        if tri_keep.size:
            tri_rows = tri_rows_full[tri_keep]
            weights = weights_full[tri_keep]

    g0_use = None
    g1_use = None
    g2_use = None
    area_use = None
    if ctx is not None:
        _area_cache, g0_cache, g1_cache, g2_cache, tri_rows_cache = (
            ctx.geometry.p1_triangle_shape_gradients(mesh, positions)
        )
    else:
        _area_cache, g0_cache, g1_cache, g2_cache, tri_rows_cache = (
            mesh.p1_triangle_shape_gradient_cache(positions)
        )
    if tri_rows_cache.size and tri_rows_cache.shape[0] == tri_rows_full.shape[0]:
        if tri_keep.size:
            area_use = _area_cache[tri_keep]
            g0_use = g0_cache[tri_keep]
            g1_use = g1_cache[tri_keep]
            g2_use = g2_cache[tri_keep]
        else:
            area_use = _area_cache
            g0_use = g0_cache
            g1_use = g1_cache
            g2_use = g2_cache

    payload = {
        "k_vecs": k_vecs,
        "vertex_areas_vor": vertex_areas_vor,
        "weights_full": weights_full,
        "tri_rows_full": tri_rows_full,
        "weights": weights,
        "tri_rows": tri_rows,
        "tri_keep": tri_keep,
        "tri_area": area_use,
        "g0": g0_use,
        "g1": g1_use,
        "g2": g2_use,
    }
    if use_cache:
        setattr(mesh, cache_attr, {"key": cache_key, "value": payload})
    return payload


def _leaflet_static_tilt_payload(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    k_vecs: np.ndarray,
    vertex_areas_vor: np.ndarray,
    tri_rows: np.ndarray,
    kappa_key: str,
    cache_tag: str,
) -> dict[str, np.ndarray]:
    """Return cached fixed-geometry arrays used by tilt-only leaflet coupling."""
    model = _energy_model(global_params)
    if model != "helfrich":
        model = "helfrich"
    kappa_default = _resolve_bending_modulus(global_params, kappa_key)
    c0_key = f"spontaneous_curvature_{cache_tag}"
    c0_default = global_params.get(c0_key)
    if c0_default is None:
        c0_default = (
            _spontaneous_curvature(global_params) if model == "helfrich" else 0.0
        )
    c0_default = float(c0_default or 0.0)

    presets = _assume_J0_presets(global_params, cache_tag=cache_tag)
    radius_max = _assume_J0_radius_max(global_params, cache_tag=cache_tag)
    center_xy = _assume_J0_center_xy(global_params)
    center_key = (float(center_xy[0]), float(center_xy[1]))
    region_mode = _base_term_region_mode(global_params)
    region_radius = _base_term_region_radius(global_params)
    boundary_group = _base_term_boundary_group(global_params, cache_tag=cache_tag)

    use_cache = mesh._geometry_cache_active(positions)
    cache_attr = f"_bending_tilt_leaflet_static_cache_{cache_tag}"
    cache_key = (
        int(mesh._version),
        int(mesh._facet_loops_version),
        int(mesh._vertex_ids_version),
        int(mesh._topology_version),
        id(positions),
        id(k_vecs),
        id(vertex_areas_vor),
        id(tri_rows),
        str(model),
        float(kappa_default),
        float(c0_default),
        None if boundary_group is None else str(boundary_group),
        presets,
        None if radius_max is None else float(radius_max),
        center_key,
        str(region_mode),
        None if region_radius is None else float(region_radius),
    )
    if use_cache:
        cached = getattr(mesh, cache_attr, None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["value"]

    kappa_arr, c0_arr = _per_vertex_params_leaflet(
        mesh, global_params, model=model, kappa_key=kappa_key, cache_tag=cache_tag
    )
    safe_areas_vor = np.maximum(vertex_areas_vor, 1e-12)
    k_mag = np.linalg.norm(k_vecs, axis=1)
    h_vor = k_mag / (2.0 * safe_areas_vor)
    is_interior = _interior_mask_leaflet(
        mesh, global_params, cache_tag=cache_tag, index_map=index_map
    )

    base_term = (2.0 * h_vor) - c0_arr
    base_term[~is_interior] = 0.0
    if presets:
        rows = _collect_preset_rows(
            mesh,
            presets=presets,
            cache_tag=cache_tag,
            index_map=index_map,
            radius_max=radius_max,
            center_xy=center_xy,
        )
        if rows.size:
            base_term[rows] = 0.0
    region_rows = _base_term_region_zero_rows(
        mesh,
        global_params,
        cache_tag=cache_tag,
        index_map=index_map,
    )
    if region_rows.size:
        base_term[region_rows] = 0.0

    value = {
        "base_tri": base_term[tri_rows],
        "kappa_tri": kappa_arr[tri_rows],
    }
    if use_cache:
        setattr(mesh, cache_attr, {"key": cache_key, "value": value})
    return value


def _total_energy_leaflet(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts: np.ndarray,
    kappa_key: str,
    cache_tag: str,
    div_sign: float,
) -> float:
    """Energy-only helper for finite-difference debugging."""
    payload = _leaflet_triangle_payload(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
        cache_tag=cache_tag,
    )
    k_vecs = np.asarray(payload["k_vecs"], dtype=float)
    vertex_areas_vor = np.asarray(payload["vertex_areas_vor"], dtype=float)
    weights = np.asarray(payload["weights"], dtype=float)
    tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
    tri_area = payload["tri_area"]
    if tri_rows.size == 0:
        return 0.0

    transport_model = _resolve_transport_model(
        global_params.get("tilt_transport_model", "ambient_v1")
        if global_params is not None
        else "ambient_v1"
    )
    g0 = payload["g0"]
    g1 = payload["g1"]
    g2 = payload["g2"]
    if g0 is not None and g1 is not None and g2 is not None:
        div_tri = compute_divergence_from_basis(
            mesh=mesh,
            tilts=tilts,
            tri_rows=tri_rows,
            g0=np.asarray(g0, dtype=float),
            g1=np.asarray(g1, dtype=float),
            g2=np.asarray(g2, dtype=float),
            positions=positions,
            transport_model=transport_model,
        )
    else:
        div_tri, _, _, _, _ = p1_triangle_divergence(
            mesh=mesh,
            positions=positions,
            tilts=tilts,
            tri_rows=tri_rows,
            transport_model=transport_model,
        )
    if tri_area is None:
        tri_area = mesh.p1_triangle_shape_gradient_cache(positions)[0]
    tri_area = np.asarray(tri_area, dtype=float)
    div_term = float(div_sign) * div_tri
    div_term = _apply_inner_divergence_update_mode(
        mesh,
        global_params,
        positions=positions,
        tri_rows=tri_rows,
        cache_tag=cache_tag,
        div_term=div_term,
    )
    div_eval_tri, _, _ = _inner_recovered_divergence(
        cache_tag=cache_tag,
        tri_rows=tri_rows,
        tri_area=tri_area,
        div_tri=div_term,
        n_vertices=len(mesh.vertex_ids),
        scratch_tag=f"btl_{cache_tag}",
    )

    _, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        compute_vertex_areas=False,
    )
    static_payload = _leaflet_static_tilt_payload(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
        k_vecs=k_vecs,
        vertex_areas_vor=vertex_areas_vor,
        tri_rows=tri_rows,
        kappa_key=kappa_key,
        cache_tag=cache_tag,
    )
    base_tri = np.asarray(static_payload["base_tri"], dtype=float)
    term_tri = base_tri + div_eval_tri[:, None]
    va_eff = np.stack([va0_eff, va1_eff, va2_eff], axis=1)
    kappa_tri = np.asarray(static_payload["kappa_tri"], dtype=float)

    return float(0.5 * np.sum(kappa_tri * term_tri**2 * va_eff))


def _finite_difference_gradient_shape_leaflet(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts: np.ndarray,
    kappa_key: str,
    cache_tag: str,
    div_sign: float,
    eps: float,
) -> np.ndarray:
    """Energy-consistent shape gradient by central differences (slow)."""
    grad = np.zeros_like(positions)
    base = positions.copy()
    for row, vid in enumerate(mesh.vertex_ids):
        if getattr(mesh.vertices[int(vid)], "fixed", False):
            continue
        for d in range(3):
            pos_plus = base.copy()
            pos_minus = base.copy()
            pos_plus[row, d] += eps
            pos_minus[row, d] -= eps
            e_plus = _total_energy_leaflet(
                mesh,
                global_params,
                positions=pos_plus,
                index_map=index_map,
                tilts=tilts,
                kappa_key=kappa_key,
                cache_tag=cache_tag,
                div_sign=div_sign,
            )
            e_minus = _total_energy_leaflet(
                mesh,
                global_params,
                positions=pos_minus,
                index_map=index_map,
                tilts=tilts,
                kappa_key=kappa_key,
                cache_tag=cache_tag,
                div_sign=div_sign,
            )
            grad[row, d] = (e_plus - e_minus) / (2.0 * eps)
    return grad


def compute_energy_and_gradient_array_leaflet(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray | None,
    ctx=None,
    tilts: np.ndarray,
    tilt_grad_arr: np.ndarray | None,
    kappa_key: str,
    cache_tag: str,
    div_sign: float = 1.0,
) -> float:
    """Compute coupled bending+tilt energy and accumulate gradients."""
    _ = param_resolver
    payload = _leaflet_triangle_payload(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
        cache_tag=cache_tag,
        ctx=ctx,
    )
    k_vecs = np.asarray(payload["k_vecs"], dtype=float)
    vertex_areas_vor = np.asarray(payload["vertex_areas_vor"], dtype=float)
    tri_rows_full = np.asarray(payload["tri_rows_full"], dtype=np.int32)
    weights = np.asarray(payload["weights"], dtype=float)
    tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
    tri_keep = np.asarray(payload["tri_keep"], dtype=bool)
    tri_area = payload["tri_area"]

    if tri_rows_full.size == 0 or tri_rows.size == 0:
        return 0.0

    tilts = np.asarray(tilts, dtype=float)
    if tilts.shape != (len(mesh.vertex_ids), 3):
        raise ValueError("tilts must have shape (N_vertices, 3)")

    transport_model = _resolve_transport_model(
        global_params.get("tilt_transport_model", "ambient_v1")
        if global_params is not None
        else "ambient_v1"
    )

    g0_use = payload["g0"]
    g1_use = payload["g1"]
    g2_use = payload["g2"]
    if g0_use is not None and g1_use is not None and g2_use is not None:
        div_tri = compute_divergence_from_basis(
            mesh=mesh,
            tilts=tilts,
            tri_rows=tri_rows,
            g0=np.asarray(g0_use, dtype=float),
            g1=np.asarray(g1_use, dtype=float),
            g2=np.asarray(g2_use, dtype=float),
            positions=positions,
            transport_model=transport_model,
        )
        g0 = np.asarray(g0_use, dtype=float)
        g1 = np.asarray(g1_use, dtype=float)
        g2 = np.asarray(g2_use, dtype=float)
    else:
        div_tri, _, g0, g1, g2 = p1_triangle_divergence(
            mesh=mesh,
            positions=positions,
            tilts=tilts,
            tri_rows=tri_rows,
            transport_model=transport_model,
        )
    if tri_area is None:
        tri_area = mesh.p1_triangle_shape_gradient_cache(positions)[0]
    tri_area = np.asarray(tri_area, dtype=float)
    div_term = float(div_sign) * div_tri
    div_term = _apply_inner_divergence_update_mode(
        mesh,
        global_params,
        positions=positions,
        tri_rows=tri_rows,
        cache_tag=cache_tag,
        div_term=div_term,
    )
    div_eval_tri, _, div_eval_vertex_area = _inner_recovered_divergence(
        cache_tag=cache_tag,
        tri_rows=tri_rows,
        tri_area=tri_area,
        div_tri=div_term,
        n_vertices=len(mesh.vertex_ids),
        ctx=ctx,
        scratch_tag=f"btl_{cache_tag}",
    )

    vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token=f"bending_tilt_leaflet_{cache_tag}",
        compute_vertex_areas=grad_arr is not None,
    )
    if grad_arr is None:
        static_payload = _leaflet_static_tilt_payload(
            mesh,
            global_params,
            positions=positions,
            index_map=index_map,
            k_vecs=k_vecs,
            vertex_areas_vor=vertex_areas_vor,
            tri_rows=tri_rows,
            kappa_key=kappa_key,
            cache_tag=cache_tag,
        )
        base_tri = np.asarray(static_payload["base_tri"], dtype=float)
        kappa_tri = np.asarray(static_payload["kappa_tri"], dtype=float)
        if ctx is not None:
            term_tri = ctx.scratch_array(
                f"btl_{cache_tag}_tilt_only_term_tri",
                shape=base_tri.shape,
                dtype=base_tri.dtype,
            )
            np.copyto(term_tri, base_tri)
            term_tri += div_eval_tri[:, None]
        else:
            term_tri = base_tri + div_eval_tri[:, None]
        total_energy = float(
            0.5
            * np.sum(
                (kappa_tri[:, 0] * term_tri[:, 0] ** 2 * va0_eff)
                + (kappa_tri[:, 1] * term_tri[:, 1] ** 2 * va1_eff)
                + (kappa_tri[:, 2] * term_tri[:, 2] ** 2 * va2_eff)
            )
        )

        if tilt_grad_arr is not None:
            tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
            if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
                raise ValueError("tilt_grad_arr must have shape (N_vertices, 3)")

            dE_ddiv_base, mode_stats = _inner_bending_tilt_dE_ddiv(
                mesh=mesh,
                global_params=global_params,
                cache_tag=cache_tag,
                kappa_tri=kappa_tri,
                base_tri=base_tri,
                div_term=div_eval_tri,
                va0_eff=va0_eff,
                va1_eff=va1_eff,
                va2_eff=va2_eff,
            )
            if str(cache_tag) == "in":
                setattr(mesh, "_last_bending_tilt_in_update_mode_stats", mode_stats)
            dE_ddiv = _inner_recovered_divergence_pullback(
                cache_tag=cache_tag,
                tri_rows=tri_rows,
                tri_area=tri_area,
                coeff_div_eval=float(div_sign) * dE_ddiv_base,
                v_area=div_eval_vertex_area,
                ctx=ctx,
                scratch_tag=f"btl_{cache_tag}",
            )
            factor = dE_ddiv[:, None]
            _accumulate_leaflet_tilt_gradient(
                tilt_grad_arr,
                tri_rows,
                factor,
                g0,
                g1,
                g2,
                ctx=ctx,
                scratch_tag=f"btl_{cache_tag}_tilt_only_scaled",
            )
        return float(total_energy)

    safe_areas_vor = np.maximum(vertex_areas_vor, 1e-12)

    model = _energy_model(global_params)
    if model != "helfrich":
        model = "helfrich"
    kappa_arr, c0_arr = _per_vertex_params_leaflet(
        mesh, global_params, model=model, kappa_key=kappa_key, cache_tag=cache_tag
    )

    k_mag = np.linalg.norm(k_vecs, axis=1)
    H_vor = k_mag / (2.0 * safe_areas_vor)

    is_interior = _interior_mask_leaflet(
        mesh, global_params, cache_tag=cache_tag, index_map=index_map
    )

    base_term = (2.0 * H_vor) - c0_arr
    base_term[~is_interior] = 0.0
    presets = _assume_J0_presets(global_params, cache_tag=cache_tag)
    if presets:
        radius_max = _assume_J0_radius_max(global_params, cache_tag=cache_tag)
        center_xy = _assume_J0_center_xy(global_params)
        rows = _collect_preset_rows(
            mesh,
            presets=presets,
            cache_tag=cache_tag,
            index_map=index_map,
            radius_max=radius_max,
            center_xy=center_xy,
        )
        if rows.size:
            base_term[rows] = 0.0
    region_rows = _base_term_region_zero_rows(
        mesh,
        global_params,
        cache_tag=cache_tag,
        index_map=index_map,
    )
    if region_rows.size:
        base_term[region_rows] = 0.0

    base_tri = base_term[tri_rows]
    term_tri = base_tri + div_eval_tri[:, None]
    kappa_tri = kappa_arr[tri_rows]
    total_energy = float(
        0.5
        * np.sum(
            (kappa_tri[:, 0] * term_tri[:, 0] ** 2 * va0_eff)
            + (kappa_tri[:, 1] * term_tri[:, 1] ** 2 * va1_eff)
            + (kappa_tri[:, 2] * term_tri[:, 2] ** 2 * va2_eff)
        )
    )

    if ctx is not None:
        ratio = ctx.scratch_array(
            f"btl_{cache_tag}_ratio",
            shape=vertex_areas_eff.shape,
            dtype=vertex_areas_eff.dtype,
        )
    else:
        ratio = np.zeros_like(vertex_areas_eff)
    mask_vor = safe_areas_vor > 1e-15
    ratio[mask_vor] = vertex_areas_eff[mask_vor] / safe_areas_vor[mask_vor]

    if ctx is not None:
        div_eff_num = ctx.scratch_array(
            f"btl_{cache_tag}_div_eff_num",
            shape=base_term.shape,
            dtype=base_term.dtype,
        )
    else:
        div_eff_num = np.zeros_like(base_term)
    div_eff_num = scatter_triangle_scalar_to_vertices(
        tri_rows=tri_rows,
        w0=va0_eff * div_term,
        w1=va1_eff * div_term,
        w2=va2_eff * div_term,
        n_vertices=base_term.shape[0],
        out=div_eff_num,
    )
    if ctx is not None:
        div_eff = ctx.scratch_array(
            f"btl_{cache_tag}_div_eff",
            shape=base_term.shape,
            dtype=base_term.dtype,
        )
    else:
        div_eff = np.zeros_like(base_term)
    mask_eff = vertex_areas_eff > 1e-20
    div_eff[mask_eff] = div_eff_num[mask_eff] / vertex_areas_eff[mask_eff]

    term = base_term + div_eff
    term[~is_interior] = 0.0

    mode = _gradient_mode(global_params)
    normals = _vertex_normals(mesh, positions, tri_rows)
    if ctx is not None:
        K_dir = ctx.scratch_array(
            f"btl_{cache_tag}_K_dir", shape=k_vecs.shape, dtype=k_vecs.dtype
        )
    else:
        K_dir = np.zeros_like(k_vecs)
    mask_k = k_mag > 1e-15
    K_dir[mask_k] = k_vecs[mask_k] / k_mag[mask_k][:, None]
    K_dir[~mask_k] = normals[~mask_k]

    scale_K = (kappa_arr * term * ratio).astype(float, copy=False)
    if ctx is not None:
        factor_K_vec = ctx.scratch_array(
            f"btl_{cache_tag}_factor_K_vec", shape=K_dir.shape, dtype=K_dir.dtype
        )
    else:
        factor_K_vec = np.empty_like(K_dir, order="F")
    np.multiply(K_dir, scale_K[:, None], out=factor_K_vec)

    fA_eff = 0.5 * kappa_arr * term**2
    fA_vor = -2.0 * kappa_arr * term * ratio * H_vor

    if mode == "finite_difference":  # pragma: no cover - slow debugging path
        eps = float(global_params.get("bending_fd_eps", 1e-6))
        grad_arr[:] += _finite_difference_gradient_shape_leaflet(
            mesh,
            global_params,
            positions=positions,
            index_map=index_map,
            tilts=tilts,
            kappa_key=kappa_key,
            cache_tag=cache_tag,
            div_sign=div_sign,
            eps=eps,
        )
    elif mode == "approx":
        grad_arr[:] -= _apply_beltrami_laplacian(weights, tri_rows, factor_K_vec)
        grad_arr[~is_interior] = 0.0
    else:
        # --- Analytic gradient backpropagation (copied from bending.py) ---
        v0_idxs, v1_idxs, v2_idxs = tri_rows[:, 0], tri_rows[:, 1], tri_rows[:, 2]
        v0, v1, v2 = positions[v0_idxs], positions[v1_idxs], positions[v2_idxs]
        e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0
        c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]

        cached = _cached_cotan_gradients(
            mesh,
            positions=positions,
            tri_rows=tri_rows_full if tri_keep.size else tri_rows,
        )
        if cached is not None and tri_keep.size:
            cached = tuple(arr[tri_keep] for arr in cached)

        # Term 1: Variation assuming cotans constant (L constant)
        # factor_K_vec already zeroed at boundaries
        grad_linear = -_apply_beltrami_laplacian(weights, tri_rows, factor_K_vec)

        # Term 2: Variation of L (cotangents)
        fK = factor_K_vec
        dE_dc0 = -0.5 * np.einsum("ij,ij->i", fK[v1_idxs] - fK[v2_idxs], v1 - v2)
        dE_dc1 = -0.5 * np.einsum("ij,ij->i", fK[v2_idxs] - fK[v0_idxs], v2 - v0)
        dE_dc2 = -0.5 * np.einsum("ij,ij->i", fK[v0_idxs] - fK[v1_idxs], v0 - v1)

        if cached is not None:
            (
                g_c0_u,
                g_c0_v,
                g_c1_u,
                g_c1_v,
                g_c2_u,
                g_c2_v,
                gc0u,
                gc0v,
                gc1u,
                gc1v,
                gc2u,
                gc2v,
            ) = cached
        else:
            u0, v0_vec = v1 - v0, v2 - v0
            g_c0_u, g_c0_v = _grad_cotan(u0, v0_vec)
            u1, v1_vec = v2 - v1, v0 - v1
            g_c1_u, g_c1_v = _grad_cotan(u1, v1_vec)
            u2, v2_vec = v0 - v2, v1 - v2
            g_c2_u, g_c2_v = _grad_cotan(u2, v2_vec)

        if ctx is not None:
            grad_cot = ctx.scratch_array(
                f"btl_{cache_tag}_grad_cot",
                shape=positions.shape,
                dtype=positions.dtype,
            )
        else:
            grad_cot = np.zeros_like(positions)
        val0, val1, val2 = dE_dc0[:, None], dE_dc1[:, None], dE_dc2[:, None]
        np.add.at(grad_cot, v1_idxs, val0 * g_c0_u)
        np.add.at(grad_cot, v2_idxs, val0 * g_c0_v)
        np.add.at(grad_cot, v0_idxs, val0 * -(g_c0_u + g_c0_v))
        np.add.at(grad_cot, v2_idxs, val1 * g_c1_u)
        np.add.at(grad_cot, v0_idxs, val1 * g_c1_v)
        np.add.at(grad_cot, v1_idxs, val1 * -(g_c1_u + g_c1_v))
        np.add.at(grad_cot, v0_idxs, val2 * g_c2_u)
        np.add.at(grad_cot, v1_idxs, val2 * g_c2_v)
        np.add.at(grad_cot, v2_idxs, val2 * -(g_c2_u + g_c2_v))

        # --- Area Gradients (Step 2: Propagate area reassignment) ---

        tri_is_int = is_interior[tri_rows]
        interior_counts = np.sum(tri_is_int, axis=1)

        tri_fA_eff = fA_eff[tri_rows]
        sum_fA_eff_int = np.sum(tri_fA_eff * tri_is_int, axis=1)

        if ctx is not None:
            avg_fA_eff = ctx.scratch_array(
                f"btl_{cache_tag}_avg_fA_eff",
                shape=(len(tri_rows),),
                dtype=float,
            )
        else:
            avg_fA_eff = np.zeros(len(tri_rows), dtype=float)
        mask_has_int = interior_counts > 0
        avg_fA_eff[mask_has_int] = (
            sum_fA_eff_int[mask_has_int] / interior_counts[mask_has_int]
        )

        C_eff = np.where(tri_is_int, tri_fA_eff, avg_fA_eff[:, None])
        tri_fA_vor = fA_vor[tri_rows]
        C = C_eff + tri_fA_vor

        if ctx is not None:
            grad_area = ctx.scratch_array(
                f"btl_{cache_tag}_grad_area",
                shape=positions.shape,
                dtype=positions.dtype,
            )
        else:
            grad_area = np.zeros_like(positions)

        is_obtuse = (c0 < 0) | (c1 < 0) | (c2 < 0)
        m_std = ~is_obtuse

        if np.any(m_std):
            c0s, c1s, c2s = c0[m_std], c1[m_std], c2[m_std]
            C0s, C1s, C2s = C[m_std, 0], C[m_std, 1], C[m_std, 2]
            e0s, e1s, e2s = e0[m_std], e1[m_std], e2[m_std]
            v0s, v1s, v2s = v0_idxs[m_std], v1_idxs[m_std], v2_idxs[m_std]

            coeff = 0.25 * c1s * C0s
            np.add.at(grad_area, v0s, coeff[:, None] * e1s)
            np.add.at(grad_area, v2s, -coeff[:, None] * e1s)
            coeff = 0.25 * c2s * C0s
            np.add.at(grad_area, v1s, coeff[:, None] * e2s)
            np.add.at(grad_area, v0s, -coeff[:, None] * e2s)
            coeff = 0.25 * c2s * C1s
            np.add.at(grad_area, v1s, coeff[:, None] * e2s)
            np.add.at(grad_area, v0s, -coeff[:, None] * e2s)
            coeff = 0.25 * c0s * C1s
            np.add.at(grad_area, v2s, coeff[:, None] * e0s)
            np.add.at(grad_area, v1s, -coeff[:, None] * e0s)
            coeff = 0.25 * c0s * C2s
            np.add.at(grad_area, v2s, coeff[:, None] * e0s)
            np.add.at(grad_area, v1s, -coeff[:, None] * e0s)
            coeff = 0.25 * c1s * C2s
            np.add.at(grad_area, v0s, coeff[:, None] * e1s)
            np.add.at(grad_area, v2s, -coeff[:, None] * e1s)

            l0sq = np.einsum("ij,ij->i", e0s, e0s)
            l1sq = np.einsum("ij,ij->i", e1s, e1s)
            l2sq = np.einsum("ij,ij->i", e2s, e2s)

            coeff_c0 = 0.125 * l0sq * (C1s + C2s)
            coeff_c1 = 0.125 * l1sq * (C0s + C2s)
            coeff_c2 = 0.125 * l2sq * (C0s + C1s)

            if cached is None:
                gc0u, gc0v = _grad_cotan(e2s, -e1s)
                gc1u, gc1v = _grad_cotan(e0s, -e2s)
                gc2u, gc2v = _grad_cotan(e1s, -e0s)
            else:
                gc0u, gc0v = gc0u[m_std], gc0v[m_std]
                gc1u, gc1v = gc1u[m_std], gc1v[m_std]
                gc2u, gc2v = gc2u[m_std], gc2v[m_std]

            v_c0, v_c1, v_c2 = (
                coeff_c0[:, None],
                coeff_c1[:, None],
                coeff_c2[:, None],
            )
            np.add.at(grad_area, v1s, v_c0 * gc0u)
            np.add.at(grad_area, v2s, v_c0 * gc0v)
            np.add.at(grad_area, v0s, v_c0 * -(gc0u + gc0v))
            np.add.at(grad_area, v2s, v_c1 * gc1u)
            np.add.at(grad_area, v0s, v_c1 * gc1v)
            np.add.at(grad_area, v1s, v_c1 * -(gc1u + gc1v))
            np.add.at(grad_area, v0s, v_c2 * gc2u)
            np.add.at(grad_area, v1s, v_c2 * gc2v)
            np.add.at(grad_area, v2s, v_c2 * -(gc2u + gc2v))

        if np.any(is_obtuse):
            for i, m_sub in enumerate([(c0 < 0), (c1 < 0), (c2 < 0)]):
                m_do = m_sub & is_obtuse
                if np.any(m_do):
                    v0o, v1o, v2o = v0_idxs[m_do], v1_idxs[m_do], v2_idxs[m_do]
                    gT_u, gT_v = grad_triangle_area(
                        positions[v1o] - positions[v0o],
                        positions[v2o] - positions[v0o],
                    )

                    C0o, C1o, C2o = C[m_do, 0], C[m_do, 1], C[m_do, 2]
                    if i == 0:
                        factor = (0.5 * C0o + 0.25 * C1o + 0.25 * C2o)[:, None]
                    elif i == 1:
                        factor = (0.5 * C1o + 0.25 * C0o + 0.25 * C2o)[:, None]
                    else:
                        factor = (0.5 * C2o + 0.25 * C0o + 0.25 * C1o)[:, None]

                    np.add.at(grad_area, v1o, factor * gT_u)
                    np.add.at(grad_area, v2o, factor * gT_v)
                    np.add.at(grad_area, v0o, factor * -(gT_u + gT_v))

        grad_arr[:] += grad_linear
        grad_arr[:] += grad_cot
        grad_arr[:] += grad_area

    if tilt_grad_arr is not None:
        tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
        if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_grad_arr must have shape (N_vertices, 3)")

        dE_ddiv_base, mode_stats = _inner_bending_tilt_dE_ddiv(
            mesh=mesh,
            global_params=global_params,
            cache_tag=cache_tag,
            kappa_tri=kappa_tri,
            base_tri=base_tri,
            div_term=div_eval_tri,
            va0_eff=va0_eff,
            va1_eff=va1_eff,
            va2_eff=va2_eff,
        )
        if str(cache_tag) == "in":
            setattr(mesh, "_last_bending_tilt_in_update_mode_stats", mode_stats)
        dE_ddiv = _inner_recovered_divergence_pullback(
            cache_tag=cache_tag,
            tri_rows=tri_rows,
            tri_area=tri_area,
            coeff_div_eval=float(div_sign) * dE_ddiv_base,
            v_area=div_eval_vertex_area,
            ctx=ctx,
            scratch_tag=f"btl_{cache_tag}",
        )
        factor = dE_ddiv[:, None]

        np.add.at(tilt_grad_arr, tri_rows[:, 0], factor * g0)
        np.add.at(tilt_grad_arr, tri_rows[:, 1], factor * g1)
        np.add.at(tilt_grad_arr, tri_rows[:, 2], factor * g2)

    return total_energy


def compute_energy_and_gradient_leaflet(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    tilts: np.ndarray,
    kappa_key: str,
    cache_tag: str,
    div_sign: float = 1.0,
    compute_gradient: bool = True,
) -> (
    tuple[float, Dict[int, np.ndarray]]
    | tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Dict-based wrapper returning (E, shape_grad[, tilt_grad])."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions) if compute_gradient else None
    energy = compute_energy_and_gradient_array_leaflet(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts=tilts,
        tilt_grad_arr=tilt_grad_arr,
        kappa_key=kappa_key,
        cache_tag=cache_tag,
        div_sign=div_sign,
    )
    if not compute_gradient:
        return float(energy), {}

    shape_grad = {
        int(vid): grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }
    tilt_grad = {
        int(vid): tilt_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if tilt_grad_arr is not None and np.any(tilt_grad_arr[row])
    }
    return float(energy), shape_grad, tilt_grad


__all__ = [
    "compute_energy_and_gradient_array_leaflet",
    "compute_energy_and_gradient_leaflet",
]
