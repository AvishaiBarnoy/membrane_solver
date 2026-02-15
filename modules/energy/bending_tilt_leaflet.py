"""Shared helpers for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.bending_derivatives import grad_triangle_area
from geometry.curvature import compute_curvature_data
from geometry.entities import Mesh
from geometry.tilt_operators import (
    p1_triangle_divergence,
    p1_triangle_divergence_from_shape_gradients,
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


def _collect_preset_rows(
    mesh: Mesh,
    *,
    presets: tuple[str, ...],
    cache_tag: str,
    index_map: Dict[int, int],
) -> np.ndarray:
    """Return vertex-row indices whose ``preset`` option is in ``presets``."""
    if not presets:
        return np.zeros(0, dtype=int)
    cache_key = (mesh._vertex_ids_version, presets)
    cache_attr = f"_bending_tilt_assume_J0_rows_cache_{cache_tag}"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["rows"]
    presets_set = set(presets)
    present: set[str] = set()

    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        preset = str(opts.get("preset") or "")
        present.add(preset)
        if preset in presets_set:
            row = index_map.get(int(vid))
            if row is not None:
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
    mesh.build_position_cache()
    k_vecs, vertex_areas_vor, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )
    if tri_rows.size == 0:
        return 0.0

    div_tri, _, _, _, _ = p1_triangle_divergence(
        positions=positions, tilts=tilts, tri_rows=tri_rows
    )
    div_term = float(div_sign) * div_tri

    _, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh, positions, tri_rows, weights, index_map
    )
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
        rows = _collect_preset_rows(
            mesh, presets=presets, cache_tag=cache_tag, index_map=index_map
        )
        if rows.size:
            base_term[rows] = 0.0

    term_tri = base_term[tri_rows] + div_term[:, None]
    va_eff = np.stack([va0_eff, va1_eff, va2_eff], axis=1)
    kappa_tri = kappa_arr[tri_rows]

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
    mesh.build_position_cache()
    k_vecs, vertex_areas_vor, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )

    if tri_rows.size == 0:
        return 0.0

    tri_rows_full = tri_rows
    weights_full = weights
    tri_keep = np.array([], dtype=bool)
    if cache_tag == "out":
        absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet="out")
        tri_keep = leaflet_present_triangle_mask(
            mesh, tri_rows_full, absent_vertex_mask=absent_mask
        )
        if tri_keep.size:
            tri_rows = tri_rows_full[tri_keep]
            weights = weights_full[tri_keep]
            if tri_rows.size == 0:
                return 0.0

    tilts = np.asarray(tilts, dtype=float)
    if tilts.shape != (len(mesh.vertex_ids), 3):
        raise ValueError("tilts must have shape (N_vertices, 3)")

    # Use cached triangle P1 basis gradients when geometry is frozen/cached.
    if ctx is not None:
        area_cache, g0_cache, g1_cache, g2_cache, tri_rows_cache = (
            ctx.geometry.p1_triangle_shape_gradients(mesh, positions)
        )
    else:
        area_cache, g0_cache, g1_cache, g2_cache, tri_rows_cache = (
            mesh.p1_triangle_shape_gradient_cache(positions)
        )
    if tri_rows_cache.size and tri_rows_cache.shape[0] == tri_rows_full.shape[0]:
        g0_use = g0_cache
        g1_use = g1_cache
        g2_use = g2_cache
        if cache_tag == "out" and tri_keep.size:
            g0_use = g0_use[tri_keep]
            g1_use = g1_use[tri_keep]
            g2_use = g2_use[tri_keep]
        div_tri = p1_triangle_divergence_from_shape_gradients(
            tilts=tilts, tri_rows=tri_rows, g0=g0_use, g1=g1_use, g2=g2_use
        )
        # Keep using the standard divergence routine for gradients (g0,g1,g2)
        # which we already have in cached form.
        g0, g1, g2 = g0_use, g1_use, g2_use
        _ = area_cache
    else:
        div_tri, _, g0, g1, g2 = p1_triangle_divergence(
            positions=positions, tilts=tilts, tri_rows=tri_rows
        )
    div_term = float(div_sign) * div_tri

    vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token=f"bending_tilt_leaflet_{cache_tag}",
    )
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

    base_term = (2.0 * H_vor) - c0_arr
    base_term[~is_interior] = 0.0
    presets = _assume_J0_presets(global_params, cache_tag=cache_tag)
    if presets:
        rows = _collect_preset_rows(
            mesh, presets=presets, cache_tag=cache_tag, index_map=index_map
        )
        if rows.size:
            base_term[rows] = 0.0

    term_tri = base_term[tri_rows] + div_term[:, None]
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
        div_eff_num = ctx.scratch_array(
            f"btl_{cache_tag}_div_eff_num",
            shape=base_term.shape,
            dtype=base_term.dtype,
        )
    else:
        div_eff_num = np.zeros_like(base_term)
    np.add.at(div_eff_num, tri_rows[:, 0], va0_eff * div_term)
    np.add.at(div_eff_num, tri_rows[:, 1], va1_eff * div_term)
    np.add.at(div_eff_num, tri_rows[:, 2], va2_eff * div_term)
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

    if grad_arr is None:
        if tilt_grad_arr is not None:
            tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
            if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
                raise ValueError("tilt_grad_arr must have shape (N_vertices, 3)")

            dE_ddiv = float(div_sign) * (
                (kappa_tri[:, 0] * term_tri[:, 0] * va0_eff)
                + (kappa_tri[:, 1] * term_tri[:, 1] * va1_eff)
                + (kappa_tri[:, 2] * term_tri[:, 2] * va2_eff)
            )
            factor = dE_ddiv[:, None]

            np.add.at(tilt_grad_arr, tri_rows[:, 0], factor * g0)
            np.add.at(tilt_grad_arr, tri_rows[:, 1], factor * g1)
            np.add.at(tilt_grad_arr, tri_rows[:, 2], factor * g2)
        return float(total_energy)

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

        dE_ddiv = float(div_sign) * (
            (kappa_tri[:, 0] * term_tri[:, 0] * va0_eff)
            + (kappa_tri[:, 1] * term_tri[:, 1] * va1_eff)
            + (kappa_tri[:, 2] * term_tri[:, 2] * va2_eff)
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
