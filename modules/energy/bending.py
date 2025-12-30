# modules/energy/bending.py

from __future__ import annotations

import logging
import os
from typing import Dict, Literal

import numpy as np

from fortran_kernels.loader import (
    get_bending_grad_cotan_kernel,
    get_bending_laplacian_kernel,
)
from geometry.bending_derivatives import (
    grad_cotan as _grad_cotan_numpy,
)
from geometry.bending_derivatives import (
    grad_triangle_area,
)
from geometry.curvature import compute_curvature_data
from geometry.entities import Mesh

logger = logging.getLogger("membrane_solver")


BendingEnergyModel = Literal["willmore", "helfrich"]
BendingGradientMode = Literal["approx", "finite_difference", "analytic"]


def _energy_model(global_params) -> BendingEnergyModel:
    model = str(global_params.get("bending_energy_model", "helfrich") or "helfrich")
    model = model.lower().strip()
    return "helfrich" if model == "helfrich" else "willmore"


def _gradient_mode(global_params) -> BendingGradientMode:
    mode = str(global_params.get("bending_gradient_mode", "analytic") or "analytic")
    mode = mode.lower().strip()
    if mode in {"fd", "finite_difference"}:
        return "finite_difference"
    if mode == "analytic":
        return "analytic"
    return "approx"


def _spontaneous_curvature(global_params) -> float:
    val = global_params.get("spontaneous_curvature")
    if val is None:
        val = global_params.get("intrinsic_curvature", 0.0)
    return float(val or 0.0)


def _per_vertex_params(
    mesh: Mesh,
    global_params,
    *,
    model: BendingEnergyModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-vertex (kappa, c0) arrays in vertex-row order.

    Allows local overrides via `vertex.options`:
    - `bending_modulus`
    - `spontaneous_curvature` (alias: `intrinsic_curvature`)
    """
    mesh.build_position_cache()

    n = len(mesh.vertex_ids)
    kappa_default = float(global_params.get("bending_modulus", 0.0) or 0.0)
    if model == "helfrich":
        c0_default = _spontaneous_curvature(global_params)
    else:
        c0_default = 0.0

    cache_key = (
        mesh._vertex_ids_version,
        model,
        float(kappa_default),
        float(c0_default),
    )
    cached = getattr(mesh, "_bending_vertex_param_cache", None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["kappa"], cached["c0"]

    kappa = np.full(n, kappa_default, dtype=float)
    c0 = np.full(n, c0_default, dtype=float)

    override_rows_k: list[int] = []
    override_vals_k: list[float] = []
    override_rows_c0: list[int] = []
    override_vals_c0: list[float] = []

    # One-time scan over vertices to find overrides; results are cached and
    # reused across minimization iterations (no per-step Python loops).
    for vid, vertex in mesh.vertices.items():
        row = mesh.vertex_index_to_row.get(int(vid))
        if row is None:
            continue
        opts = getattr(vertex, "options", None) or {}
        if "bending_modulus" in opts:
            try:
                override_rows_k.append(row)
                override_vals_k.append(float(opts["bending_modulus"]))
            except (TypeError, ValueError):
                pass
        if model == "helfrich":
            if "spontaneous_curvature" in opts:
                try:
                    override_rows_c0.append(row)
                    override_vals_c0.append(float(opts["spontaneous_curvature"]))
                except (TypeError, ValueError):
                    pass
            elif "intrinsic_curvature" in opts:
                try:
                    override_rows_c0.append(row)
                    override_vals_c0.append(float(opts["intrinsic_curvature"]))
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

    mesh._bending_vertex_param_cache = {"key": cache_key, "kappa": kappa, "c0": c0}
    return kappa, c0


def _vertex_normals(
    mesh: Mesh, positions: np.ndarray, tri_rows: np.ndarray
) -> np.ndarray:
    normals = np.zeros((len(mesh.vertex_ids), 3), dtype=float)
    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]
    tri_normals = np.cross(v1 - v0, v2 - v0)
    np.add.at(normals, tri_rows[:, 0], tri_normals)
    np.add.at(normals, tri_rows[:, 1], tri_normals)
    np.add.at(normals, tri_rows[:, 2], tri_normals)
    nrm = np.linalg.norm(normals, axis=1)
    mask = nrm > 1e-15
    normals[mask] /= nrm[mask, None]
    return normals


def _mean_curvature_vectors(
    mesh: Mesh, positions: np.ndarray, index_map: Dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Hn, |Hn|, A, weights, tri_rows).

    We use the magnitude ``|Hn|`` for scalar curvature to avoid dependence on
    global facet orientation, which is not guaranteed to be consistent in all
    generated meshes.
    """
    mesh.build_position_cache()
    k_vecs, vertex_areas, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )
    if tri_rows.size == 0:
        n = len(mesh.vertex_ids)
        return (
            np.zeros((n, 3), dtype=float),
            np.zeros(n, dtype=float),
            np.zeros(n, dtype=float),
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=int),
        )

    safe_areas = np.maximum(vertex_areas, 1e-12)
    h_vecs = k_vecs / (2.0 * safe_areas[:, None])  # Hn
    h_mag = np.linalg.norm(h_vecs, axis=1)
    return h_vecs, h_mag, vertex_areas, weights, tri_rows


def compute_total_energy(
    mesh: Mesh, global_params, positions: np.ndarray, index_map: Dict[int, int]
) -> float:
    model = _energy_model(global_params)
    kappa_arr, c0_arr = _per_vertex_params(mesh, global_params, model=model)
    if float(np.max(kappa_arr)) == 0.0:
        return 0.0

    _, H, A, _, tri_rows = _mean_curvature_vectors(mesh, positions, index_map)
    if tri_rows.size == 0:
        return 0.0

    if model == "helfrich":
        density = 0.5 * (2.0 * H - c0_arr) ** 2
    else:
        density = H**2

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        density[boundary_rows] = 0.0

    return float(np.sum(kappa_arr * density * A))


def compute_energy_array(mesh, global_params, positions, index_map) -> np.ndarray:
    """Compute energy contribution per vertex. Returns (N_verts,) array."""
    model = _energy_model(global_params)
    kappa_arr, c0_arr = _per_vertex_params(mesh, global_params, model=model)
    if float(np.max(kappa_arr)) == 0.0:
        return np.zeros(len(mesh.vertex_ids), dtype=float)

    _, H, A, _, tri_rows = _mean_curvature_vectors(mesh, positions, index_map)
    if tri_rows.size == 0:
        return np.zeros(len(mesh.vertex_ids), dtype=float)

    if model == "helfrich":
        density = 0.5 * (2.0 * H - c0_arr) ** 2
    else:
        density = H**2

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        density[boundary_rows] = 0.0

    return kappa_arr * density * A


def _apply_beltrami_laplacian(
    weights: np.ndarray, tri_rows: np.ndarray, field: np.ndarray
) -> np.ndarray:
    """Apply the discrete Laplace-Beltrami operator (based on cotangent weights)."""
    kernel_spec = get_bending_laplacian_kernel()
    if kernel_spec is not None:
        strict = os.environ.get("MEMBRANE_FORTRAN_STRICT_NOCOPY") in {
            "1",
            "true",
            "TRUE",
        }
        if (
            field.dtype != np.float64
            or weights.dtype != np.float64
            or tri_rows.dtype != np.int32
        ):
            if strict:
                raise TypeError(
                    "Fortran bending kernels require float64 weights/field and int32 tri_rows."
                )
            kernel_spec = None
        elif not (
            weights.flags["F_CONTIGUOUS"]
            and tri_rows.flags["F_CONTIGUOUS"]
            and field.flags["F_CONTIGUOUS"]
        ):
            if strict:
                raise ValueError(
                    "Fortran bending kernels require F-contiguous weights/tri_rows/field (to avoid hidden copies)."
                )
            kernel_spec = None

    if kernel_spec is not None:
        if kernel_spec.expects_transpose:
            # Legacy wrapper (dim,nv) / (3,nf) style.
            try:
                out_t = np.empty_like(field.T, order="F")
                kernel_spec.func(weights.T, tri_rows.T, field.T, out_t, 1)
                return out_t.T
            except TypeError:
                out_t = kernel_spec.func(weights.T, tri_rows.T, field.T, 1)
                return np.asarray(out_t).T

        try:
            out = np.empty_like(field, order="F")
            kernel_spec.func(weights, tri_rows, field, out, 1)
            return out
        except Exception:
            out = kernel_spec.func(weights, tri_rows, field, 1)
            return np.asarray(out)

    c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]
    v0, v1, v2 = tri_rows[:, 0], tri_rows[:, 1], tri_rows[:, 2]
    f0, f1, f2 = field[v0], field[v1], field[v2]
    out = np.zeros_like(field)
    np.add.at(out, v0, 0.5 * (c1[:, None] * (f0 - f2) + c2[:, None] * (f0 - f1)))
    np.add.at(out, v1, 0.5 * (c2[:, None] * (f1 - f0) + c0[:, None] * (f1 - f2)))
    np.add.at(out, v2, 0.5 * (c0[:, None] * (f2 - f1) + c1[:, None] * (f2 - f0)))
    return out


def _grad_cotan(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (grad_u, grad_v) for cotan(u, v), using Fortran if available."""
    kernel_spec = get_bending_grad_cotan_kernel()
    if kernel_spec is None:
        return _grad_cotan_numpy(u, v)

    strict = os.environ.get("MEMBRANE_FORTRAN_STRICT_NOCOPY") in {"1", "true", "TRUE"}
    if u.dtype != np.float64 or v.dtype != np.float64:
        if strict:
            raise TypeError("Fortran bending kernels require float64 u/v.")
        return _grad_cotan_numpy(u, v)
    if not (u.flags["F_CONTIGUOUS"] and v.flags["F_CONTIGUOUS"]):
        if strict:
            raise ValueError(
                "Fortran bending kernels require F-contiguous u/v (to avoid hidden copies)."
            )
        return _grad_cotan_numpy(u, v)

    if kernel_spec.expects_transpose:
        try:
            gu_t = np.zeros_like(u.T, order="F")
            gv_t = np.zeros_like(v.T, order="F")
            kernel_spec.func(u.T, v.T, gu_t, gv_t)
            return gu_t.T, gv_t.T
        except Exception:
            gu_t, gv_t = kernel_spec.func(u.T, v.T)
            return np.asarray(gu_t).T, np.asarray(gv_t).T

    try:
        gu = np.zeros_like(u, order="F")
        gv = np.zeros_like(v, order="F")
        kernel_spec.func(u, v, gu, gv)
        return gu, gv
    except Exception:
        gu, gv = kernel_spec.func(u, v)
        return np.asarray(gu), np.asarray(gv)


def _finite_difference_gradient(
    mesh: Mesh,
    global_params,
    positions: np.ndarray,
    index_map: Dict[int, int],
    *,
    eps: float,
) -> np.ndarray:
    """Energy-consistent gradient by central differences (slow)."""
    grad = np.zeros_like(positions)
    base = positions.copy()
    mesh.build_position_cache()
    for row, vid in enumerate(mesh.vertex_ids):
        if getattr(mesh.vertices[int(vid)], "fixed", False):
            continue
        for d in range(3):
            pos_plus = base.copy()
            pos_minus = base.copy()
            pos_plus[row, d] += eps
            pos_minus[row, d] -= eps
            e_plus = compute_total_energy(mesh, global_params, pos_plus, index_map)
            e_minus = compute_total_energy(mesh, global_params, pos_minus, index_map)
            grad[row, d] = (e_plus - e_minus) / (2.0 * eps)

    boundary_vids = mesh.boundary_vertex_ids
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        grad[boundary_rows] = 0.0
    return grad


def _laplacian_on_vertex_field(
    weights: np.ndarray, tri_rows: np.ndarray, field: np.ndarray
) -> np.ndarray:
    c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]
    v0, v1, v2 = tri_rows[:, 0], tri_rows[:, 1], tri_rows[:, 2]
    f0, f1, f2 = field[v0], field[v1], field[v2]
    out = np.zeros_like(field)
    np.add.at(out, v0, 0.5 * (c1[:, None] * (f0 - f2) + c2[:, None] * (f0 - f1)))
    np.add.at(out, v1, 0.5 * (c2[:, None] * (f1 - f0) + c0[:, None] * (f1 - f2)))
    np.add.at(out, v2, 0.5 * (c0[:, None] * (f2 - f1) + c1[:, None] * (f2 - f0)))
    return out


def _compute_effective_areas(
    mesh: Mesh,
    positions: np.ndarray,
    tri_rows: np.ndarray,
    weights: np.ndarray,
    index_map: Dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute effective vertex areas by redistributing boundary vertex contributions.

    Returns (vertex_areas_eff, va0, va1, va2).
    """
    n_verts = len(mesh.vertex_ids)
    if tri_rows.size == 0:
        return (
            np.zeros(n_verts),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
        )

    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    l0_sq = np.einsum("ij,ij->i", e0, e0)
    l1_sq = np.einsum("ij,ij->i", e1, e1)
    l2_sq = np.einsum("ij,ij->i", e2, e2)

    c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]

    # Triangle area (doubled)
    cross = np.cross(e1, e2)
    area_doubled = np.linalg.norm(cross, axis=1)
    area_doubled = np.maximum(area_doubled, 1e-12)
    tri_areas = 0.5 * area_doubled

    # Check for obtuse angles
    is_obtuse_v0 = c0 < 0
    is_obtuse_v1 = c1 < 0
    is_obtuse_v2 = c2 < 0
    any_obtuse = is_obtuse_v0 | is_obtuse_v1 | is_obtuse_v2

    # Standard Voronoi contributions
    va0 = np.where(~any_obtuse, (l1_sq * c1 + l2_sq * c2) / 8.0, 0.0)
    va1 = np.where(~any_obtuse, (l2_sq * c2 + l0_sq * c0) / 8.0, 0.0)
    va2 = np.where(~any_obtuse, (l0_sq * c0 + l1_sq * c1) / 8.0, 0.0)

    # Obtuse contributions
    va0 = np.where(is_obtuse_v0, tri_areas / 2.0, va0)
    va0 = np.where(is_obtuse_v1 | is_obtuse_v2, tri_areas / 4.0, va0)
    va1 = np.where(is_obtuse_v1, tri_areas / 2.0, va1)
    va1 = np.where(is_obtuse_v0 | is_obtuse_v2, tri_areas / 4.0, va1)
    va2 = np.where(is_obtuse_v2, tri_areas / 2.0, va2)
    va2 = np.where(is_obtuse_v0 | is_obtuse_v1, tri_areas / 4.0, va2)

    # Redistribution logic
    boundary_rows = np.array(
        [
            index_map[vid]
            for vid in (mesh.boundary_vertex_ids or [])
            if vid in index_map
        ],
        dtype=int,
    )
    is_boundary = np.zeros(n_verts, dtype=bool)
    if boundary_rows.size:
        is_boundary[boundary_rows] = True

    tri_is_b = is_boundary[tri_rows]  # (nf, 3)
    interior_mask = ~tri_is_b
    interior_counts = np.sum(interior_mask, axis=1)

    va_eff = np.stack([va0, va1, va2], axis=1)

    # Find triangles with 1 or 2 interior vertices
    mask_has_interior = interior_counts > 0
    mask_some_boundary = np.any(tri_is_b, axis=1)
    to_redistribute = mask_has_interior & mask_some_boundary

    if np.any(to_redistribute):
        # Area from boundary vertices in these triangles
        b_area_sums = np.sum(va_eff * tri_is_b, axis=1)
        # Distribute equally to interior vertices (per triangle)
        extra_per_int = np.zeros_like(b_area_sums)
        extra_per_int[to_redistribute] = (
            b_area_sums[to_redistribute] / interior_counts[to_redistribute]
        )

        # For redistributed triangles:
        # - boundary vertex area contributions are removed (set to 0)
        # - each interior vertex receives its share of the boundary area
        va_eff[to_redistribute] = (
            va_eff[to_redistribute] * interior_mask[to_redistribute]
            + interior_mask[to_redistribute] * extra_per_int[to_redistribute, None]
        )

    vertex_areas_eff = np.zeros(n_verts, dtype=float)
    np.add.at(vertex_areas_eff, tri_rows[:, 0], va_eff[:, 0])
    np.add.at(vertex_areas_eff, tri_rows[:, 1], va_eff[:, 1])
    np.add.at(vertex_areas_eff, tri_rows[:, 2], va_eff[:, 2])

    return vertex_areas_eff, va_eff[:, 0], va_eff[:, 1], va_eff[:, 2]


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray,
) -> float:
    """Compute bending energy and accumulate analytic or FD gradient."""
    mesh.build_position_cache()
    k_vecs, _, weights, tri_rows = compute_curvature_data(mesh, positions, index_map)

    if tri_rows.size == 0:
        return 0.0

    # Step 1: Boundary area reassignment
    vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh, positions, tri_rows, weights, index_map
    )
    safe_areas = np.maximum(vertex_areas_eff, 1e-12)

    model = _energy_model(global_params)
    kappa_arr, c0_arr = _per_vertex_params(mesh, global_params, model=model)

    k_mag = np.linalg.norm(k_vecs, axis=1)

    # Only interior vertices contribute to energy
    boundary_vids = mesh.boundary_vertex_ids
    is_interior = np.ones(len(mesh.vertex_ids), dtype=bool)
    if boundary_vids:
        boundary_rows = [index_map[vid] for vid in boundary_vids if vid in index_map]
        is_interior[boundary_rows] = False

    if model == "helfrich":
        # E_i = 0.5 * kappa * ( |K|/A - c0 )^2 * A
        term = (k_mag / safe_areas) - c0_arr
        # Zero out boundary curvature terms
        term[~is_interior] = 0.0
        total_energy = float(0.5 * np.sum(kappa_arr * term**2 * safe_areas))

        # factor_K = kappa * term * (K / |K|)
        # factor_A = 0.5 * kappa * term^2 - kappa * term * (k_mag / safe_areas)
        normals = _vertex_normals(mesh, positions, tri_rows)
        K_dir = np.zeros_like(k_vecs)
        mask = k_mag > 1e-15
        K_dir[mask] = k_vecs[mask] / k_mag[mask][:, None]
        K_dir[~mask] = normals[~mask]

        scale_K = (kappa_arr * term).astype(float, copy=False)
        factor_K_vec = np.empty_like(K_dir, order="F")
        np.multiply(K_dir, scale_K[:, None], out=factor_K_vec)
        factor_A = 0.5 * kappa_arr * term**2 - kappa_arr * term * (k_mag / safe_areas)

    else:
        # Willmore: E_i = kappa * |K|^2 / (4A)
        # Zero out boundary curvature terms
        k_mag_eff = k_mag.copy()
        k_mag_eff[~is_interior] = 0.0
        total_energy = float(np.sum(kappa_arr * (k_mag_eff**2 / (4.0 * safe_areas))))

        scale_K = (kappa_arr / (2.0 * safe_areas)).astype(float, copy=False)
        factor_K_vec = np.empty_like(k_vecs, order="F")
        # Ensure k_vec contribution from boundaries is zero
        k_vecs_eff = k_vecs.copy()
        k_vecs_eff[~is_interior] = 0.0
        np.multiply(k_vecs_eff, scale_K[:, None], out=factor_K_vec)
        factor_A = -kappa_arr * k_mag_eff**2 / (4.0 * safe_areas**2)

    # Step 1: Force on boundary vertices is zeroed.
    # Note: factor_K_vec and factor_A already have 0 for boundary vertices via term/k_mag_eff.

    # Now verify if user wants FD/approx/analytic
    mode = _gradient_mode(global_params)
    if mode == "finite_difference":
        eps = float(global_params.get("bending_fd_eps", 1e-6))
        grad_arr[:] += _finite_difference_gradient(
            mesh, global_params, positions, index_map, eps=eps
        )
        return total_energy

    if mode == "approx":
        h_vecs = k_vecs / (2.0 * safe_areas[:, None])
        # Mask boundary h_vecs
        h_vecs[~is_interior] = 0.0

        if model == "helfrich":
            h_norms = np.linalg.norm(h_vecs, axis=1)
            n_hats = np.zeros_like(h_vecs)
            mask_n = h_norms > 1e-12
            n_hats[mask_n] = h_vecs[mask_n] / h_norms[mask_n][:, None]
            target_h_vecs = (c0_arr / 2.0)[:, None] * n_hats
            diff_h_vecs = h_vecs - target_h_vecs
            # Again, ensure boundary diff is 0
            diff_h_vecs[~is_interior] = 0.0
            field = np.empty_like(diff_h_vecs, order="F")
            np.multiply(diff_h_vecs, kappa_arr[:, None], out=field)
            grad_arr[:] -= _apply_beltrami_laplacian(weights, tri_rows, field)
        else:
            field = np.empty_like(h_vecs, order="F")
            np.multiply(h_vecs, kappa_arr[:, None], out=field)
            grad_arr[:] -= _apply_beltrami_laplacian(weights, tri_rows, field)

        # Zero out boundary forces
        if boundary_vids:
            boundary_rows = [
                index_map[vid] for vid in boundary_vids if vid in index_map
            ]
            grad_arr[boundary_rows] = 0.0
        return total_energy

    # --- Analytic gradient backpropagation ---
    v0_idxs, v1_idxs, v2_idxs = tri_rows[:, 0], tri_rows[:, 1], tri_rows[:, 2]
    v0, v1, v2 = positions[v0_idxs], positions[v1_idxs], positions[v2_idxs]
    e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0
    c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]

    # Term 1: Variation assuming cotans constant (L constant)
    # factor_K_vec already zeroed at boundaries
    grad_linear = -_apply_beltrami_laplacian(weights, tri_rows, factor_K_vec)

    # Term 2: Variation of L (cotangents)
    fK = factor_K_vec
    dE_dc0 = -0.5 * np.einsum("ij,ij->i", fK[v1_idxs] - fK[v2_idxs], v1 - v2)
    dE_dc1 = -0.5 * np.einsum("ij,ij->i", fK[v2_idxs] - fK[v0_idxs], v2 - v0)
    dE_dc2 = -0.5 * np.einsum("ij,ij->i", fK[v0_idxs] - fK[v1_idxs], v0 - v1)

    u0, v0_vec = v1 - v0, v2 - v0
    g_c0_u, g_c0_v = _grad_cotan(u0, v0_vec)
    u1, v1_vec = v2 - v1, v0 - v1
    g_c1_u, g_c1_v = _grad_cotan(u1, v1_vec)
    u2, v2_vec = v0 - v2, v1 - v2
    g_c2_u, g_c2_v = _grad_cotan(u2, v2_vec)

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
    fA = factor_A

    # We define per-triangle-vertex coefficients C.
    # The energy contribution from area variation is sum(C * grad(A_voronoi)).
    # - For interior vertices: C = fA (standard)
    # - For boundary vertices: A_voronoi is redistributed to interior neighbors.
    #   Effectively, A_b contributes to energy with weight average(fA_interior).

    tri_is_int = is_interior[tri_rows]
    interior_counts = np.sum(tri_is_int, axis=1)

    tri_fA = fA[tri_rows]
    sum_fA_int = np.sum(tri_fA * tri_is_int, axis=1)

    avg_fA = np.zeros(len(tri_rows), dtype=float)
    mask_has_int = interior_counts > 0
    avg_fA[mask_has_int] = sum_fA_int[mask_has_int] / interior_counts[mask_has_int]

    # C[t, k] = tri_fA[t, k] if vertex k is interior, else avg_fA[t]
    C = np.where(tri_is_int, tri_fA, avg_fA[:, None])

    grad_area = np.zeros_like(positions)

    # Now run standard Voronoi gradient logic using C instead of fA
    # This applies to ALL triangles (pure interior and boundary adjacent)

    # Check for obtuse triangles
    is_obtuse = (c0 < 0) | (c1 < 0) | (c2 < 0)
    m_std = ~is_obtuse

    # Standard non-obtuse logic
    if np.any(m_std):
        c0s, c1s, c2s = c0[m_std], c1[m_std], c2[m_std]
        C0s, C1s, C2s = C[m_std, 0], C[m_std, 1], C[m_std, 2]
        e0s, e1s, e2s = e0[m_std], e1[m_std], e2[m_std]
        v0s, v1s, v2s = v0_idxs[m_std], v1_idxs[m_std], v2_idxs[m_std]

        # Length variation
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

        # Cotan variation
        l0sq = np.einsum("ij,ij->i", e0s, e0s)
        l1sq = np.einsum("ij,ij->i", e1s, e1s)
        l2sq = np.einsum("ij,ij->i", e2s, e2s)

        coeff_c0 = 0.125 * l0sq * (C1s + C2s)
        coeff_c1 = 0.125 * l1sq * (C0s + C2s)
        coeff_c2 = 0.125 * l2sq * (C0s + C1s)

        gc0u, gc0v = _grad_cotan(e2s, -e1s)
        gc1u, gc1v = _grad_cotan(e0s, -e2s)
        gc2u, gc2v = _grad_cotan(e1s, -e0s)

        v_c0, v_c1, v_c2 = coeff_c0[:, None], coeff_c1[:, None], coeff_c2[:, None]
        np.add.at(grad_area, v1s, v_c0 * gc0u)
        np.add.at(grad_area, v2s, v_c0 * gc0v)
        np.add.at(grad_area, v0s, v_c0 * -(gc0u + gc0v))
        np.add.at(grad_area, v2s, v_c1 * gc1u)
        np.add.at(grad_area, v0s, v_c1 * gc1v)
        np.add.at(grad_area, v1s, v_c1 * -(gc1u + gc1v))
        np.add.at(grad_area, v0s, v_c2 * gc2u)
        np.add.at(grad_area, v1s, v_c2 * gc2v)
        np.add.at(grad_area, v2s, v_c2 * -(gc2u + gc2v))

    # Obtuse logic
    if np.any(is_obtuse):
        for i, m_sub in enumerate([(c0 < 0), (c1 < 0), (c2 < 0)]):
            m_do = m_sub & is_obtuse
            if np.any(m_do):
                v0o, v1o, v2o = v0_idxs[m_do], v1_idxs[m_do], v2_idxs[m_do]
                # Gradient of Triangle Area
                gT_u, gT_v = grad_triangle_area(
                    positions[v1o] - positions[v0o], positions[v2o] - positions[v0o]
                )

                C0o, C1o, C2o = C[m_do, 0], C[m_do, 1], C[m_do, 2]
                if i == 0:  # Angle at v0 obtuse
                    # A0 = T/2, A1 = T/4, A2 = T/4
                    factor = (0.5 * C0o + 0.25 * C1o + 0.25 * C2o)[:, None]
                elif i == 1:  # Angle at v1 obtuse
                    factor = (0.5 * C1o + 0.25 * C0o + 0.25 * C2o)[:, None]
                else:  # Angle at v2 obtuse
                    factor = (0.5 * C2o + 0.25 * C0o + 0.25 * C1o)[:, None]

                np.add.at(grad_area, v1o, factor * gT_u)
                np.add.at(grad_area, v2o, factor * gT_v)
                np.add.at(grad_area, v0o, factor * -(gT_u + gT_v))

    grad_arr[:] += grad_linear + grad_cot + grad_area

    return total_energy


def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> tuple[float, Dict[int, np.ndarray]]:
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    E = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
    )
    if not compute_gradient:
        return float(E), {}

    grad = {
        int(vid): grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }
    return float(E), grad
