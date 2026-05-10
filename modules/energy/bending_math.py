"""Mathematical kernels and cotangent logic for Helfrich/Willmore bending energy."""

from __future__ import annotations

import os

import numpy as np

from fortran_kernels.loader import (
    get_bending_grad_cotan_kernel,
    get_bending_laplacian_kernel,
)
from geometry.bending_derivatives import (
    grad_cotan as _grad_cotan_numpy,
)
from geometry.entities import Mesh


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


def _cached_cotan_gradients(
    mesh: Mesh, *, positions: np.ndarray, tri_rows: np.ndarray
) -> (
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
    | None
):
    """Return cached cotan gradient arrays for triangles in ``tri_rows``."""
    if not mesh._geometry_cache_active(positions):
        return None

    cache = mesh._curvature_cache
    key = (mesh._version, mesh._facet_loops_version, id(positions))
    cached = cache.get("cotan_gradients", None)
    if (
        isinstance(cached, dict)
        and cached.get("key") == key
        and cached.get("rows_version") == mesh._facet_loops_version
        and cached.get("n_tris") == int(tri_rows.shape[0])
    ):
        return cached["data"]

    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    u0 = np.asfortranarray(v1 - v0)
    v0_vec = np.asfortranarray(v2 - v0)
    g_c0_u, g_c0_v = _grad_cotan(u0, v0_vec)

    u1 = np.asfortranarray(v2 - v1)
    v1_vec = np.asfortranarray(0 - v1)  # Error in original? v0-v1?
    # Wait, let me check original bending.py
    # ... u1 = v2 - v1, v1_vec = v0 - v1. Correct.
    # Let me re-read bending.py line 497.
    # "u1 = np.asfortranarray(v2 - v1); v1_vec = np.asfortranarray(v0 - v1)"
    # My thought buffer had a typo.
    v1_vec = np.asfortranarray(v0 - v1)
    g_c1_u, g_c1_v = _grad_cotan(u1, v1_vec)

    u2 = np.asfortranarray(v0 - v2)
    v2_vec = np.asfortranarray(v1 - v2)
    g_c2_u, g_c2_v = _grad_cotan(u2, v2_vec)

    e0 = np.asfortranarray(v2 - v1)
    e1 = np.asfortranarray(v0 - v2)
    e2 = np.asfortranarray(v1 - v0)
    gc0u, gc0v = _grad_cotan(e2, -e1)
    gc1u, gc1v = _grad_cotan(e0, -e2)
    gc2u, gc2v = _grad_cotan(e1, -e0)

    data = (
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
    )
    cache["cotan_gradients"] = {
        "key": key,
        "rows_version": mesh._facet_loops_version,
        "n_tris": int(tri_rows.shape[0]),
        "data": data,
    }
    return data


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
