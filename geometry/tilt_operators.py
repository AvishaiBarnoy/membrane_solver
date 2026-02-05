"""Discrete tilt/surface operators.

This module collects vectorized (SoA) helpers for per-vertex *ambient* 3D tilt
vectors stored on a triangle mesh. Tilt vectors are expected to be projected
to vertex tangent planes by the caller (the minimizer enforces this after
updates).

The primary operator needed by the Kozlov–Hamm coupling is the *surface
divergence* of a piecewise-linear (P1) vector field defined on triangle
vertices. For a triangle with vertices ``(v0, v1, v2)`` the gradients of the
P1 basis functions (barycentric coordinates) are:

  ∇φ0 = (n × (v2 - v1)) / |n|²
  ∇φ1 = (n × (v0 - v2)) / |n|²
  ∇φ2 = (n × (v1 - v0)) / |n|²

with ``n = (v1 - v0) × (v2 - v0)`` (non-unit). The P1 divergence on the triangle
is then

  div(t) = Σ_k t_k · ∇φ_k

which is constant over the triangle.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from fortran_kernels.loader import get_tilt_divergence_kernel
from geometry.entities import _fast_cross


def p1_triangle_shape_gradients(
    *,
    positions: np.ndarray,
    tri_rows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return triangle areas and P1 basis gradients.

    Parameters
    ----------
    positions:
        Dense vertex position array of shape ``(N_vertices, 3)``.
    tri_rows:
        Integer array of shape ``(N_triangles, 3)`` with vertex-row indices.

    Returns
    -------
    area:
        Array of shape ``(N_triangles,)`` with triangle areas.
    g0, g1, g2:
        Arrays of shape ``(N_triangles, 3)`` with gradients of the P1 basis
        functions associated with the first/second/third triangle vertex.
    """
    tri_rows = np.asarray(tri_rows, dtype=np.int32)
    if tri_rows.size == 0:
        zeros1 = np.zeros(0, dtype=float)
        zeros3 = np.zeros((0, 3), dtype=float)
        return zeros1, zeros3, zeros3, zeros3

    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    n = _fast_cross(v1 - v0, v2 - v0)
    n2 = np.einsum("ij,ij->i", n, n)
    denom = np.maximum(n2, 1e-20)

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    g0 = _fast_cross(n, e0) / denom[:, None]
    g1 = _fast_cross(n, e1) / denom[:, None]
    g2 = _fast_cross(n, e2) / denom[:, None]

    area = 0.5 * np.sqrt(np.maximum(n2, 0.0))
    return area, g0, g1, g2


def p1_triangle_divergence(
    *,
    positions: np.ndarray,
    tilts: np.ndarray,
    tri_rows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute triangle-wise P1 divergence of a vertex vector field.

    Parameters
    ----------
    positions:
        Dense vertex position array of shape ``(N_vertices, 3)``.
    tilts:
        Dense vertex vector array of shape ``(N_vertices, 3)``.
    tri_rows:
        Integer array of shape ``(N_triangles, 3)`` with vertex-row indices.

    Returns
    -------
    div_tri:
        Array of shape ``(N_triangles,)`` with constant divergence per triangle.
    area:
        Array of shape ``(N_triangles,)`` with triangle areas.
    g0, g1, g2:
        Arrays of shape ``(N_triangles, 3)`` with P1 basis gradients.
    """
    tri_rows = np.asarray(tri_rows, dtype=np.int32)
    if tri_rows.size == 0:
        zeros1 = np.zeros(0, dtype=float)
        zeros3 = np.zeros((0, 3), dtype=float)
        return zeros1, zeros1, zeros3, zeros3, zeros3

    kernel_spec = get_tilt_divergence_kernel()
    if kernel_spec is not None:
        strict = os.environ.get("MEMBRANE_FORTRAN_STRICT_NOCOPY") in {
            "1",
            "true",
            "TRUE",
        }
        if (
            positions.dtype != np.float64
            or tilts.dtype != np.float64
            or tri_rows.dtype != np.int32
        ):
            if strict:
                raise TypeError(
                    "Fortran tilt kernels require float64 positions/tilts and int32 tri_rows."
                )
            kernel_spec = None
        elif not (
            positions.flags["F_CONTIGUOUS"]
            and tilts.flags["F_CONTIGUOUS"]
            and tri_rows.flags["F_CONTIGUOUS"]
        ):
            if strict:
                raise ValueError(
                    "Fortran tilt kernels require F-contiguous positions/tilts/tri_rows (to avoid hidden copies)."
                )
            kernel_spec = None

    if kernel_spec is not None:
        nf = tri_rows.shape[0]
        if kernel_spec.expects_transpose:
            pos_t = positions.T
            tilts_t = tilts.T
            tri_t = tri_rows.T
            div_tri = np.zeros(nf, dtype=np.float64, order="F")
            area = np.zeros(nf, dtype=np.float64, order="F")
            g0 = np.zeros((3, nf), dtype=np.float64, order="F")
            g1 = np.zeros((3, nf), dtype=np.float64, order="F")
            g2 = np.zeros((3, nf), dtype=np.float64, order="F")
            kernel_spec.func(pos_t, tilts_t, tri_t, div_tri, area, g0, g1, g2, 1)
            return (
                np.asarray(div_tri),
                np.asarray(area),
                np.asarray(g0).T,
                np.asarray(g1).T,
                np.asarray(g2).T,
            )

        div_tri = np.zeros(nf, dtype=np.float64, order="F")
        area = np.zeros(nf, dtype=np.float64, order="F")
        g0 = np.zeros((nf, 3), dtype=np.float64, order="F")
        g1 = np.zeros((nf, 3), dtype=np.float64, order="F")
        g2 = np.zeros((nf, 3), dtype=np.float64, order="F")
        kernel_spec.func(positions, tilts, tri_rows, div_tri, area, g0, g1, g2, 1)
        return (
            np.asarray(div_tri),
            np.asarray(area),
            np.asarray(g0),
            np.asarray(g1),
            np.asarray(g2),
        )

    area, g0, g1, g2 = p1_triangle_shape_gradients(
        positions=positions, tri_rows=tri_rows
    )

    t0 = tilts[tri_rows[:, 0]]
    t1 = tilts[tri_rows[:, 1]]
    t2 = tilts[tri_rows[:, 2]]

    div_tri = (
        np.einsum("ij,ij->i", t0, g0)
        + np.einsum("ij,ij->i", t1, g1)
        + np.einsum("ij,ij->i", t2, g2)
    )
    return div_tri, area, g0, g1, g2


def p1_triangle_divergence_from_shape_gradients(
    *,
    tilts: np.ndarray,
    tri_rows: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
) -> np.ndarray:
    """Compute triangle-wise P1 divergence using precomputed basis gradients.

    Parameters
    ----------
    tilts:
        Dense vertex vector array of shape ``(N_vertices, 3)``.
    tri_rows:
        Integer array of shape ``(N_triangles, 3)`` with vertex-row indices.
    g0, g1, g2:
        Arrays of shape ``(N_triangles, 3)`` with P1 basis gradients.

    Returns
    -------
    div_tri:
        Array of shape ``(N_triangles,)`` with constant divergence per triangle.
    """
    tri_rows = np.asarray(tri_rows, dtype=np.int32)
    if tri_rows.size == 0:
        return np.zeros(0, dtype=float)

    t0 = tilts[tri_rows[:, 0]]
    t1 = tilts[tri_rows[:, 1]]
    t2 = tilts[tri_rows[:, 2]]

    return (
        np.einsum("ij,ij->i", t0, g0)
        + np.einsum("ij,ij->i", t1, g1)
        + np.einsum("ij,ij->i", t2, g2)
    )


def p1_vertex_divergence(
    *,
    n_vertices: int,
    positions: np.ndarray,
    tilts: np.ndarray,
    tri_rows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a barycentric-area-averaged vertex divergence field.

    This is a convenience wrapper around :func:`p1_triangle_divergence` that
    averages constant triangle divergence values onto vertices using barycentric
    area weights.

    Returns
    -------
    div_v:
        Array of shape ``(N_vertices,)`` with averaged divergence per vertex.
    area_bary:
        Array of shape ``(N_vertices,)`` with the accumulated barycentric area
        weights used for averaging.
    """
    if n_vertices <= 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    div_tri, area, *_ = p1_triangle_divergence(
        positions=positions, tilts=tilts, tri_rows=tri_rows
    )
    if div_tri.size == 0:
        return np.zeros(n_vertices, dtype=float), np.zeros(n_vertices, dtype=float)

    w = area / 3.0
    accum_div = np.zeros(n_vertices, dtype=float)
    accum_area = np.zeros(n_vertices, dtype=float)
    np.add.at(accum_div, tri_rows[:, 0], w * div_tri)
    np.add.at(accum_div, tri_rows[:, 1], w * div_tri)
    np.add.at(accum_div, tri_rows[:, 2], w * div_tri)
    np.add.at(accum_area, tri_rows[:, 0], w)
    np.add.at(accum_area, tri_rows[:, 1], w)
    np.add.at(accum_area, tri_rows[:, 2], w)

    div_v = np.zeros(n_vertices, dtype=float)
    mask = accum_area > 1e-20
    div_v[mask] = accum_div[mask] / accum_area[mask]
    return div_v, accum_area


__all__ = [
    "p1_triangle_divergence",
    "p1_triangle_shape_gradients",
    "p1_triangle_divergence_from_shape_gradients",
    "p1_vertex_divergence",
]
