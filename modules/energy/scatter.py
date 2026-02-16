"""Shared scatter helpers for energy assembly paths."""

from __future__ import annotations

import numpy as np


def scatter_triangle_scalar_to_vertices(
    *,
    tri_rows: np.ndarray,
    w0: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    n_vertices: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Scatter per-triangle scalar weights to vertex rows.

    Parameters
    ----------
    tri_rows:
        Integer array ``(N_triangles, 3)`` with vertex-row indices.
    w0, w1, w2:
        Arrays ``(N_triangles,)`` with contribution weights for triangle
        columns 0/1/2 respectively.
    n_vertices:
        Number of vertex rows in the destination vector.
    out:
        Optional preallocated output vector ``(n_vertices,)``. When provided,
        it is reset to zero and reused.
    """
    if int(n_vertices) < 0:
        raise ValueError("n_vertices must be >= 0")
    tri_rows = np.asarray(tri_rows)
    if tri_rows.ndim != 2 or tri_rows.shape[1] != 3:
        raise ValueError("tri_rows must have shape (N_triangles, 3)")

    n_tri = tri_rows.shape[0]
    w0 = np.asarray(w0, dtype=float)
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    if w0.shape != (n_tri,) or w1.shape != (n_tri,) or w2.shape != (n_tri,):
        raise ValueError("w0/w1/w2 must each have shape (N_triangles,)")

    if out is None:
        out = np.zeros(int(n_vertices), dtype=float)
    else:
        if out.shape != (int(n_vertices),):
            raise ValueError("out must have shape (n_vertices,)")
        out.fill(0.0)

    if n_tri == 0:
        return out

    out += np.bincount(tri_rows[:, 0], weights=w0, minlength=int(n_vertices))
    out += np.bincount(tri_rows[:, 1], weights=w1, minlength=int(n_vertices))
    out += np.bincount(tri_rows[:, 2], weights=w2, minlength=int(n_vertices))
    return out
