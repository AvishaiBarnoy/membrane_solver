from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.energy.scatter import scatter_triangle_scalar_to_vertices


def _reference_add_at(
    *,
    tri_rows: np.ndarray,
    w0: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    n_vertices: int,
) -> np.ndarray:
    out = np.zeros(n_vertices, dtype=float)
    np.add.at(out, tri_rows[:, 0], w0)
    np.add.at(out, tri_rows[:, 1], w1)
    np.add.at(out, tri_rows[:, 2], w2)
    return out


def test_scatter_triangle_scalar_matches_add_at_reference() -> None:
    rng = np.random.default_rng(0)
    n_vertices = 20
    n_tri = 50
    tri_rows = rng.integers(0, n_vertices, size=(n_tri, 3), dtype=np.int32)
    w0 = rng.normal(size=n_tri)
    w1 = rng.normal(size=n_tri)
    w2 = rng.normal(size=n_tri)

    got = scatter_triangle_scalar_to_vertices(
        tri_rows=tri_rows,
        w0=w0,
        w1=w1,
        w2=w2,
        n_vertices=n_vertices,
    )
    ref = _reference_add_at(
        tri_rows=tri_rows,
        w0=w0,
        w1=w1,
        w2=w2,
        n_vertices=n_vertices,
    )
    assert np.allclose(got, ref, atol=1e-12, rtol=1e-12)


def test_scatter_triangle_scalar_reuses_out_buffer() -> None:
    tri_rows = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)
    w0 = np.array([1.0, 2.0])
    w1 = np.array([3.0, 4.0])
    w2 = np.array([5.0, 6.0])
    out = np.full(4, 123.0)

    got = scatter_triangle_scalar_to_vertices(
        tri_rows=tri_rows,
        w0=w0,
        w1=w1,
        w2=w2,
        n_vertices=4,
        out=out,
    )
    assert got is out
    ref = _reference_add_at(tri_rows=tri_rows, w0=w0, w1=w1, w2=w2, n_vertices=4)
    assert np.allclose(got, ref, atol=1e-12, rtol=1e-12)


def test_scatter_triangle_scalar_handles_empty_triangles() -> None:
    tri_rows = np.zeros((0, 3), dtype=np.int32)
    w = np.zeros(0, dtype=float)
    got = scatter_triangle_scalar_to_vertices(
        tri_rows=tri_rows,
        w0=w,
        w1=w,
        w2=w,
        n_vertices=5,
    )
    assert np.allclose(got, np.zeros(5, dtype=float))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_vertices": -1}, "n_vertices"),
        ({"tri_rows": np.array([1, 2, 3])}, "tri_rows"),
        ({"w0": np.array([1.0, 2.0])}, "w0/w1/w2"),
        ({"out": np.zeros(3)}, "out"),
    ],
)
def test_scatter_triangle_scalar_rejects_invalid_inputs(kwargs, match) -> None:
    base = {
        "tri_rows": np.array([[0, 1, 2]], dtype=np.int32),
        "w0": np.array([1.0]),
        "w1": np.array([1.0]),
        "w2": np.array([1.0]),
        "n_vertices": 4,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        scatter_triangle_scalar_to_vertices(**base)
