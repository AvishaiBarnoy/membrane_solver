from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import geometry.tilt_operators as tilt_ops
from geometry.tilt_operators import p1_triangle_divergence, p1_triangle_shape_gradients


def _sample_inputs(*, dtype=np.float64):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        order="C",
    )
    tilts = np.array(
        [
            [0.1, -0.2, 0.0],
            [0.0, 0.4, -0.1],
            [0.3, 0.0, 0.2],
            [-0.2, 0.1, 0.0],
        ],
        dtype=dtype,
        order="C",
    )
    tri_rows = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32, order="C")
    return positions, tilts, tri_rows


def _reference_divergence(
    *, positions: np.ndarray, tilts: np.ndarray, tri_rows: np.ndarray
) -> np.ndarray:
    _area, g0, g1, g2 = p1_triangle_shape_gradients(
        positions=positions, tri_rows=tri_rows
    )
    t0 = tilts[tri_rows[:, 0]]
    t1 = tilts[tri_rows[:, 1]]
    t2 = tilts[tri_rows[:, 2]]
    return (
        np.einsum("ij,ij->i", t0, g0)
        + np.einsum("ij,ij->i", t1, g1)
        + np.einsum("ij,ij->i", t2, g2)
    )


def test_strict_nocopy_rejects_non_fortran_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions, tilts, tri_rows = _sample_inputs()

    called = {"kernel": False}

    def _kernel(*args, **kwargs):
        called["kernel"] = True

    monkeypatch.setenv("MEMBRANE_FORTRAN_STRICT_NOCOPY", "1")
    monkeypatch.setattr(
        tilt_ops,
        "get_tilt_divergence_kernel",
        lambda: type("K", (), {"func": _kernel, "expects_transpose": False})(),
    )

    with pytest.raises(ValueError, match="F-contiguous"):
        p1_triangle_divergence(positions=positions, tilts=tilts, tri_rows=tri_rows)

    assert called["kernel"] is False


def test_strict_nocopy_rejects_wrong_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    positions, tilts, tri_rows = _sample_inputs(dtype=np.float32)

    monkeypatch.setenv("MEMBRANE_FORTRAN_STRICT_NOCOPY", "1")
    monkeypatch.setattr(
        tilt_ops,
        "get_tilt_divergence_kernel",
        lambda: type(
            "K", (), {"func": lambda *args, **kwargs: None, "expects_transpose": False}
        )(),
    )

    with pytest.raises(TypeError, match="float64"):
        p1_triangle_divergence(positions=positions, tilts=tilts, tri_rows=tri_rows)


def test_non_strict_falls_back_to_numpy_when_layout_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions, tilts, tri_rows = _sample_inputs()

    def _kernel(*args, **kwargs):
        raise AssertionError("Fortran kernel should not be called in fallback path")

    monkeypatch.delenv("MEMBRANE_FORTRAN_STRICT_NOCOPY", raising=False)
    monkeypatch.setattr(
        tilt_ops,
        "get_tilt_divergence_kernel",
        lambda: type("K", (), {"func": _kernel, "expects_transpose": False})(),
    )

    div_tri, _area, _g0, _g1, _g2 = p1_triangle_divergence(
        positions=positions, tilts=tilts, tri_rows=tri_rows
    )
    ref = _reference_divergence(positions=positions, tilts=tilts, tri_rows=tri_rows)

    assert np.allclose(div_tri, ref, atol=1e-12, rtol=1e-12)
