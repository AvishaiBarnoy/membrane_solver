from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import modules.energy.bending_math as bending_math


def _laplacian_inputs(*, order: str = "C", tri_dtype=np.int32):
    weights = np.array(
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=np.float64, order=order
    )
    tri_rows = np.array([[0, 1, 2], [0, 2, 3]], dtype=tri_dtype, order=order)
    field = np.array(
        [
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
        ],
        dtype=np.float64,
        order=order,
    )
    return weights, tri_rows, field


def test_bending_laplacian_non_strict_copies_c_contiguous_inputs_to_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights, tri_rows, field = _laplacian_inputs(order="C", tri_dtype=np.int64)
    called = {"kernel": False}

    def _kernel(weights_arg, tri_arg, field_arg, out, zero_based):
        called["kernel"] = True
        assert zero_based == 1
        assert weights_arg.dtype == np.float64
        assert tri_arg.dtype == np.int32
        assert field_arg.dtype == np.float64
        assert weights_arg.flags["F_CONTIGUOUS"]
        assert tri_arg.flags["F_CONTIGUOUS"]
        assert field_arg.flags["F_CONTIGUOUS"]
        out[:] = field_arg + 1.0

    monkeypatch.delenv("MEMBRANE_FORTRAN_STRICT_NOCOPY", raising=False)
    monkeypatch.setattr(
        bending_math,
        "get_bending_laplacian_kernel",
        lambda: SimpleNamespace(func=_kernel, expects_transpose=False),
    )

    out = bending_math._apply_beltrami_laplacian(weights, tri_rows, field)

    assert called["kernel"] is True
    np.testing.assert_allclose(out, field + 1.0)


def test_bending_laplacian_strict_rejects_non_fortran_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights, tri_rows, field = _laplacian_inputs(order="C")
    called = {"kernel": False}

    def _kernel(*args, **kwargs):
        called["kernel"] = True

    monkeypatch.setenv("MEMBRANE_FORTRAN_STRICT_NOCOPY", "1")
    monkeypatch.setattr(
        bending_math,
        "get_bending_laplacian_kernel",
        lambda: SimpleNamespace(func=_kernel, expects_transpose=False),
    )

    with pytest.raises(ValueError, match="F-contiguous weights/tri_rows/field"):
        bending_math._apply_beltrami_laplacian(weights, tri_rows, field)
    assert called["kernel"] is False


def test_bending_laplacian_strict_rejects_wrong_tri_dtype_without_casting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights, _tri_rows, field = _laplacian_inputs(order="F")
    tri_rows = np.asfortranarray(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))
    called = {"kernel": False}

    def _kernel(*args, **kwargs):
        called["kernel"] = True

    monkeypatch.setenv("MEMBRANE_FORTRAN_STRICT_NOCOPY", "1")
    monkeypatch.setattr(
        bending_math,
        "get_bending_laplacian_kernel",
        lambda: SimpleNamespace(func=_kernel, expects_transpose=False),
    )

    with pytest.raises(TypeError, match="int32 tri_rows"):
        bending_math._apply_beltrami_laplacian(weights, tri_rows, field)
    assert called["kernel"] is False


def test_grad_cotan_non_strict_copies_c_contiguous_inputs_to_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    u = np.array([[1.0, 0.0, 0.0], [0.5, 0.2, 0.1]], dtype=np.float64, order="C")
    v = np.array([[0.0, 1.0, 0.0], [0.1, 0.4, 0.3]], dtype=np.float64, order="C")
    called = {"kernel": False}

    def _kernel(u_arg, v_arg, gu, gv):
        called["kernel"] = True
        assert u_arg.dtype == np.float64
        assert v_arg.dtype == np.float64
        assert u_arg.flags["F_CONTIGUOUS"]
        assert v_arg.flags["F_CONTIGUOUS"]
        gu[:] = u_arg + 2.0
        gv[:] = v_arg + 3.0

    monkeypatch.delenv("MEMBRANE_FORTRAN_STRICT_NOCOPY", raising=False)
    monkeypatch.setattr(
        bending_math,
        "get_bending_grad_cotan_kernel",
        lambda: SimpleNamespace(func=_kernel, expects_transpose=False),
    )

    gu, gv = bending_math._grad_cotan(u, v)

    assert called["kernel"] is True
    np.testing.assert_allclose(gu, u + 2.0)
    np.testing.assert_allclose(gv, v + 3.0)


def test_grad_cotan_strict_rejects_non_fortran_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    u = np.array([[1.0, 0.0, 0.0], [0.5, 0.2, 0.1]], dtype=np.float64, order="C")
    v = np.array([[0.0, 1.0, 0.0], [0.1, 0.4, 0.3]], dtype=np.float64, order="C")
    called = {"kernel": False}

    def _kernel(*args, **kwargs):
        called["kernel"] = True

    monkeypatch.setenv("MEMBRANE_FORTRAN_STRICT_NOCOPY", "1")
    monkeypatch.setattr(
        bending_math,
        "get_bending_grad_cotan_kernel",
        lambda: SimpleNamespace(func=_kernel, expects_transpose=False),
    )

    with pytest.raises(ValueError, match="F-contiguous u/v"):
        bending_math._grad_cotan(u, v)
    assert called["kernel"] is False
