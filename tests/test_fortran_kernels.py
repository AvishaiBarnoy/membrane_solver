import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.bending_derivatives import grad_cotan
from modules.energy.bending import _apply_beltrami_laplacian


def _import_optional(module_names: list[str]):
    for name in module_names:
        try:
            return __import__(name, fromlist=["*"])
        except Exception:
            continue
    return None


def _get_attr_candidates(mod, dotted_names: list[str]):
    for dotted in dotted_names:
        cur = mod
        ok = True
        for part in dotted.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                ok = False
                break
        if ok and callable(cur):
            return cur
    return None


def _expects_transpose_from_doc(doc: str, *, expected_first_dim: str) -> bool:
    doc = doc or ""
    # f2py wrappers include lines like:
    #   u : input rank-2 array('d') with bounds (n,3)
    # or with bounds (3,n)
    return f"bounds({expected_first_dim}," in doc.replace(" ", "")


@pytest.mark.parametrize("n", [4, 17])
def test_grad_cotan_batch_matches_numpy(n: int):
    mod = _import_optional(["fortran_kernels.bending_kernels", "bending_kernels"])
    if mod is None:
        pytest.skip("bending_kernels f2py module not available")

    fn = _get_attr_candidates(
        mod,
        [
            "bending_kernels_mod.grad_cotan_batch",
            "grad_cotan_batch",
        ],
    )
    if fn is None:
        pytest.skip("grad_cotan_batch not found in bending_kernels module")

    doc = getattr(fn, "__doc__", "") or ""
    expects_transpose = _expects_transpose_from_doc(doc, expected_first_dim="3")

    rng = np.random.default_rng(123)
    u = rng.normal(size=(n, 3))
    v = rng.normal(size=(n, 3))
    # Avoid near-colinear vectors (S -> 0) so both implementations return nonzero grads.
    v += 0.3 * rng.normal(size=(n, 3))

    gu_ref, gv_ref = grad_cotan(u, v)

    if expects_transpose:
        u_in = np.asfortranarray(u.T, dtype=np.float64)
        v_in = np.asfortranarray(v.T, dtype=np.float64)
    else:
        u_in = np.asfortranarray(u, dtype=np.float64)
        v_in = np.asfortranarray(v, dtype=np.float64)

    # f2py may expose either:
    #  - (grad_u, grad_v) = fn(u, v, [n])
    #  - fn(n, u, v, grad_u, grad_v)
    try:
        gu_out, gv_out = fn(u_in, v_in)
    except TypeError:
        gu_out = np.zeros_like(u_in, order="F")
        gv_out = np.zeros_like(v_in, order="F")
        fn(n, u_in, v_in, gu_out, gv_out)

    if expects_transpose:
        gu = np.asarray(gu_out).T
        gv = np.asarray(gv_out).T
    else:
        gu = np.asarray(gu_out)
        gv = np.asarray(gv_out)

    assert np.allclose(gu, gu_ref, atol=1e-10, rtol=1e-10)
    assert np.allclose(gv, gv_ref, atol=1e-10, rtol=1e-10)


def test_apply_beltrami_laplacian_matches_numpy():
    mod = _import_optional(["fortran_kernels.bending_kernels", "bending_kernels"])
    if mod is None:
        pytest.skip("bending_kernels f2py module not available")

    fn = _get_attr_candidates(
        mod,
        [
            "bending_kernels_mod.apply_beltrami_laplacian",
            "apply_beltrami_laplacian",
        ],
    )
    if fn is None:
        pytest.skip("apply_beltrami_laplacian not found in bending_kernels module")

    doc = getattr(fn, "__doc__", "") or ""
    expects_transpose = _expects_transpose_from_doc(doc, expected_first_dim="3")

    nv = 9
    nf = 5
    dim = 3
    rng = np.random.default_rng(456)

    tri = rng.integers(0, nv, size=(nf, 3), dtype=np.int32)
    weights = rng.normal(size=(nf, 3)).astype(np.float64)
    field = rng.normal(size=(nv, dim)).astype(np.float64)

    out_ref = _apply_beltrami_laplacian(weights, tri, field)

    if expects_transpose:
        tri_in = np.asfortranarray(tri.T, dtype=np.int32)
        weights_in = np.asfortranarray(weights.T, dtype=np.float64)
        field_in = np.asfortranarray(field.T, dtype=np.float64)
    else:
        tri_in = np.asfortranarray(tri, dtype=np.int32)
        weights_in = np.asfortranarray(weights, dtype=np.float64)
        field_in = np.asfortranarray(field, dtype=np.float64)

    # f2py may expose either:
    #  - out = fn(weights, tri, field, zero_based, [dim, nv, nf])
    #  - fn(dim, nv, nf, weights, tri, field, out, zero_based)
    try:
        out_out = fn(weights_in, tri_in, field_in, 1)
    except TypeError:
        if expects_transpose:
            out_out = np.zeros((dim, nv), dtype=np.float64, order="F")
            fn(dim, nv, nf, weights_in, tri_in, field_in, out_out, 1)
        else:
            out_out = np.zeros((nv, dim), dtype=np.float64, order="F")
            fn(dim, nv, nf, weights_in, tri_in, field_in, out_out, 1)

    out_out = np.asarray(out_out)
    out = out_out.T if expects_transpose else out_out

    assert np.allclose(out, out_ref, atol=1e-10, rtol=1e-10)
