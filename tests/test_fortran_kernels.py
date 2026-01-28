import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.bending_derivatives import grad_cotan
from geometry.entities import _fast_cross
from geometry.tilt_operators import p1_triangle_shape_gradients
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
        try:
            gu_out, gv_out = fn(u_in, v_in)
        except Exception:
            gu_out = np.zeros_like(u_in, order="F")
            gv_out = np.zeros_like(v_in, order="F")
            fn(u_in, v_in, gu_out, gv_out)
    else:
        u_in = np.asfortranarray(u, dtype=np.float64)
        v_in = np.asfortranarray(v, dtype=np.float64)
        try:
            gu_out, gv_out = fn(u_in, v_in)
        except Exception:
            gu_out = np.zeros_like(u_in, order="F")
            gv_out = np.zeros_like(v_in, order="F")
            fn(u_in, v_in, gu_out, gv_out)

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

    if expects_transpose:
        out_out = np.zeros((dim, nv), dtype=np.float64, order="F")
        try:
            out_out = fn(weights_in, tri_in, field_in, 1)
        except Exception:
            fn(weights_in, tri_in, field_in, out_out, 1)
    else:
        out_out = np.zeros((nv, dim), dtype=np.float64, order="F")
        try:
            out_out = fn(weights_in, tri_in, field_in, 1)
        except Exception:
            fn(weights_in, tri_in, field_in, out_out, 1)

    out_out = np.asarray(out_out)
    out = out_out.T if expects_transpose else out_out

    assert np.allclose(out, out_ref, atol=1e-10, rtol=1e-10)


def _curvature_data_reference(positions: np.ndarray, tri_rows: np.ndarray):
    n_verts = positions.shape[0]
    if tri_rows.size == 0:
        return (
            np.zeros((n_verts, 3)),
            np.zeros(n_verts),
            np.zeros((0, 3)),
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

    cross = _fast_cross(e1, e2)
    area_doubled = np.linalg.norm(cross, axis=1)
    area_doubled = np.maximum(area_doubled, 1e-12)

    def get_cot(a, b, areas_2):
        return np.einsum("ij,ij->i", a, b) / areas_2

    c0 = get_cot(-e1, e2, area_doubled)
    c1 = get_cot(-e2, e0, area_doubled)
    c2 = get_cot(-e0, e1, area_doubled)

    k_vecs = np.zeros((n_verts, 3), dtype=float, order="F")
    np.add.at(k_vecs, tri_rows[:, 0], 0.5 * (c1[:, None] * -e1 + c2[:, None] * e2))
    np.add.at(k_vecs, tri_rows[:, 1], 0.5 * (c2[:, None] * -e2 + c0[:, None] * e0))
    np.add.at(k_vecs, tri_rows[:, 2], 0.5 * (c0[:, None] * -e0 + c1[:, None] * e1))

    tri_areas = 0.5 * area_doubled
    vertex_areas = np.zeros(n_verts, dtype=float)

    is_obtuse_v0 = c0 < 0
    is_obtuse_v1 = c1 < 0
    is_obtuse_v2 = c2 < 0
    any_obtuse = is_obtuse_v0 | is_obtuse_v1 | is_obtuse_v2

    va0 = np.where(~any_obtuse, (l1_sq * c1 + l2_sq * c2) / 8.0, 0.0)
    va1 = np.where(~any_obtuse, (l2_sq * c2 + l0_sq * c0) / 8.0, 0.0)
    va2 = np.where(~any_obtuse, (l0_sq * c0 + l1_sq * c1) / 8.0, 0.0)

    va0 = np.where(is_obtuse_v0, tri_areas / 2.0, va0)
    va0 = np.where(is_obtuse_v1 | is_obtuse_v2, tri_areas / 4.0, va0)

    va1 = np.where(is_obtuse_v1, tri_areas / 2.0, va1)
    va1 = np.where(is_obtuse_v0 | is_obtuse_v2, tri_areas / 4.0, va1)

    va2 = np.where(is_obtuse_v2, tri_areas / 2.0, va2)
    va2 = np.where(is_obtuse_v0 | is_obtuse_v1, tri_areas / 4.0, va2)

    np.add.at(vertex_areas, tri_rows[:, 0], va0)
    np.add.at(vertex_areas, tri_rows[:, 1], va1)
    np.add.at(vertex_areas, tri_rows[:, 2], va2)

    weights = np.empty((len(tri_rows), 3), dtype=float, order="F")
    weights[:, 0] = c0
    weights[:, 1] = c1
    weights[:, 2] = c2

    return k_vecs, vertex_areas, weights


def test_p1_triangle_divergence_kernel_matches_numpy():
    mod = _import_optional(["fortran_kernels.tilt_kernels", "tilt_kernels"])
    if mod is None:
        pytest.skip("tilt_kernels f2py module not available")

    fn = _get_attr_candidates(
        mod,
        [
            "tilt_kernels_mod.p1_triangle_divergence",
            "p1_triangle_divergence",
        ],
    )
    if fn is None:
        pytest.skip("p1_triangle_divergence not found in tilt_kernels module")

    doc = getattr(fn, "__doc__", "") or ""
    expects_transpose = _expects_transpose_from_doc(doc, expected_first_dim="3")

    rng = np.random.default_rng(999)
    nv = 10
    nf = 7
    positions = rng.normal(size=(nv, 3)).astype(np.float64)
    tilts = rng.normal(size=(nv, 3)).astype(np.float64)
    tri = rng.integers(0, nv, size=(nf, 3), dtype=np.int32)

    area_ref, g0_ref, g1_ref, g2_ref = p1_triangle_shape_gradients(
        positions=positions, tri_rows=tri
    )
    t0 = tilts[tri[:, 0]]
    t1 = tilts[tri[:, 1]]
    t2 = tilts[tri[:, 2]]
    div_ref = (
        np.einsum("ij,ij->i", t0, g0_ref)
        + np.einsum("ij,ij->i", t1, g1_ref)
        + np.einsum("ij,ij->i", t2, g2_ref)
    )

    if expects_transpose:
        pos_in = np.asfortranarray(positions.T, dtype=np.float64)
        tilts_in = np.asfortranarray(tilts.T, dtype=np.float64)
        tri_in = np.asfortranarray(tri.T, dtype=np.int32)
        try:
            div_tri, area, g0, g1, g2 = fn(pos_in, tilts_in, tri_in, 1)
        except Exception:
            div_tri = np.zeros(nf, dtype=np.float64, order="F")
            area = np.zeros(nf, dtype=np.float64, order="F")
            g0 = np.zeros((3, nf), dtype=np.float64, order="F")
            g1 = np.zeros((3, nf), dtype=np.float64, order="F")
            g2 = np.zeros((3, nf), dtype=np.float64, order="F")
            fn(pos_in, tilts_in, tri_in, div_tri, area, g0, g1, g2, 1)
        g0 = np.asarray(g0).T
        g1 = np.asarray(g1).T
        g2 = np.asarray(g2).T
    else:
        pos_in = np.asfortranarray(positions, dtype=np.float64)
        tilts_in = np.asfortranarray(tilts, dtype=np.float64)
        tri_in = np.asfortranarray(tri, dtype=np.int32)
        try:
            div_tri, area, g0, g1, g2 = fn(pos_in, tilts_in, tri_in, 1)
        except Exception:
            div_tri = np.zeros(nf, dtype=np.float64, order="F")
            area = np.zeros(nf, dtype=np.float64, order="F")
            g0 = np.zeros((nf, 3), dtype=np.float64, order="F")
            g1 = np.zeros((nf, 3), dtype=np.float64, order="F")
            g2 = np.zeros((nf, 3), dtype=np.float64, order="F")
            fn(pos_in, tilts_in, tri_in, div_tri, area, g0, g1, g2, 1)

    assert np.allclose(np.asarray(div_tri), div_ref, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.asarray(area), area_ref, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.asarray(g0), g0_ref, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.asarray(g1), g1_ref, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.asarray(g2), g2_ref, atol=1e-10, rtol=1e-10)


def test_curvature_data_kernel_matches_numpy():
    mod = _import_optional(["fortran_kernels.tilt_kernels", "tilt_kernels"])
    if mod is None:
        pytest.skip("tilt_kernels f2py module not available")

    fn = _get_attr_candidates(
        mod,
        [
            "tilt_kernels_mod.compute_curvature_data",
            "compute_curvature_data",
        ],
    )
    if fn is None:
        pytest.skip("compute_curvature_data not found in tilt_kernels module")

    doc = getattr(fn, "__doc__", "") or ""
    expects_transpose = _expects_transpose_from_doc(doc, expected_first_dim="3")

    rng = np.random.default_rng(2024)
    nv = 12
    nf = 9
    positions = rng.normal(size=(nv, 3)).astype(np.float64)
    tri = rng.integers(0, nv, size=(nf, 3), dtype=np.int32)

    k_ref, areas_ref, weights_ref = _curvature_data_reference(positions, tri)

    if expects_transpose:
        pos_in = np.asfortranarray(positions.T, dtype=np.float64)
        tri_in = np.asfortranarray(tri.T, dtype=np.int32)
        try:
            k_vecs, vertex_areas, weights = fn(pos_in, tri_in, 1)
        except Exception:
            k_vecs = np.zeros((3, nv), dtype=np.float64, order="F")
            vertex_areas = np.zeros(nv, dtype=np.float64, order="F")
            weights = np.zeros((3, nf), dtype=np.float64, order="F")
            fn(pos_in, tri_in, k_vecs, vertex_areas, weights, 1)
        k_vecs = np.asarray(k_vecs).T
        weights = np.asarray(weights).T
    else:
        pos_in = np.asfortranarray(positions, dtype=np.float64)
        tri_in = np.asfortranarray(tri, dtype=np.int32)
        try:
            k_vecs, vertex_areas, weights = fn(pos_in, tri_in, 1)
        except Exception:
            k_vecs = np.zeros((nv, 3), dtype=np.float64, order="F")
            vertex_areas = np.zeros(nv, dtype=np.float64, order="F")
            weights = np.zeros((nf, 3), dtype=np.float64, order="F")
            fn(pos_in, tri_in, k_vecs, vertex_areas, weights, 1)

    assert np.allclose(np.asarray(k_vecs), k_ref, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.asarray(vertex_areas), areas_ref, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.asarray(weights), weights_ref, atol=1e-10, rtol=1e-10)
