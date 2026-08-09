import numpy as np

from modules.energy.bt_divergence import (
    _inner_bending_tilt_dE_ddiv,
    _inner_recovered_divergence,
    _inner_recovered_divergence_area_pullback,
    _inner_recovered_divergence_pullback,
)


class MockMesh:
    def __init__(self, n_vertices=4):
        self.vertex_ids = list(range(n_vertices))
        self._vertex_ids_version = 1


def test_inner_recovered_divergence_basic():
    # 4 vertices, 2 triangles
    tri_rows = np.array([[0, 1, 2], [1, 2, 3]])
    tri_area = np.array([3.0, 3.0])  # w = 1.0 for each corner
    div_tri = np.array([10.0, 20.0])
    n_vertices = 4

    # Enable recovered divergence via global_params
    global_params = {"theory_parity_lane": "test"}
    cache_tag = "in"

    div_eval, v_div, v_area = _inner_recovered_divergence(
        global_params=global_params,
        cache_tag=cache_tag,
        tri_rows=tri_rows,
        tri_area=tri_area,
        div_tri=div_tri,
        n_vertices=n_vertices,
        scratch_tag="test",
    )

    # Expected v_area:
    # v0: T0 corner -> 1.0
    # v1: T0, T1 corners -> 1.0 + 1.0 = 2.0
    # v2: T0, T1 corners -> 1.0 + 1.0 = 2.0
    # v3: T1 corner -> 1.0
    assert np.allclose(v_area, [1.0, 2.0, 2.0, 1.0])

    # Expected v_div_num:
    # v0: 1.0 * 10.0 = 10.0
    # v1: 1.0 * 10.0 + 1.0 * 20.0 = 30.0
    # v2: 1.0 * 10.0 + 1.0 * 20.0 = 30.0
    # v3: 1.0 * 20.0 = 20.0

    # Expected v_div: num / area
    # v0: 10.0 / 1.0 = 10.0
    # v1: 30.0 / 2.0 = 15.0
    # v2: 30.0 / 2.0 = 15.0
    # v3: 20.0 / 1.0 = 20.0
    assert np.allclose(v_div, [10.0, 15.0, 15.0, 20.0])

    # Expected div_eval (avg of corner v_div):
    # T0: (10 + 15 + 15) / 3 = 40/3 = 13.333
    # T1: (15 + 15 + 20) / 3 = 50/3 = 16.666
    assert np.allclose(div_eval, [40.0 / 3.0, 50.0 / 3.0])


def test_inner_recovered_divergence_keeps_interface_domains_one_sided():
    tri_rows = np.array([[0, 1, 2], [1, 2, 3]])
    tri_area = np.array([3.0, 3.0])
    div_tri = np.array([10.0, 20.0])
    triangle_domains = np.array([0, 1], dtype=np.int32)

    div_eval, v_div, v_area = _inner_recovered_divergence(
        global_params={"theory_parity_lane": "test"},
        cache_tag="in",
        tri_rows=tri_rows,
        tri_area=tri_area,
        div_tri=div_tri,
        n_vertices=4,
        triangle_domains=triangle_domains,
        scratch_tag="test",
    )

    assert v_div.shape == (2, 4)
    assert v_area.shape == (2, 4)
    assert np.allclose(div_eval, div_tri)
    assert np.allclose(v_div[0, :3], 10.0)
    assert np.allclose(v_div[1, 1:], 20.0)


def test_one_sided_recovery_pullbacks_match_finite_difference():
    tri_rows = np.array([[0, 1, 2], [1, 2, 3], [1, 3, 4]])
    tri_area = np.array([3.0, 4.0, 2.0])
    div_tri = np.array([10.0, 20.0, 14.0])
    triangle_domains = np.array([0, 1, 1], dtype=np.int32)
    coeff_div_eval = np.array([1.2, -0.7, 0.4])
    global_params = {"theory_parity_lane": "test"}

    _, v_div, v_area = _inner_recovered_divergence(
        global_params=global_params,
        cache_tag="in",
        tri_rows=tri_rows,
        tri_area=tri_area,
        div_tri=div_tri,
        n_vertices=5,
        triangle_domains=triangle_domains,
        scratch_tag="test",
    )
    analytic_div = _inner_recovered_divergence_pullback(
        global_params=global_params,
        cache_tag="in",
        tri_rows=tri_rows,
        tri_area=tri_area,
        coeff_div_eval=coeff_div_eval,
        v_area=v_area,
        triangle_domains=triangle_domains,
        scratch_tag="test",
    )
    analytic_area = _inner_recovered_divergence_area_pullback(
        global_params=global_params,
        cache_tag="in",
        tri_rows=tri_rows,
        div_tri=div_tri,
        coeff_div_eval=coeff_div_eval,
        v_div=v_div,
        v_area=v_area,
        triangle_domains=triangle_domains,
        scratch_tag="test",
    )

    def objective(areas, divergences):
        div_eval, _, _ = _inner_recovered_divergence(
            global_params=global_params,
            cache_tag="in",
            tri_rows=tri_rows,
            tri_area=areas,
            div_tri=divergences,
            n_vertices=5,
            triangle_domains=triangle_domains,
            scratch_tag="test",
        )
        return float(np.dot(coeff_div_eval, div_eval))

    eps = 1.0e-6
    finite_div = np.zeros_like(div_tri)
    finite_area = np.zeros_like(tri_area)
    for idx in range(tri_area.size):
        direction = np.zeros_like(tri_area)
        direction[idx] = eps
        finite_area[idx] = (
            objective(tri_area + direction, div_tri)
            - objective(tri_area - direction, div_tri)
        ) / (2.0 * eps)
        finite_div[idx] = (
            objective(tri_area, div_tri + direction)
            - objective(tri_area, div_tri - direction)
        ) / (2.0 * eps)

    assert np.allclose(analytic_div, finite_div, rtol=1.0e-8, atol=1.0e-9)
    assert np.allclose(analytic_area, finite_area, rtol=1.0e-8, atol=1.0e-9)


def test_inner_recovered_divergence_pullback():
    tri_rows = np.array([[0, 1, 2], [1, 2, 3]])
    tri_area = np.array([3.0, 3.0])
    v_area = np.array([1.0, 2.0, 2.0, 1.0])
    coeff_div_eval = np.array([1.0, 1.0])

    global_params = {"theory_parity_lane": "test"}
    cache_tag = "in"

    coeff_div = _inner_recovered_divergence_pullback(
        global_params=global_params,
        cache_tag=cache_tag,
        tri_rows=tri_rows,
        tri_area=tri_area,
        coeff_div_eval=coeff_div_eval,
        v_area=v_area,
        scratch_tag="test",
    )

    # v_grad (sum of coeff/3 per vertex):
    # v0: T0 -> 1/3
    # v1: T0, T1 -> 1/3 + 1/3 = 2/3
    # v2: T0, T1 -> 1/3 + 1/3 = 2/3
    # v3: T1 -> 1/3

    # coeff_div = (tri_area/3) * sum(v_grad * inv_v_area)
    # T0: (3/3) * (v_grad[0]/v_area[0] + v_grad[1]/v_area[1] + v_grad[2]/v_area[2])
    # T0: 1 * ( (1/3)/1.0 + (2/3)/2.0 + (2/3)/2.0 ) = 1 * (1/3 + 1/3 + 1/3) = 1.0
    # T1: 1 * ( (2/3)/2.0 + (2/3)/2.0 + (1/3)/1.0 ) = 1 * (1/3 + 1/3 + 1/3) = 1.0
    assert np.allclose(coeff_div, [1.0, 1.0])


def test_inner_recovered_divergence_area_pullback_matches_finite_difference():
    tri_rows = np.array([[0, 1, 2], [1, 2, 3]])
    tri_area = np.array([3.0, 4.0])
    div_tri = np.array([10.0, 20.0])
    coeff_div_eval = np.array([1.2, -0.7])
    global_params = {"theory_parity_lane": "test"}

    _, v_div, v_area = _inner_recovered_divergence(
        global_params=global_params,
        cache_tag="in",
        tri_rows=tri_rows,
        tri_area=tri_area,
        div_tri=div_tri,
        n_vertices=4,
        scratch_tag="test",
    )
    analytic = _inner_recovered_divergence_area_pullback(
        global_params=global_params,
        cache_tag="in",
        tri_rows=tri_rows,
        div_tri=div_tri,
        coeff_div_eval=coeff_div_eval,
        v_div=v_div,
        v_area=v_area,
        scratch_tag="test",
    )

    def objective(areas):
        div_eval, _, _ = _inner_recovered_divergence(
            global_params=global_params,
            cache_tag="in",
            tri_rows=tri_rows,
            tri_area=areas,
            div_tri=div_tri,
            n_vertices=4,
            scratch_tag="test",
        )
        return float(np.dot(coeff_div_eval, div_eval))

    eps = 1.0e-6
    finite_difference = np.zeros_like(tri_area)
    for idx in range(tri_area.size):
        direction = np.zeros_like(tri_area)
        direction[idx] = eps
        finite_difference[idx] = (
            objective(tri_area + direction) - objective(tri_area - direction)
        ) / (2.0 * eps)

    assert np.allclose(analytic, finite_difference, rtol=1.0e-8, atol=1.0e-9)


def test_inner_bending_tilt_dE_ddiv_off():
    mesh = MockMesh()
    global_params = {"bending_tilt_in_update_mode": "off"}
    cache_tag = "in"
    kappa_tri = np.array([[1.0, 1.0, 1.0]])
    base_tri = np.array([[2.0, 2.0, 2.0]])
    div_term = np.array([0.5])
    va_eff = np.array([1.0, 1.0, 1.0])

    dE, stats = _inner_bending_tilt_dE_ddiv(
        mesh=mesh,
        global_params=global_params,
        cache_tag=cache_tag,
        kappa_tri=kappa_tri,
        base_tri=base_tri,
        div_term=div_term,
        va0_eff=va_eff,
        va1_eff=va_eff,
        va2_eff=va_eff,
    )

    # Expected dE: kappa * (base + div) * va_eff summed over corners
    # T0: 1.0 * (2.0 + 0.5) * 1.0 * 3 = 7.5
    assert np.allclose(dE, [7.5])
    assert stats["enabled"] is False


def test_inner_bending_tilt_dE_ddiv_cross_term_off():
    mesh = MockMesh()
    global_params = {"bending_tilt_in_update_mode": "radial_cross_term_off_v1"}
    cache_tag = "in"
    kappa_tri = np.array([[1.0, 1.0, 1.0]])
    base_tri = np.array([[2.0, 2.0, 2.0]])
    div_term = np.array([0.5])
    va_eff = np.array([1.0, 1.0, 1.0])

    dE, stats = _inner_bending_tilt_dE_ddiv(
        mesh=mesh,
        global_params=global_params,
        cache_tag=cache_tag,
        kappa_tri=kappa_tri,
        base_tri=base_tri,
        div_term=div_term,
        va0_eff=va_eff,
        va1_eff=va_eff,
        va2_eff=va_eff,
    )

    # Expected dE: kappa * div * va_eff summed over corners (base term removed)
    # T0: 1.0 * 0.5 * 1.0 * 3 = 1.5
    assert np.allclose(dE, [1.5])
    assert stats["enabled"] is True
    assert stats["cross_term_removed"] is True
