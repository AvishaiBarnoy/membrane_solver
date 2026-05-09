import numpy as np

from modules.energy.bt_divergence import (
    _inner_bending_tilt_dE_ddiv,
    _inner_recovered_divergence,
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
