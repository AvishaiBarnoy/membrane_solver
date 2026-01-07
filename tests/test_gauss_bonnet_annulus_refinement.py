import math

import numpy as np
from sample_meshes import square_annulus_mesh

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from modules.energy import gaussian_curvature
from runtime.diagnostics.gauss_bonnet import (
    extract_boundary_loops,
    find_boundary_edges,
    gauss_bonnet_invariant,
)
from runtime.refinement import refine_triangle_mesh


def _sorted_loop_sums(per_loop: dict[int, float]) -> list[float]:
    return sorted(float(v) for v in per_loop.values())


def test_gauss_bonnet_annulus_refinement_invariant():
    """Annulus has χ=0 so total Gauss–Bonnet invariant should stay ~0."""
    mesh = square_annulus_mesh()
    g0, _, _, per0 = gauss_bonnet_invariant(mesh)
    assert abs(g0) < 1e-9
    assert len(per0) == 2

    expected = [-2.0 * math.pi, 2.0 * math.pi]
    assert np.allclose(_sorted_loop_sums(per0), expected, atol=1e-9)

    refined = refine_triangle_mesh(mesh)
    g1, _, _, per1 = gauss_bonnet_invariant(refined)
    assert abs(g1) < 1e-8
    assert len(per1) == 2
    assert np.allclose(_sorted_loop_sums(per1), expected, atol=1e-8)

    boundary_edges = find_boundary_edges(refined)
    loops = extract_boundary_loops(refined, boundary_edges)
    assert len(loops) == 2


def test_gaussian_curvature_energy_annulus_stable_under_refinement():
    mesh = square_annulus_mesh()

    gp = GlobalParameters(
        {
            "gaussian_modulus": 1.0,
            "gaussian_curvature_strict_topology": True,
            "gaussian_curvature_defect_tol": 1e-10,
        }
    )
    resolver = ParameterResolver(gp)

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    e0 = gaussian_curvature.compute_energy_and_gradient_array(
        mesh,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
    )
    assert np.isclose(float(e0), 0.0, atol=1e-9)
    assert float(np.max(np.abs(grad_arr))) == 0.0

    refined = refine_triangle_mesh(mesh)
    positions = refined.positions_view()
    idx_map = refined.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    e1 = gaussian_curvature.compute_energy_and_gradient_array(
        refined,
        gp,
        resolver,
        positions=positions,
        index_map=idx_map,
        grad_arr=grad_arr,
    )
    assert np.isclose(float(e1), 0.0, atol=1e-8)
    assert float(np.max(np.abs(grad_arr))) == 0.0
