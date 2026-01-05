import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from sample_meshes import square_annulus_mesh

from runtime.diagnostics.gauss_bonnet import (
    extract_boundary_loops,
    find_boundary_edges,
    gauss_bonnet_invariant,
)
from runtime.refinement import refine_triangle_mesh


def _signed_area_xy(loop) -> float:
    coords = np.array([v.position[:2] for v in loop], dtype=float)
    xs = coords[:, 0]
    ys = coords[:, 1]
    xs_next = np.roll(xs, -1)
    ys_next = np.roll(ys, -1)
    return 0.5 * float(np.dot(xs, ys_next) - np.dot(xs_next, ys))


def _loop_area_signs(loops) -> tuple[int, int]:
    areas = [_signed_area_xy(loop) for loop in loops]
    outer_idx = int(np.argmax(np.abs(areas)))
    inner_idx = 1 - outer_idx
    outer_sign = int(np.sign(areas[outer_idx]) or 0)
    inner_sign = int(np.sign(areas[inner_idx]) or 0)
    return outer_sign, inner_sign


def test_gauss_bonnet_annulus_invariant_and_loop_signs() -> None:
    """Annulus: χ=0 so Gauss–Bonnet invariant should cancel to ~0."""
    mesh = square_annulus_mesh()
    g_total, k_int_total, b_total, per_loop = gauss_bonnet_invariant(mesh)

    assert k_int_total == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert b_total == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert g_total == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert len(per_loop) == 2

    vals = sorted(per_loop.values())
    assert vals[0] == pytest.approx(-2.0 * np.pi, rel=0.0, abs=1e-6)
    assert vals[1] == pytest.approx(2.0 * np.pi, rel=0.0, abs=1e-6)


def test_boundary_loop_orientation_and_count_stable_under_refinement() -> None:
    mesh = square_annulus_mesh()
    boundary_edges = find_boundary_edges(mesh)
    loops0 = extract_boundary_loops(mesh, boundary_edges)
    loops1 = extract_boundary_loops(mesh, boundary_edges)

    ids0 = [[v.index for v in loop] for loop in loops0]
    ids1 = [[v.index for v in loop] for loop in loops1]
    assert ids0 == ids1
    assert len(loops0) == 2

    outer_sign0, inner_sign0 = _loop_area_signs(loops0)
    assert outer_sign0 != 0
    assert inner_sign0 != 0

    refined = refine_triangle_mesh(mesh)
    refined_edges = find_boundary_edges(refined)
    refined_loops = extract_boundary_loops(refined, refined_edges)
    assert len(refined_loops) == 2

    outer_sign1, inner_sign1 = _loop_area_signs(refined_loops)
    assert outer_sign1 == outer_sign0
    assert inner_sign1 == inner_sign0

    g0, _, _, per0 = gauss_bonnet_invariant(mesh)
    g1, _, _, per1 = gauss_bonnet_invariant(refined)

    assert g1 == pytest.approx(g0, rel=0.0, abs=1e-5)
    assert sorted(per1.values()) == pytest.approx(sorted(per0.values()), abs=1e-5)
