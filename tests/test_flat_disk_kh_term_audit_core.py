from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.refinement import refine_triangle_mesh
from tools.diagnostics.flat_disk_kh_term_audit import (
    _mesh_internal_band_split,
    _mesh_internal_region_split,
    _mesh_internal_triangle_terms,
)
from tools.diagnostics.flat_disk_one_leaflet_theory import (
    compute_flat_disk_kh_physical_theory,
    physical_to_dimensionless_theory_params,
)
from tools.reproduce_flat_disk_one_leaflet import (
    DEFAULT_FIXTURE,
    _build_minimizer,
    _configure_benchmark_mesh,
    _load_mesh_from_fixture,
    _run_theta_relaxation,
)


def _assert_numeric_dict_close(
    observed: dict[str, float],
    expected: dict[str, float],
    *,
    abs_tol: float = 1.0e-12,
) -> None:
    """Assert that two numeric dictionaries match key-for-key within tolerance."""
    assert observed.keys() == expected.keys()
    for key in observed:
        assert float(observed[key]) == pytest.approx(
            float(expected[key]), rel=0.0, abs=abs_tol
        ), key


@lru_cache(maxsize=2)
def _relaxed_flat_kh_mesh(smoothness_model: str):
    """Return a cached relaxed flat KH mesh and theory data for audit-core tests."""
    params = physical_to_dimensionless_theory_params(
        kappa_physical=10.0,
        kappa_t_physical=10.0,
        radius_physical=7.0,
        drive_physical=(2.0 / 0.7),
        length_scale=15.0,
    )
    theory = compute_flat_disk_kh_physical_theory(params)

    mesh = _load_mesh_from_fixture(Path(DEFAULT_FIXTURE))
    mesh = refine_triangle_mesh(mesh)
    _configure_benchmark_mesh(
        mesh,
        theory_params=params,
        parameterization="kh_physical",
        outer_mode="disabled",
        smoothness_model=str(smoothness_model),
        splay_modulus_scale_in=1.0,
        tilt_mass_mode_in="consistent",
    )
    minim = _build_minimizer(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.project_tilts_to_tangent()
    _run_theta_relaxation(minim, theta_value=0.138, reset_outer=True)
    return mesh, theory


@pytest.mark.regression
@pytest.mark.parametrize("smoothness_model", ["dirichlet", "splay_twist"])
def test_flat_disk_kh_triangle_terms_match_region_and_centroid_band_splits(
    smoothness_model: str,
) -> None:
    mesh, theory = _relaxed_flat_kh_mesh(smoothness_model)
    triangle_terms = _mesh_internal_triangle_terms(
        mesh, smoothness_model=smoothness_model
    )

    region_direct = _mesh_internal_region_split(
        mesh,
        smoothness_model=smoothness_model,
        radius=float(theory.radius),
    )
    region_factored = _mesh_internal_region_split(
        mesh,
        smoothness_model=smoothness_model,
        radius=float(theory.radius),
        triangle_terms=triangle_terms,
    )
    _assert_numeric_dict_close(region_factored, region_direct)

    band_direct = _mesh_internal_band_split(
        mesh,
        smoothness_model=smoothness_model,
        radius=float(theory.radius),
        lambda_value=float(theory.lambda_value),
        partition_mode="centroid",
    )
    band_factored = _mesh_internal_band_split(
        mesh,
        smoothness_model=smoothness_model,
        radius=float(theory.radius),
        lambda_value=float(theory.lambda_value),
        partition_mode="centroid",
        triangle_terms=triangle_terms,
    )
    _assert_numeric_dict_close(band_factored, band_direct)

    tri_total = float(np.sum(np.asarray(triangle_terms["internal_tri"], dtype=float)))
    assert float(region_factored["mesh_internal_total_from_regions"]) == pytest.approx(
        tri_total, rel=0.0, abs=1.0e-10
    )
    assert (
        float(band_factored["mesh_internal_disk_core"])
        + float(band_factored["mesh_internal_rim_band"])
        + float(band_factored["mesh_internal_outer_near"])
        + float(band_factored["mesh_internal_outer_far"])
    ) == pytest.approx(tri_total, rel=0.0, abs=1.0e-10)


@pytest.mark.regression
def test_flat_disk_kh_triangle_terms_match_fractional_band_split() -> None:
    mesh, theory = _relaxed_flat_kh_mesh("splay_twist")
    triangle_terms = _mesh_internal_triangle_terms(mesh, smoothness_model="splay_twist")

    band_direct = _mesh_internal_band_split(
        mesh,
        smoothness_model="splay_twist",
        radius=float(theory.radius),
        lambda_value=float(theory.lambda_value),
        partition_mode="fractional",
    )
    band_factored = _mesh_internal_band_split(
        mesh,
        smoothness_model="splay_twist",
        radius=float(theory.radius),
        lambda_value=float(theory.lambda_value),
        partition_mode="fractional",
        triangle_terms=triangle_terms,
    )

    _assert_numeric_dict_close(band_factored, band_direct, abs_tol=1.0e-10)
